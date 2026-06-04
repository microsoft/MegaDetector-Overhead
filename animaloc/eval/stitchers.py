__copyright__ = """
    Copyright (C) 2024 University of Liège, Gembloux Agro-Bio Tech, Forest Is Life
    All rights reserved.

    This source code is under the MIT License.

    Please contact the author Alexandre Delplanque (alexandre.delplanque@uliege.be) for any questions.

    Last modification: March 18, 2024
    """
__author__ = "Alexandre Delplanque"
__license__ = "MIT License"
__version__ = "0.2.1"


import torch
import torchvision

import torch.nn.functional as F
import numpy as np

from typing import List, Tuple
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

from .utils import HannWindow2D

from ..data import ImageToPatches

from ..utils.registry import Registry

STITCHERS = Registry("stitchers", module_key="animaloc.eval.stitchers")

__all__ = ["STITCHERS", *STITCHERS.registry_names]


@STITCHERS.register()
class Stitcher(ImageToPatches):
    """Class to stitch detections of patches into original image
    coordinates system

    This algorithm works as follow:
        1) Cut original image into patches
        2) Make inference on each patches and harvest the detections
        3) Patch the detections maps into the coordinate system of the original image
        Optional:
        4) Upsample the patched detection map
    """

    def __init__(
        self,
        model: torch.nn.Module,
        size: Tuple[int, int],
        overlap: int = 100,
        batch_size: int = 1,
        down_ratio: int = 1,
        up: bool = False,
        reduction: str = "sum",
        device_name: str = "cuda",
    ) -> None:
        """
        Args:
            model (torch.nn.Module): CNN detection model, that takes as inputs image and returns
                output and dict (i.e. wrapped by LossWrapper)
            size (tuple): patches size (height, width), in pixels
            overlap (int, optional): overlap between patches, in pixels.
                Defaults to 100.
            batch_size (int, optional): batch size used for inference over patches.
                Defaults to 1.
            down_ratio (int, optional): downsample ratio. Set to 1 to get output of the same
                size as input (i.e. no downsample). Defaults to 1.
            up (bool, optional): set to True to upsample the patched map. Defaults to False.
            reduction (str, optional): specifies the reduction to apply on overlapping areas.
                Possible values are 'sum', 'mean', 'max'. Defaults to 'sum'.
            device_name (str, optional): the device name on which tensors will be allocated
                ('cpu' or 'cuda'). Defaults to 'cuda'.
        """

        assert isinstance(model, torch.nn.Module), (
            "model argument must be an instance of nn.Module()"
        )

        assert reduction in ["sum", "mean", "max"], (
            "reduction argument possible values are 'sum', 'mean' and 'max' "
            f"got '{reduction}'"
        )

        self.model = model
        self.size = size
        self.overlap = overlap
        self.batch_size = batch_size
        self.down_ratio = down_ratio
        self.up = up
        self.reduction = reduction
        self.device = torch.device(device_name)

        self.model.to(self.device)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Apply the stitching algorithm to the image

        Args:
            image (torch.Tensor): image of shape [C,H,W]

        Returns:
            torch.Tensor
                the detections into the coordinate system of the original image
        """

        # 2026-04-27: pad if either spatial dim is below the patch size in
        # self.size. Without this, make_patches uses min(image_dim, patch_dim)
        # for any small dim, producing non-square patches whose dims aren't
        # divisible by DLA's 32x cumulative stride — model crashes with
        # "tensor a (k) must match tensor b (k-1) at non-singleton dim".
        # Pad with zeros at bottom/right; track original dims so we can crop
        # the stitched heatmap back to the un-padded coordinate system.
        _, orig_h, orig_w = image.shape
        pad_h = max(0, self.size[0] - orig_h)
        pad_w = max(0, self.size[1] - orig_w)
        if pad_h > 0 or pad_w > 0:
            # F.pad spec for last-two dims is (left, right, top, bottom)
            image = F.pad(image, (0, pad_w, 0, pad_h), mode="constant", value=0)
        self._orig_h = orig_h
        self._orig_w = orig_w

        super(Stitcher, self).__init__(image, self.size, self.overlap)

        self.image = image.to(torch.device("cpu"))

        # OPTIMIZATION: Check if image size <= patch size to avoid unnecessary patching
        if self.image.size(1) <= self.size[0] and self.image.size(2) <= self.size[1]:
            # Process the entire image as a single patch
            patched_map = self._process_single_image(image)
        else:
            # ADAPTIVE OPTIMIZATION: Reduce overlap if it's creating too much redundancy
            if self._would_create_excessive_patches():
                print(
                    f"Warning: Current settings would create {self._estimate_patch_count()} patches. "
                    f"Consider reducing overlap from {self.overlap} for better efficiency."
                )

            # step 1 - get patches and limits
            patches = self.make_patches()

            # step 2 - inference to get maps
            det_maps = self._inference(patches)

            # step 3 - patch the maps into initial coordinates system
            patched_map = self._patch_maps(det_maps)
            patched_map = self._reduce(patched_map)

            # (step 4 - upsample)
            if self.up:
                patched_map = F.interpolate(
                    patched_map,
                    scale_factor=self.down_ratio,
                    mode="bilinear",
                    align_corners=True,
                )

        # If we padded, crop the heatmap back to the un-padded coordinate system.
        # patched_map is at heatmap scale (orig_h // down_ratio when up=False)
        # or image scale (orig_h when up=True).
        if pad_h > 0 or pad_w > 0:
            if self.up:
                crop_h, crop_w = orig_h, orig_w
            else:
                crop_h = orig_h // self.down_ratio
                crop_w = orig_w // self.down_ratio
            patched_map = patched_map[:, :, :crop_h, :crop_w]

        return patched_map

    def _process_single_image(self, image: torch.Tensor) -> torch.Tensor:
        """Process a single image without patching when image size <= patch size

        Args:
            image (torch.Tensor): image of shape [C,H,W]

        Returns:
            torch.Tensor: the detections for the entire image
        """
        # Add batch dimension and process directly
        image_batch = image.unsqueeze(0).to(self.device)

        with torch.no_grad():
            self.model.eval()
            outputs, _ = self.model(image_batch)
            if isinstance(outputs, tuple):
                outputs = outputs[0]

        # Remove batch dimension and apply upsampling if needed
        output_map = outputs.squeeze(0).unsqueeze(0)  # Keep compatible shape [1,C,H,W]

        if self.up:
            output_map = F.interpolate(
                output_map,
                scale_factor=self.down_ratio,
                mode="bilinear",
                align_corners=True,
            )

        return output_map

    def _would_create_excessive_patches(self) -> bool:
        """Check if current settings would create excessive patches"""
        estimated_patches = self._estimate_patch_count()
        # Consider excessive if more than 4 patches for an image close to patch size
        image_area = self.image.size(1) * self.image.size(2)
        patch_area = self.size[0] * self.size[1]
        area_ratio = image_area / patch_area

        return estimated_patches > 4 and area_ratio < 2.0

    def _estimate_patch_count(self) -> int:
        """Estimate number of patches that would be created"""
        h, w = self.image.size(1), self.image.size(2)
        patch_h, patch_w = self.size[0], self.size[1]

        stride_h = patch_h - self.overlap
        stride_w = patch_w - self.overlap

        n_rows = (h - patch_h) // stride_h + 1
        n_cols = (w - patch_w) // stride_w + 1

        # Add residual patches
        if (h - patch_h) % stride_h != 0:
            n_rows += 1
        if (w - patch_w) % stride_w != 0:
            n_cols += 1

        return max(1, n_rows * n_cols)

    @torch.no_grad()
    def _inference(self, patches: torch.Tensor) -> List[torch.Tensor]:

        self.model.eval()

        dataset = TensorDataset(patches)
        dataloader = DataLoader(
            dataset, batch_size=self.batch_size, sampler=SequentialSampler(dataset)
        )

        maps = []
        for patch in dataloader:
            patch = patch[0].to(self.device)
            outputs, _ = self.model(patch)
            maps = [*maps, *outputs.unsqueeze(0)]

        return maps

    def _patch_maps(self, maps: List[torch.Tensor]) -> torch.Tensor:

        _, h, w = self.image.shape
        dh, dw = h // self.down_ratio, w // self.down_ratio
        kernel_size = np.array(self.size) // self.down_ratio
        stride = kernel_size - self.overlap // self.down_ratio
        output_size = (
            self._ncol * kernel_size[0]
            - ((self._ncol - 1) * self.overlap // self.down_ratio),
            self._nrow * kernel_size[1]
            - ((self._nrow - 1) * self.overlap // self.down_ratio),
        )

        maps = torch.cat(maps, dim=0)

        if self.reduction == "max":
            out_map = self._max_fold(
                maps,
                output_size=output_size,
                kernel_size=tuple(kernel_size),
                stride=tuple(stride),
            )
        else:
            n_patches = maps.shape[0]
            maps = maps.permute(1, 2, 3, 0).contiguous().view(1, -1, n_patches)
            out_map = F.fold(
                maps,
                output_size=output_size,
                kernel_size=tuple(kernel_size),
                stride=tuple(stride),
            )

        out_map = out_map[:, :, 0:dh, 0:dw]

        return out_map

    def _reduce(self, map: torch.Tensor) -> torch.Tensor:

        dh = self.image.shape[1] // self.down_ratio
        dw = self.image.shape[2] // self.down_ratio
        ones = torch.ones(self.image.shape[0], dh, dw)

        if self.reduction == "mean":
            ones_patches = ImageToPatches(
                ones,
                np.array(self.size) // self.down_ratio,
                self.overlap // self.down_ratio,
            ).make_patches()

            ones_patches = [
                p.unsqueeze(0).unsqueeze(0) for p in ones_patches[:, 1, :, :]
            ]
            norm_map = self._patch_maps(ones_patches)

        else:
            norm_map = ones[1, :, :]

        return torch.div(map.to(self.device), norm_map.to(self.device))

    def _max_fold(
        self, maps: torch.Tensor, output_size: tuple, kernel_size: tuple, stride: tuple
    ) -> torch.Tensor:

        output = torch.zeros((1, maps.shape[1], *output_size))

        fn = lambda x: [
            [i, i + kernel_size[x]] for i in range(0, output_size[x], stride[x])
        ][:-1]
        locs = [[*h, *w] for h in fn(0) for w in fn(1)]

        for loc, m in zip(locs, maps):
            patch = torch.zeros(output.shape)
            patch[:, :, loc[0] : loc[1], loc[2] : loc[3]] = m
            output = torch.max(output, patch)

        return output


@STITCHERS.register()
class HerdNetStitcher(Stitcher):
    @torch.no_grad()
    def _inference(self, patches: torch.Tensor) -> List[torch.Tensor]:

        self.model.eval()

        dataset = TensorDataset(patches)
        dataloader = DataLoader(
            dataset, batch_size=self.batch_size, sampler=SequentialSampler(dataset)
        )

        maps = []
        for patch in dataloader:
            patch = patch[0].to(self.device)
            outputs = self.model(patch)[0]
            heatmap = outputs[0]
            scale_factor = 16
            clsmap = F.interpolate(
                outputs[1], scale_factor=scale_factor, mode="nearest"
            )
            # cat
            outmaps = torch.cat([heatmap, clsmap], dim=1)
            maps = [*maps, *outmaps.unsqueeze(0)]

        return maps


@STITCHERS.register()
class HerdNet_Detection_Branch_Stitcher(Stitcher):
    @torch.no_grad()
    def _inference(self, patches: torch.Tensor) -> List[torch.Tensor]:

        self.model.eval()

        dataset = TensorDataset(patches)
        dataloader = DataLoader(
            dataset, batch_size=self.batch_size, sampler=SequentialSampler(dataset)
        )

        maps = []
        for patch in dataloader:
            patch = patch[0].to(self.device)
            outputs = self.model(patch)[0]
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            heatmap = outputs
            outmaps = heatmap.unsqueeze(0)
            maps = [*maps, *outmaps]

        return maps


@STITCHERS.register()
class HerdNet_Detection_Count_Branch_Stitcher(Stitcher):
    def _pad_to_multiple(x, mult=32):
        _, _, h, w = x.shape
        new_h = ((h + mult - 1) // mult) * mult
        new_w = ((w + mult - 1) // mult) * mult
        pad_h = new_h - h
        pad_w = new_w - w
        return F.pad(x, (0, pad_w, 0, pad_h)), (h, w)

    def _crop_to(x, size_hw):
        h, w = size_hw
        return x[..., :h, :w]

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Apply the stitching algorithm to the image

        Args:
            image (torch.Tensor): image of shape [C,H,W]

        Returns:
            2-tuple of torch.Tensor
                1) the detections into the coordinate system of the original image with shape [1,1,H,W]
                2) the counts into the coordinate system of the original image with shape [1,1]
        """

        super(Stitcher, self).__init__(image, self.size, self.overlap)

        self.image = image.to(torch.device("cpu"))

        # step 1 - get patches and limits
        patches = self.make_patches()

        # step 2 - inference to get maps and counts
        det_maps, counts = self._inference(patches)

        # step 3 - patch the maps into initial coordinates system
        patched_map = self._patch_maps(det_maps)
        patched_map = self._reduce(patched_map)

        # (step 4 - upsample)
        if self.up:
            patched_map = F.interpolate(
                patched_map,
                scale_factor=self.down_ratio,
                mode="bilinear",
                align_corners=True,
            )

        aggregated_count = self._aggregate_counts(counts)  # single value
        aggregated_count = aggregated_count.view(1, 1)

        return patched_map, aggregated_count

    @torch.no_grad()
    def _inference(self, patches: torch.Tensor) -> List[torch.Tensor]:
        self.model.eval()

        dataset = TensorDataset(patches)
        dataloader = DataLoader(
            dataset, batch_size=self.batch_size, sampler=SequentialSampler(dataset)
        )

        maps = []
        counts = []
        for patch in dataloader:
            patch = patch[0].to(self.device)
            patch_padded, original_size = self._pad_to_multiple(patch, mult=32)
            outputs = self.model(patch_padded)[0]
            outputs = self._crop_to(outputs, original_size)
            # outputs = self.model(patch)[0] TODO: Check if padding is needed
            heatmap, count = outputs  # heatmap [B, 1, H, W], counts [B, 1]
            maps.append(heatmap)
            counts.append(count)
        return maps, counts

    def _aggregate_counts(self, counts: List[torch.Tensor]) -> torch.Tensor:
        """
        Aggregate the counts from different patches, taking into account the overlap.

        Args:
            counts (List[torch.Tensor]): List of count tensors, each of shape [B, 1].

        Returns:
            torch.Tensor: The aggregated count tensor of shape [1, 1].
        """
        # Initialize the count for the entire image
        total_count = torch.zeros(1, 1, device=self.device)

        # Get the number of rows and columns of patches
        n_rows, n_cols = self._nrow, self._ncol

        # Calculate the size of each patch
        patch_height, patch_width = self.size

        # Create a matrix to keep track of the overlap count
        overlap_matrix = torch.zeros(n_rows, n_cols, device=self.device)

        # Fill the overlap matrix with the number of overlapping patches for each section
        for row in range(n_rows):
            for col in range(n_cols):
                # Determine the number of overlapping patches for the current section
                overlap_count = 1
                if row > 0 and row < n_rows - 1:
                    overlap_count *= 2
                if col > 0 and col < n_cols - 1:
                    overlap_count *= 2
                overlap_matrix[row, col] = overlap_count

        # Flatten the overlap matrix to match the counts list
        overlap_matrix_flat = overlap_matrix.view(-1)

        # Distribute the counts proportionally based on the overlap
        for count, overlap_count in zip(counts, overlap_matrix_flat):
            total_count += count / overlap_count

        return total_count


@STITCHERS.register()
class FasterRCNNStitcher(Stitcher):
    def __init__(
        self,
        model: torch.nn.Module,
        size: Tuple[int, int],
        overlap: int = 100,
        nms_threshold: float = 0.5,
        score_threshold: float = 0.0,
        batch_size: int = 1,
        device_name: str = "cuda",
    ) -> None:
        super().__init__(
            model, size, overlap=overlap, batch_size=batch_size, device_name=device_name
        )

        self.nms_threshold = nms_threshold
        self.score_threshold = score_threshold
        self.up = False

    @torch.no_grad()
    def _inference(self, patches: torch.Tensor) -> List[dict]:

        self.model.eval()
        dataset = TensorDataset(patches)
        dataloader = DataLoader(
            dataset, batch_size=self.batch_size, sampler=SequentialSampler(dataset)
        )

        maps = []
        for patch in dataloader:
            patch = patch[0].to(self.device)
            outputs, _ = self.model(patch)
            maps.append(*outputs)

        return maps

    def _patch_maps(self, maps: List[dict]) -> dict:
        boxes, labels, scores = [], [], []
        for map, limit in zip(maps, self.get_limits().values()):
            for box in map["boxes"].tolist():
                x1, y1, x2, y2 = box
                new_box = [
                    x1 + limit.x_min,
                    y1 + limit.y_min,
                    x2 + limit.x_min,
                    y2 + limit.y_min,
                ]
                boxes = [*boxes, new_box]

            labels = [*labels, *map["labels"].tolist()]
            scores = [*scores, *map["scores"].tolist()]

        return dict(
            boxes=torch.Tensor(boxes),
            labels=torch.Tensor(labels),
            scores=torch.Tensor(scores),
        )

    def _reduce(self, map: dict) -> dict:
        if map["boxes"].nelement() == 0:
            return map
        else:
            indices = torchvision.ops.nms(
                map["boxes"], map["scores"], self.nms_threshold
            )
            reduced = dict(
                boxes=map["boxes"][indices],
                labels=map["labels"][indices],
                scores=map["scores"][indices],
            )
            # score thresholding
            indices = torch.nonzero(
                (reduced["scores"] > self.score_threshold), as_tuple=True
            )[0]
            reduced = dict(
                boxes=reduced["boxes"][indices],
                labels=reduced["labels"][indices],
                scores=reduced["scores"][indices],
            )

            return reduced


@STITCHERS.register()
class DensityMapStitcher(Stitcher):
    def __init__(
        self,
        model: torch.nn.Module,
        size: Tuple[int, int],
        overlap: int = 100,
        batch_size: int = 1,
        down_ratio: int = 2,
        adapt_ts: float = 0.0,
        reduction: str = "mean",
        device_name: str = "cuda",
    ) -> None:
        super().__init__(
            model,
            size,
            overlap=overlap,
            batch_size=batch_size,
            down_ratio=down_ratio,
            reduction=reduction,
            device_name=device_name,
        )

        self.adapt_ts = adapt_ts
        self.up = False

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Apply the stitching algorithm to the image

        Args:
            image (torch.Tensor): image of shape [C,H,W]

        Returns:
            torch.Tensor
                the detections into the coordinate system of the original image
        """

        patched_map = super(DensityMapStitcher, self).__call__(image)

        B, C, H, W = patched_map.shape

        # thresholding
        max_values = patched_map.max(3)[0].max(2)[0]
        thresholds = (max_values * self.adapt_ts).repeat(B, H, W, 1).permute(0, 3, 1, 2)
        patched_map = patched_map * (patched_map > thresholds).float()
        # outputs = F.threshold(outputs, self.adapt_ts * max_value, 0.0)

        return patched_map

    @torch.no_grad()
    def _inference(self, patches: torch.Tensor) -> List[torch.Tensor]:

        self.model.eval()

        dataset = TensorDataset(patches)
        dataloader = DataLoader(
            dataset, batch_size=self.batch_size, sampler=SequentialSampler(dataset)
        )

        # 2D Hann windows matrix
        self.hann_matrix = self._make_hann_matrix()
        if len(patches) == 1:
            hann = HannWindow2D(size=self.size[0] // self.down_ratio)
            self.hann_matrix = [hann.get_window("original", "up")]

        maps = []
        for patch, hann_2D in zip(dataloader, self.hann_matrix):
            patch = patch[0].to(self.device)
            outputs, _ = self.model(patch)

            # hann filter
            outputs = outputs * hann_2D.to(outputs.device)

            maps = [*maps, *outputs.unsqueeze(0)]

        return maps

    def _make_hann_matrix(self) -> list:

        hann = HannWindow2D(size=self.size[0] // self.down_ratio)

        first_row = [hann.get_window("edge", "up")] * self._nrow
        first_row[0] = hann.get_window("corner", "up_left")
        first_row[-1] = hann.get_window("corner", "up_right")

        middle_row = [hann.get_window("original", "up")] * self._nrow
        middle_row[0] = hann.get_window("edge", "left")
        middle_row[-1] = hann.get_window("edge", "right")

        last_row = [hann.get_window("edge", "down")] * self._nrow
        last_row[0] = hann.get_window("corner", "down_left")
        last_row[-1] = hann.get_window("corner", "down_right")

        matrix = [*first_row, *middle_row * (self._ncol - 2), *last_row]

        return matrix
