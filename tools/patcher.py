__copyright__ = \
    """
    Copyright (C) 2024 University of Liège, Gembloux Agro-Bio Tech, Forest Is Life
    All rights reserved.

    This source code is under the MIT License.

    Please contact the author Alexandre Delplanque (alexandre.delplanque@uliege.be) for any questions.

    Last modification: April 2025 by Daniel Chacon
    """
__author__ = "Alexandre Delplanque"
__license__ = "MIT License"
__version__ = "0.2.1"


import argparse
import os
import PIL
import torchvision
import numpy
import cv2
import pandas
import random
import torch

from albumentations import PadIfNeeded

from tqdm import tqdm

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from animaloc.data import ImageToPatches, PatchesBuffer, save_batch_images, save_batch_images_with_indices

parser = argparse.ArgumentParser(prog='patcher', description='Cut images into patches')

parser.add_argument('root', type=str,
    help='path to the images directory (str)')
parser.add_argument('height', type=int,
    help='height of the patches, in pixels (int)')
parser.add_argument('width', type=int,
    help='width of the patches, in pixels (int)')
parser.add_argument('overlap', type=int,
    help='overlap between patches, in pixels (int)')
parser.add_argument('dest', type=str,
    help='destination path (str)')
parser.add_argument('-csv', type=str,
    help='path to a csv file containing annotations (str). Defaults to None')
parser.add_argument('-min', type=float, default=0.25,
    help='minimum fraction of area for an annotation to be kept (float). Defautls to 0.1')
parser.add_argument('-only_csv', action='store_true', 
    help='if set, only keep images that are in the csv (bool)', default=True)
parser.add_argument('-keep_unannotated', type=float, default=0.0,
    help='keeps a fraction of unannotated (float). Defaults to 0.0')
parser.add_argument('-single_patch', action='store_true',
    help='If set, only extract the top-left patch without additional patches. Defaults to False.')

args = parser.parse_args()
# Seed for reproducibility
random.seed(42)

def main():

    if not os.path.exists(args.dest):
        os.makedirs(args.dest)
    possible_extensions = ['.tif', '.jpg', '.png', '.jpeg', '.TIF', '.JPG', '.PNG', '.JPEG']
    images_paths = [os.path.join(args.root, p) for p in os.listdir(args.root) if any([p.endswith(ext) for ext in possible_extensions])]

    if args.csv is not None:
        patches_buffer = PatchesBuffer(args.csv, args.root, (args.height, args.width), overlap=args.overlap, min_visibility=args.min).buffer
        if args.single_patch:
            patches_buffer = patches_buffer[patches_buffer['images'].str.endswith(tuple(f'_0{ext}' for ext in possible_extensions))] # only keep the top-left patches
        patches_buffer.drop(columns='limits').to_csv(os.path.join(args.dest, 'gt.csv'), index=False)
        #if not args.only_csv: # only keep images that are in the csv
        images_paths = [os.path.join(args.root, x) for x in pandas.read_csv(args.csv)['images'].unique()]      
    
    padder = PadIfNeeded(
                args.height, args.width,
                position=PadIfNeeded.PositionType.TOP_LEFT,
                border_mode=cv2.BORDER_CONSTANT,
                value=0
            )

    for img_path in tqdm(images_paths, desc='Exporting patches'):
        pil_img = PIL.Image.open(img_path)
        img_tensor = torchvision.transforms.ToTensor()(pil_img)
        img_name = os.path.basename(img_path)

        if args.single_patch:
            # Extract only the top-left patch
            top_left_patch = img_tensor[:, :args.height, :args.width]  # Crop the top-left corner
            padded_patch = padder(image=numpy.array(top_left_patch.permute(1, 2, 0) * 255, dtype=numpy.uint8))['image']
            padded_patch = PIL.Image.fromarray(padded_patch)
            patch_name = f"{os.path.splitext(img_name)[0]}_0.{os.path.splitext(img_name)[1].lstrip('.')}"
            padded_patch.save(os.path.join(args.dest, patch_name))
            continue

        if args.csv is not None:
            # Save annotated patches
            img_ptch_df = patches_buffer[patches_buffer['base_images'].apply(os.path.basename) == img_name]
            for row in img_ptch_df[['images', 'limits']].to_numpy().tolist():
                ptch_name, limits = row[0], row[1]
                ptch_name = os.path.basename(ptch_name) 
                cropped_img = numpy.array(pil_img.crop(limits.get_tuple))
                padded_img = PIL.Image.fromarray(padder(image=cropped_img)['image'])
                padded_img.save(os.path.join(args.dest, ptch_name))

            # Save a percentage of unannotated patches
            if args.keep_unannotated > 0.0:
                all_patches = ImageToPatches(img_tensor, (args.height, args.width), overlap=args.overlap).make_patches() # Tensor with shape (N, C, H, W) with N the number of patches
                annotated_patches = img_ptch_df['images'].unique().tolist()
                annotated_patches_indices = sorted([int(p.split('_')[-1].split('.')[0]) for p in annotated_patches]) # TODO: See if sorting is needed
                unannotated_patches_indices = [i for i in range(len(all_patches)) if i not in annotated_patches_indices]
                if len(unannotated_patches_indices) == 0:
                    print(f"No unannotated patches found for {img_name}. Skipping...")
                    continue
                unannotated_patches = [all_patches[i] for i in unannotated_patches_indices]

                # Randomly sample the specified percentage of unannotated patches
                num_to_keep = int(len(unannotated_patches) * args.keep_unannotated)
                if num_to_keep == 0:
                    print(f"No unannotated patches found for {img_name}. Skipping...")
                    continue
                sampled_patches = random.sample(unannotated_patches, num_to_keep)

                # Convert sampled_patches to tensor
                sampled_patches = torch.stack(sampled_patches)
                save_batch_images_with_indices(sampled_patches, img_name, args.dest, unannotated_patches_indices)

        else:
            patches = ImageToPatches(img_tensor, (args.height, args.width), overlap=args.overlap).make_patches()
            save_batch_images(patches, img_name, args.dest)


if __name__ == '__main__':
    main()