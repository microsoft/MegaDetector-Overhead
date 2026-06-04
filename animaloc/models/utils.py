__copyright__ = \
    """
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

from typing import Union, Tuple, List, Optional

__all__ = ['load_model', 'count_parameters', 'LossWrapper']

def load_model(model: torch.nn.Module, pth_path: str, partial_load: bool = False, encoder_only: bool = False) -> torch.nn.Module:
    ''' Load model parameters from a PTH file 
    
    Args:
        model (torch.nn.Module): the model
        pth_path (str): path to the PTH file containing model parameters
        partial_load (bool, optional): set True to load only the parameters that match the model's keys. Defaults to False.
        encoder_only (bool, optional): set True to load only the encoder part of the model. Defaults to False.
    
    Returns:
        torch.nn.Module
            the model with loaded parameters
    '''
    map_location = torch.device('cpu')
    if torch.cuda.is_available():
        map_location = torch.device('cuda')
    
    checkpoint = torch.load(pth_path, map_location=map_location)
    if encoder_only:
        # Only load the weights associated with model._modules.keys() == encoder ('base_0')
        model_state_dict = model.state_dict()
        checkpoint_state_dict = checkpoint['model_state_dict']
        _encoder_prefixes = ('base_0.', 'model.base_0.', 'backbone.', 'model.backbone.', 'model.model.backbone.')
        filtered_checkpoint_state_dict = {k: v for k, v in checkpoint_state_dict.items() if any(k.startswith(p) for p in _encoder_prefixes)}
        # Ensure that the filtered state dict only contains keys that are in the model's state dict
        filtered_checkpoint_state_dict = {k: v for k, v in filtered_checkpoint_state_dict.items() if k in model_state_dict}
        model.load_state_dict(filtered_checkpoint_state_dict, strict=False)
        # Report missing and additional keys
        missing_keys = [k for k in model_state_dict if k not in filtered_checkpoint_state_dict]
        if missing_keys:
            print(f'Missing keys in current model: {missing_keys}')
        additional_keys = [k for k in filtered_checkpoint_state_dict if k not in model_state_dict or filtered_checkpoint_state_dict[k].shape != model_state_dict.get(k, None).shape]
        if additional_keys:
            print(f'Additional keys ignored in current model: {additional_keys}')
    elif partial_load:
        # Load only the weights that match the model's keys and shapes
        model_state_dict = model.state_dict()  
        checkpoint_state_dict = checkpoint['model_state_dict']  
        # Filter weights to load only those with matching names and shapes  
        filtered_checkpoint_state_dict = {}  
        for k, v in checkpoint_state_dict.items():  
            if k in model_state_dict and v.shape == model_state_dict[k].shape:  
                filtered_checkpoint_state_dict[k] = v  
        # Load the filtered weights  
        model.load_state_dict(filtered_checkpoint_state_dict, strict=False)  
        # Report missing and additional keys  
        missing_keys = [k for k in model_state_dict if k not in filtered_checkpoint_state_dict]  
        if missing_keys:  
            print(f'Missing keys in current model: {missing_keys}')  
        additional_keys = [k for k in checkpoint_state_dict if k not in model_state_dict or checkpoint_state_dict[k].shape != model_state_dict.get(k, None).shape]  
        if additional_keys:  
            print(f'Additional keys ignored in current model: {additional_keys}') 
    else:
        model.load_state_dict(checkpoint['model_state_dict'])

    return model

def count_parameters(model: torch.nn.Module) -> tuple:
    ''' Compute and print model's trainable and total parameters
    
    Args:
        model (torch.nn.Module): the model
    
    Returns:
        tuple
            trainable and total parameters
    '''

    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(f'TRAINABLE PARAMETERS: {train_params}')
    print(f'TOTAL PARAMETERS: {all_params}')

    return train_params, all_params
    
class LossWrapper(torch.nn.Module):
    ''' nn.Module wrapper to add loss output to a model '''

    def __init__(
        self, 
        model: torch.nn.Module, 
        losses: List[dict], 
        mode: str = 'module'
        ) -> None:
        '''
        Args:
            model (torch.nn.Module): the model module
            losses (list): list of dict containing 'idx', 'idy', 'name', 'lambda' and 'loss' as 
                keys, and output index, target index, loss' name, regularization term value and 
                the loss module (torch.nn.Module) as values respectively.
            mode (str, optional): output mode, possible values are:
                - 'loss_only', to output the loss dict only,
                - 'preds_only', to output the predictions only,
                - 'both', to output both loss dict and predictions,
                - 'module' (default), to output loss dict only during training (i.e.
                    model.train()) and both output and loss during evaluation (i.e.
                    model.eval()).
                Defaults to 'module'.
        '''
        
        super().__init__()

        assert isinstance(losses, list), \
            'losses argument must be a list.'

        assert mode in ['loss_only', 'preds_only', 'both', 'module'], \
            'Wrong mode argument, must be \'loss_only\', \'preds_only\', \'both\', or \'module\'.'

        self.model = model
        self.losses = losses
        self.output_mode = mode
  
    def forward(
        self, 
        x: torch.Tensor, 
        target: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None
        ) -> Union[Tuple[torch.Tensor, dict], dict, torch.Tensor]:
        '''
        Args:
            x (torch.Tensor): input of the model
            target (torch.Tensor or list): target used for the loss computation
        
        Returns:
            Union[Tuple[torch.Tensor, dict], dict, torch.Tensor]
                depends on mode value
        '''

        try:
            output = self.model(x)
        except ValueError:
            output = self.model(x, target)

        output_used = output
        if isinstance(output, torch.Tensor):
            output_used = [output]
        if isinstance(target, torch.Tensor):
            target = [target]

        output_dict = {}
        if target is not None:
            for dic in self.losses:
                i = dic['idx']
                j = dic['idy']
                reg = dic['lambda']
                loss_module = dic['loss']
                #loss = loss_module(output_used[i], target[j])
                if loss_module.__class__.__name__ == 'MSELoss' or loss_module.__class__.__name__ == 'L1Loss':
                    loss = loss_module(output_used[i].float().squeeze(1), target[j].float())
                else:
                    loss = loss_module(output_used[i].float(), target[j].float())
                output_dict.update({dic['name'] : reg * loss})

        if self.output_mode == 'module':
            if self.training:
                if not output_dict:
                  output_dict = output
                return output_dict
            else:
                return output, output_dict

        elif self.output_mode == 'loss_only':
            return output_dict
        
        elif self.output_mode == 'preds_only':
            return output
        
        elif self.output_mode == 'both':
            return output, output_dict