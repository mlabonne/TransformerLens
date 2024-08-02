"""Devices.

Utilities to get the correct device, and assist in distributing model layers across multiple
devices.
"""

from __future__ import annotations

from typing import Optional, Union

import torch
from torch import nn

import transformer_lens


# Fix odd number of GPUs, written by @FailSpy
def get_device_for_block_index(
    index: int,
    cfg: "transformer_lens.HookedTransformerConfig",
    device: Optional[Union[torch.device, str]] = None,
):
    """
    Determine the device for a given layer index based on the model configuration.

    This function assists in distributing model layers across multiple devices. The distribution
    is based on the configuration's number of layers (cfg.n_layers) and devices (cfg.n_devices).

    Args:
        index (int): Model layer index.
        cfg (HookedTransformerConfig): Model and device configuration.
        device (Optional[Union[torch.device, str]], optional): Initial device used for determining the target device.
            If not provided, the function uses the device specified in the configuration (cfg.device).

    Returns:
        torch.device: The device for the specified layer index.
    """
    assert cfg.device is not None
    if device is None:
        device = cfg.device
    device = torch.device(device)
    if device.type == "cpu":
        return device
    
    base_layers_per_device = cfg.n_layers // cfg.n_devices
    remainder_layers = cfg.n_layers % cfg.n_devices
    
    # Determine the device index
    if index < (base_layers_per_device + 1) * remainder_layers:
        device_index = index // (base_layers_per_device + 1)
    else:
        adjusted_index = index - (base_layers_per_device + 1) * remainder_layers
        device_index = adjusted_index // base_layers_per_device + remainder_layers
    
    return torch.device(device.type, (device.index or 0) + device_index)


def move_to_and_update_config(
    model: Union[
        "transformer_lens.HookedTransformer",
        "transformer_lens.HookedEncoder",
        "transformer_lens.HookedEncoderDecoder",
    ],
    device_or_dtype: Union[torch.device, str, torch.dtype],
    print_details=True,
):
    """
    Wrapper around `to` that also updates `model.cfg`.
    """
    if isinstance(device_or_dtype, torch.device):
        model.cfg.device = device_or_dtype.type
        if print_details:
            print("Moving model to device: ", model.cfg.device)
    elif isinstance(device_or_dtype, str):
        model.cfg.device = device_or_dtype
        if print_details:
            print("Moving model to device: ", model.cfg.device)
    elif isinstance(device_or_dtype, torch.dtype):
        model.cfg.dtype = device_or_dtype
        if print_details:
            print("Changing model dtype to", device_or_dtype)
        # change state_dict dtypes
        for k, v in model.state_dict().items():
            model.state_dict()[k] = v.to(device_or_dtype)
    return nn.Module.to(model, device_or_dtype)
