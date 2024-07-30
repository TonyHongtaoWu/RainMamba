# Copyright (c) OpenMMLab. All rights reserved.
from .pixelwise_loss import CharbonnierLoss, L1Loss, MSELoss

__all__ = [
    'L1Loss', 'MSELoss', 'CharbonnierLoss'
]