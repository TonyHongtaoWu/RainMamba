# Copyright (c) OpenMMLab. All rights reserved.

from .restoration_inference import restoration_inference
from .restoration_video_inference import restoration_video_inference
from .test import multi_gpu_test, single_gpu_test
from .train import init_random_seed, set_random_seed, train_model
from .matting_inference import init_model

__all__ = [
    'train_model', 'set_random_seed', 'init_model',
     'restoration_inference',
    'multi_gpu_test', 'single_gpu_test', 'restoration_video_inference',
    'init_random_seed'
]