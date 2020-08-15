import os
import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

setup(
    name='dcn_cuda_double',
    ext_modules=[
        CUDAExtension('dcn_cuda_double', 
            [
            'src/dcn_v2_double.cpp', 
            'src/dcn_v2_cuda_double.cpp', 
            'src/cuda/dcn_v2_im2col_cuda_double.cu',
            'src/cuda/dcn_v2_psroi_pooling_cuda_double.cu',
             ], 
             include_dirs = ["src", "src/cuda"], 
             
             headers = ['src/dcn_v2_double.h',
                        'src/dcn_v2_cuda_double.h',
                        'src/cuda/dcn_v2_im2col_cuda_double.h',
                        'src/cuda/dcn_v2_psroi_pooling_cuda_double.h',
                        ],
             language="c++"
             ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
