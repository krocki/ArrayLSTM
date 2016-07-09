#
#
# Author: Kamil Rocki <kmrocki@us.ibm.com> 
#
# Copyright (c) 2016, IBM Corporation. All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
#
#

# to compile:
#
# make PRECISE_MATH=0 cuda
# or 
# make PRECISE_MATH=1 cuda
# 
# cpu versions are outdated, OpenCL version is not fully implemented

OS := $(shell uname)

CC=g++
CUDA=1
NVCC=nvcc
USE_BLAS=1
USE_SYNC_MATRIX=0
USE_EIGEN=0
USE_CLBLAS=0
USE_CUDA=0
USE_CEREAL=1
INCLUDES=-I. -I./src/
LFLAGS=
CFLAGS=
DEBUG=0
PRECISE_MATH=0
NVCC_FLAGS=-D__GPU__ -m64 -ccbin=g++ --gpu-architecture=sm_52 -D__STRICT_ANSI__ -L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas -lcurand -D__USE_CUDA__

ADD_FLAGS=

ifeq ($(DEBUG),1)
	CFLAGS := -g $(CFLAGS)
endif

ifeq ($(USE_SYNC_MATRIX),1)
	CFLAGS := -D__SYNC_MATRIX__ $(CFLAGS)
endif

ifeq ($(USE_CUDA),1)
	CC := $(NVCC)
	CFLAGS := -D__CUDA_MATRIX__ $(NVCC_CFLAGS) $(CFLAGS)
else
	ifeq ($(USE_CLBLAS),1)
		CFLAGS := -D__CL_MATRIX__ $(CFLAGS)
	endif
endif

ifeq ($(OS),Linux)
	CFLAGS := $(CFLAGS)
	ifeq ($(USE_EIGEN),1)
	INCLUDES := -I/usr/include/eigen3 $(INCLUDES)
	CFLAGS := -DUSE_EIGEN $(CFLAGS)
	endif
else
	#OSX

	LFLAGS := -framework OpenCL $(LFLAGS)
	
endif

ifeq ($(USE_BLAS),1)

	ifeq ($(OS),Linux)
		INCLUDES := -I/opt/OpenBLAS/include $(INCLUDES)
		LFLAGS := -lopenblas -L/opt/OpenBLAS/lib $(LFLAGS)

	else
		#OSX
		INCLUDES := -I/usr/local/opt/openblas/include $(INCLUDES)
		LFLAGS := -lopenblas -L/usr/local/opt/openblas/lib $(LFLAGS)

	endif

	CFLAGS := -D__USE_BLAS__ $(CFLAGS)

endif

ifeq ($(USE_CLBLAS),1)
ifeq ($(OS),Linux)
LFLAGS := -lOpenCL -lclBLAS $(LFLAGS)
else
	LFLAGS := -framework OpenCL -lclblas $(LFLAGS)
endif
	CFLAGS := -D__USE_CLBLAS__ $(CFLAGS)
endif

ifeq ($(PRECISE_MATH),1)
	CFLAGS := -D__PRECISE_MATH__ $(CFLAGS)
else
	NVCC_FLAGS := --use_fast_math $(NVCC_FLAGS)
endif

ifeq ($(USE_CEREAL),1)
	CFLAGS := -D__USE_CEREAL__ $(CFLAGS)
endif

cpu:	
	$(CC) ./deeplstm.cc $(INCLUDES) $(CFLAGS) $(ADD_FLAGS) $(LFLAGS) -std=c++11 -Ofast -o deeplstm
cl:	
	$(CC) ./deeplstm.cc $(INCLUDES) $(CFLAGS) -D__USE_CLBLAS__ -D__CL_MATRIX__ $(ADD_FLAGS) $(LFLAGS) -O3 -framework OpenCL -lclblas -o deeplstm
cuda:
	$(NVCC) ./src/containers/cu_kernels.cu $(CFLAGS) $(NVCC_FLAGS) $(INCLUDES) -D__CUDA_MATRIX__ --std=c++11 -O3 -D_MWAITXINTRIN_H_INCLUDED -c cu_kernels.o
	$(NVCC) ./deeplstm.cc $(INCLUDES) $(CFLAGS) $(ADD_FLAGS) $(NVCC_FLAGS) $(LFLAGS) -D__CUDA_MATRIX__ -std=c++11 -O3 cu_kernels.o -o deeplstm

	
