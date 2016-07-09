/*
 *
 * openCL matrix
 *
 * Author: Kamil Rocki <kmrocki@us.ibm.com>
 *
 * Copyright (c) 2016, IBM Corporation. All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
 *
 */

#ifndef __CL_MATRIX_H__
#define __CL_MATRIX_H__

#include <containers/c_matrix.h>

#ifdef __PRECISE_MATH__
	#define cl_dtype cl_double
	#define clblas_gemm clblasDgemm
#else
	#define cl_dtype cl_float
	#define clblas_gemm clblasSgemm
#endif

#include <clBLAS.h>
#include <opencl/clUtils.h>

static cl_int err;
static cl_platform_id platform = 0;
static cl_device_id device = 0;
static cl_context_properties props[3] = { CL_CONTEXT_PLATFORM, 0, 0 };
static cl_context ctx = 0;
static cl_command_queue queue = 0;
static cl_event event = NULL;

int ret = 0;
static unsigned long id = 0;

template <typename H>
class cl_matrix : public matrix<H> {

	public:
	
		cl_mem device_data;
		size_t cl_bytes_allocated = 0;
		size_t local_id;
		
		void cl_alloc ( ) {
		
			// realloc if needed
			if ( cl_bytes_allocated < matrix<H>::bytes_allocated ) {
			
				cl_dealloc ( );
				
				device_data = clCreateBuffer ( ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, matrix<H>::bytes_allocated, NULL, &err );
				cl_bytes_allocated = matrix<H>::bytes_allocated;
			}
			
		}
		
		void cl_dealloc() {
		
			if ( cl_bytes_allocated > 0 )
				CL_SAFE_CALL ( clReleaseMemObject ( device_data ) );
				
			cl_bytes_allocated = 0;
			
		}
		
		cl_matrix() : matrix<H>() { };
		
		cl_matrix ( size_t rows, size_t cols ) :
			matrix<H> ( rows, cols ) {
			
		};
		
		virtual ~cl_matrix() {
		
			cl_dealloc ( );
			
		}
		
		void lazy_sync_device() {
		
			cl_alloc ( );
			
			if ( matrix<H>::write ) {
			
				sync_device();
				matrix<H>::write = false;
				
			}
		}
		
		void lazy_sync_host() {
		
			sync_host();
			
		}
		
		void sync_device() {
		
			dtype *buf = ( dtype * ) clEnqueueMapBuffer ( queue, device_data, CL_TRUE, CL_MAP_WRITE,
						 0, matrix<H>::bytes_allocated, 0, NULL, NULL, NULL );
						 
			memcpy ( buf, matrix<H>::_data_, matrix<H>::bytes_allocated );
			
			clEnqueueUnmapMemObject ( queue, device_data, buf, 0, NULL, NULL );
			
		}
		
		void sync_host() {
		
			dtype *buf = ( dtype * ) clEnqueueMapBuffer (
							 queue, device_data, CL_TRUE, CL_MAP_READ, 0,
							 matrix<H>::bytes_allocated, 0, NULL, NULL, NULL );
							 
			memcpy ( matrix<H>::_data_, buf, matrix<H>::bytes_allocated );
			
			clEnqueueUnmapMemObject ( queue, device_data, buf, 0, NULL, NULL );
			
		}
		
		cl_matrix ( const cl_matrix &other ) : matrix<H> ( other ) {
		
		};
		
		cl_matrix &operator= ( const cl_matrix &other ) {
		
			matrix<H>::operator= ( other );
			return *this;
			
		};
		
		
};

void init_clblas ( void ) {

	std::cout << "init_clblas() " << std::endl;
	
	/* Setup OpenCL environment. */
	CL_SAFE_CALL ( clGetPlatformIDs ( 1, &platform, NULL ) );
	
	
	CL_SAFE_CALL ( clGetDeviceIDs ( platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL ) );
	
	
	props[1] = ( cl_context_properties ) platform;
	ctx = clCreateContext ( props, 1, &device, NULL, NULL, &err );
	
	if ( err != CL_SUCCESS )
		printf ( "clCreateContext() failed with % d\n", err );
		
		
	const size_t BUFFER_STRING_LENGTH = 1024;
	char name[BUFFER_STRING_LENGTH];
	
	CL_SAFE_CALL ( clGetPlatformInfo
				   ( platform, CL_PLATFORM_NAME, BUFFER_STRING_LENGTH, name,
					 NULL ) );
					 
	std::cout << "OpenCL platform: "  << name << std::endl;
	
	CL_SAFE_CALL ( clGetDeviceInfo
				   ( device, CL_DEVICE_NAME, BUFFER_STRING_LENGTH, name,
					 NULL ) );
					 
	std::cout << "OpenCL device: "  << name << std::endl;
	
	queue = clCreateCommandQueue ( ctx, device, 0, &err );
	
	if ( err != CL_SUCCESS ) {
		printf ( "clCreateCommandQueue() failed with % d\n", err );
		CL_SAFE_CALL ( clReleaseContext ( ctx ) );
		
	}
	
	/* Setup clblas. */
	CL_SAFE_CALL ( clblasSetup() );
	
	if ( err != CL_SUCCESS ) {
		printf ( "clblasSetup() failed with % d\n", err );
		CL_SAFE_CALL ( clReleaseCommandQueue ( queue ) );
		CL_SAFE_CALL ( clReleaseContext ( ctx ) );
		
	}
	
}


void teardown_clblas ( void ) {

	std::cout << "teardown_clblas() " << std::endl;
	/* Finalize work with clBLAS */
	clblasTeardown( );
	
	/* Release OpenCL working objects. */
	CL_SAFE_CALL ( clReleaseCommandQueue ( queue ) );
	CL_SAFE_CALL ( clReleaseContext ( ctx ) );
	
}

/* C = alpha(A) * (B) + beta(C) */
// template <typename T>
// void GEMM ( cl_matrix<T> &C, cl_matrix<T> &A, cl_matrix<T> &B,
// 			bool a_transposed = false, bool b_transposed = false,
// 			cl_dtype alpha = ( cl_dtype ) 1, cl_dtype beta = ( cl_dtype ) 1 ) {

// 	A.lazy_sync_device ( );
// 	B.lazy_sync_device ( );
// 	C.lazy_sync_device ( );

// 	const clblasTranspose_ tA = a_transposed ? clblasTrans : clblasNoTrans;
// 	const clblasTranspose_ tB = b_transposed ? clblasTrans : clblasNoTrans;

// 	size_t M = C.rows();
// 	size_t N = C.cols();
// 	size_t K = a_transposed ? A.rows() : A.cols();

// 	size_t lda = a_transposed ? K : M;
// 	size_t ldb = b_transposed ? N : K;
// 	size_t ldc = M;

// 	CL_SAFE_CALL ( clblas_gemm ( clblasColumnMajor, tA, tB, M, N, K, alpha,
// 								 A.device_data, 0, lda, B.device_data, 0, ldb, beta, C.device_data, 0, ldc,
// 								 1, &queue, 0, NULL, &event ) );

// 	C.sync_host( );

// }
#endif /* CL_MATRIX */