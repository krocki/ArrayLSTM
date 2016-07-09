/*
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
 *	CUDA matrix
 *
 */

#ifndef __CUDA_MATRIX_H__
#define __CUDA_MATRIX_H__

#include <containers/c_matrix.h>

#if defined(__GPU__) || defined(__CUDACC__)

#include <containers/c_matrix.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <curand.h>
#include <containers/cu_kernels.h>
#include <state.h>

curandGenerator_t prng;
cublasHandle_t handle;

#ifdef __PRECISE_MATH__
	#define cublas_gemm cublasDgemm
	#define cublas_geam cublasDgeam
#else
	#define cublas_gemm cublasSgemm
	#define cublas_geam cublasSgeam
#endif

#define STREAMS 8
cudaStream_t streams[STREAMS];

template <typename T>
class cu_matrix : public matrix<T> {

	public:
	
		T *cu_data;
		size_t cu_bytes_allocated = 0;
		
		cu_matrix() : matrix<T>() { };
		
		cu_matrix ( size_t rows, size_t cols ) :
			matrix<T> ( rows, cols ) {
			
			cu_resize ( rows, cols );
			cu_zero();
			
		};
		
		virtual ~cu_matrix() {
		
			cu_dealloc();
		}
		
		void cu_alloc ( size_t rows, size_t cols ) {
		
			cudaMalloc ( ( void ** ) & ( cu_data ), rows * cols * sizeof ( dtype ) );
			cu_bytes_allocated = rows * cols * sizeof ( dtype );
			
		}
		
		void cu_dealloc() {
		
			if ( cu_bytes_allocated > 0 ) {
				cudaFree ( cu_data );
				cu_bytes_allocated = 0;
			}
			
		}
		
		void cu_resize ( const size_t new_rows, const size_t new_cols ) {
		
			size_t other_bytes = new_rows * new_cols * sizeof ( T );
			
			if ( other_bytes > cu_bytes_allocated ) {
			
				/* realloc */
				cu_dealloc();
				cu_alloc ( new_rows, new_cols );
				
			}
			
		}
		
		void sync_device_async ( size_t stream_id ) {
		
			cudaMemcpyAsync ( cu_data, matrix<T>::_data_, matrix<T>::bytes, cudaMemcpyHostToDevice, streams[stream_id] );
			
		}
		
		void sync_device() {
		
			cudaMemcpy ( cu_data, matrix<T>::_data_, matrix<T>::bytes, cudaMemcpyHostToDevice );
			
		}
		
		void sync_host_async ( size_t stream_id ) {
		
			cudaMemcpyAsync ( matrix<T>::_data_, cu_data, matrix<T>::bytes, cudaMemcpyDeviceToHost, streams[stream_id] );
			
		}
		
		void sync_host() {
		
			cudaMemcpy ( matrix<T>::_data_, cu_data, matrix<T>::bytes, cudaMemcpyDeviceToHost );
			
		}
		
		void cu_zero() {
		
			cudaMemset ( cu_data, '\0', matrix<T>::bytes );
		}
		
		cu_matrix &operator= ( const cu_matrix &other ) {
		
			matrix<T>::operator= ( other );
			cu_resize ( other.rows(), other.cols() );
			//cudaMemcpy ( cu_data, other.cu_data, matrix<T>::bytes, cudaMemcpyDeviceToDevice );
			return *this;
			
		};
		
		/*TODO: move semantics */
		
		/* copy constr from matrix */
		cu_matrix ( const cu_matrix &other ) { operator= ( other ); }
		
};

void init_curand ( void ) {

	curandCreateGenerator ( &prng, CURAND_RNG_PSEUDO_DEFAULT );
	curandSetPseudoRandomGeneratorSeed ( prng, ( unsigned long long ) clock() );
	
}

void init_cublas ( int device_no ) {

	if ( cublasCreate ( &handle ) != CUBLAS_STATUS_SUCCESS )
	
		std::cout << "!!!! CUBLAS initialization error" << std::endl;
		
	for ( size_t i = 0; i < STREAMS; i++ )
		cudaStreamCreate ( &streams[i] );
		
}

void teardown_cublas ( void ) {

	if ( cublasDestroy ( handle ) != CUBLAS_STATUS_SUCCESS )
		std::cout << "!!!! CUBLAS shutdown error" << std::endl;
		
	for ( size_t i = 0; i < STREAMS; i++ )
		cudaStreamDestroy ( streams[i] );
		
}

// void init_clblas ( void ) { cudaSetDevice ( 0 ); init_cublas(); init_curand(); };
// void teardown_clblas ( void ) { teardown_cublas(); }

// /* C = alpha(A) * (B) + beta(C) */
template<typename T>
void CU_GEMM ( cu_matrix<T> &C, cu_matrix<T> &A, cu_matrix<T> &B, bool a_transposed = false,
			   bool b_transposed = false,
			   dtype alpha = ( dtype ) 1, dtype beta = ( dtype ) 1 ) {
			   
	size_t M = C.rows();
	size_t N = C.cols();
	size_t K = a_transposed ? B.rows() : A.cols();
	
	size_t lda = a_transposed ? K : M;
	size_t ldb = b_transposed ? N : K;
	size_t ldc = M;
	
	const cublasOperation_t tA = a_transposed ? CUBLAS_OP_T : CUBLAS_OP_N;
	const cublasOperation_t tB = b_transposed ? CUBLAS_OP_T : CUBLAS_OP_N;
	
	//if (
	cublas_gemm ( handle, tA, tB, M, N, K, &alpha, A.cu_data, lda, B.cu_data, ldb, &beta, C.cu_data,
				  ldc );
				  
	//				    ) != CUBLAS_STATUS_SUCCESS )
	
	//	std::cout << "!!!! cublas_gemm error" << std::endl;
	
}

template<typename T>
T cu_cross_entropy_loss ( cu_matrix<T> &probs, cu_matrix<T> &targets ) {

	return 0;
	
	
}

template<typename T>
void cu_copy_state ( State<T> &dst, State<T> &src ) {

	for ( size_t i = 0; i < dst.matrices.size(); i++ ) {
	
		cudaMemcpy ( dst.matrices[i].cu_data,
					 src.matrices[i].cu_data,
					 src.matrices[i].size() * sizeof ( T ),
					 cudaMemcpyDeviceToDevice );
					 
	}
	
}

template<typename T>
void cu_copy ( cu_matrix<T> &dst, cu_matrix<T> &src ) {

	cudaMemcpy ( dst.cu_data,
				 src.cu_data,
				 src.size() * sizeof ( T ),
				 cudaMemcpyDeviceToDevice );
				 
				 
}

void sync_stream ( size_t stream_id ) {


	cudaStreamSynchronize ( streams[stream_id] );
	
}

#endif /* defined(__GPU__) || defined(__CUDACC__) */

#endif /* __CUDA_MATRIX__ */