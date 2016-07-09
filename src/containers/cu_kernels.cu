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
 * CUDA kernels
 *
 *
 * TODO: split into multiple files
 *
 */

#include <containers/cu_kernels.h>
#include <assert.h>
#include <iostream>

__forceinline__ __device__ dtype device_logistic ( dtype x ) {

	#ifdef __PRECISE_MATH__
	return 1.0 / ( ( dtype ) 1.0 + exp ( -x ) );
	#else
	return __frcp_rn ( ( dtype ) 1 + __expf ( -x ) );
	#endif
	
}

__forceinline__ __device__ dtype device_tanh ( dtype x ) {

	#ifdef __PRECISE_MATH__
	return tanh ( x );
	#else
	return tanhf ( x );
	#endif
	
}

__forceinline__ __device__ dtype device_tanh_prime ( dtype x ) {

	return ( dtype ) 1.0 - x * x;
	
	//c[cid] = __fmaf_rn(-c[cid], c[cid], 1.0f);
	
}

__forceinline__ __device__ dtype device_logistic_prime ( dtype x ) {

	return x * ( ( dtype ) 1.0 - x );
	
}


__forceinline__ __device__ dtype device_exp ( dtype x ) {

	#ifdef __PRECISE_MATH__
	return exp ( x );
	#else
	return expf ( x );
	#endif
	
}

__forceinline__ __device__ dtype device_sqrt_eps ( dtype x, dtype eps ) {

	// #ifdef __PRECISE_MATH__
	return sqrt ( x + eps );
	// #else
	// 	return (__frsqrt_rn(x + eps));
	// #endif
	
};
__global__ void kernel_elementwise_sub ( dtype *__restrict__ c, dtype *__restrict__ b, dtype *__restrict__ a,
		size_t n ) {
		
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	
	if ( tid < n ) {
	
		/* DEBUG */
		assert ( !isinf ( c[tid] ) );
		assert ( !isnan ( c[tid] ) );
		assert ( !isinf ( a[tid] ) );
		assert ( !isnan ( a[tid] ) );
		assert ( !isinf ( b[tid] ) );
		assert ( !isnan ( b[tid] ) );
		
		c[tid] =  b[tid] - a[tid];
		
	}
	
}

void cu_sub ( dtype *__restrict__ c, dtype *__restrict__ b, dtype *__restrict__ a, size_t elements, int stream_idx ) {

	size_t num_blocks = ( elements + NUM_THREADS - 1 ) / NUM_THREADS;
	kernel_elementwise_sub <<< num_blocks, NUM_THREADS, stream_idx>>> ( c, b, a, elements );
	
}

__global__ void kernel_elementwise_submax ( dtype *__restrict__ c, size_t n ) {

	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	
	if ( tid < n ) {
	
		// TODO
		
	}
	
}

void cu_submax ( dtype *__restrict__ data, size_t elements, int idx, int stream_idx ) {

	size_t num_blocks = ( elements + NUM_THREADS - 1 ) / NUM_THREADS;
	kernel_elementwise_submax <<< num_blocks, NUM_THREADS, stream_idx>>> ( data, elements );
	
}

__global__ void kernel_elementwise_exp ( dtype *__restrict__ c, size_t n ) {

	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	
	if ( tid < n ) {
	
		/* DEBUG code, blow-up detection */
		assert ( !isinf ( c[tid] ) );
		assert ( !isnan ( c[tid] ) );
		
		dtype pre = c[tid];
		c[tid] = device_exp ( pre );
		
		/* DEBUG code, blow-up detection */
		assert ( !isinf ( c[tid] ) );
		assert ( !isnan ( c[tid] ) );
		
	}
	
}

void cu_exp ( dtype *__restrict__ data, size_t elements, int stream_idx ) {


	size_t num_blocks = ( elements + NUM_THREADS - 1 ) / NUM_THREADS;
	kernel_elementwise_exp <<< num_blocks, NUM_THREADS, stream_idx>>> ( data, elements );
	
}

__global__ void kernel_elementwise_logistic ( dtype *__restrict__ c, size_t n ) {

	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	
	if ( tid < n )  c[tid] = device_logistic ( c[tid] );
	
}

void cu_logistic ( dtype *__restrict__ data, size_t elements, int stream_idx ) {

	size_t num_blocks = ( elements + NUM_THREADS - 1 ) / NUM_THREADS;
	kernel_elementwise_logistic <<< num_blocks, NUM_THREADS, stream_idx>>> ( data, elements );
	
}

void cu_rand ( dtype *__restrict__ data, size_t elements ) {

	#ifdef __PRECISE_MATH__
	curandGenerateUniformDouble ( prng, data, elements );
	#else
	curandGenerateUniform ( prng, data, elements );
	#endif
	
}

void cu_randn ( dtype *__restrict__ data, size_t elements, dtype mean = ( dtype ) 0, dtype stddev = ( dtype ) 1 ) {

	#ifdef __PRECISE_MATH__

	curandGenerateNormalDouble ( prng, data, elements, mean, stddev );
	
	#else
	
	curandGenerateNormal ( prng, data, elements, mean, stddev );
	
	#endif
}

__global__ void kernel_elementwise_tanh ( dtype *__restrict__ c, size_t n ) {

	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	
	if ( tid < n )  c[tid] = device_tanh ( c[tid] );
	
}

void cu_tanh ( dtype *__restrict__ data, size_t elements, int stream_idx ) {

	size_t num_blocks = ( elements + NUM_THREADS - 1 ) / NUM_THREADS;
	kernel_elementwise_tanh <<< num_blocks, NUM_THREADS, stream_idx>>> ( data, elements );
	
}

__global__ void kernel_elementwise_div_scalar ( dtype *__restrict__ c, dtype *__restrict__ src, dtype scalar,
		size_t n ) {
		
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	
	if ( tid < n )  c[tid] = src[tid] / scalar;
	
}

void cu_div_scalar ( dtype *__restrict__ data, dtype *__restrict__ src, dtype scalar, size_t elements,
					 int stream_idx ) {
					 
	size_t num_blocks = ( elements + NUM_THREADS - 1 ) / NUM_THREADS;
	kernel_elementwise_div_scalar <<< num_blocks, NUM_THREADS, stream_idx>>> ( data, src, scalar, elements );
	
}

__global__ void kernel_elementwise_mult_scalar ( dtype *__restrict__ c, dtype *__restrict__ src, dtype scalar,
		size_t n ) {
		
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	
	if ( tid < n )  c[tid] = src[tid] * scalar;
	
}

void cu_mult_scalar ( dtype *__restrict__ data, dtype *__restrict__ src, dtype scalar, size_t elements,
					  int stream_idx ) {
					  
	size_t num_blocks = ( elements + NUM_THREADS - 1 ) / NUM_THREADS;
	kernel_elementwise_mult_scalar <<< num_blocks, NUM_THREADS, stream_idx>>> ( data, src, scalar, elements );
	
}

void cu_cmp ( dtype *__restrict__ data, dtype *__restrict__ src, dtype scalar, size_t elements, int stream_idx ) {

	size_t num_blocks = ( elements + NUM_THREADS - 1 ) / NUM_THREADS;
	kernel_elementwise_cmp <<< num_blocks, NUM_THREADS, stream_idx>>> ( data, src, scalar, elements );
	
}

__global__ void kernel_elementwise_cmp_matrix ( dtype *__restrict__ c, dtype *__restrict__ a, dtype *__restrict__ m,
		size_t n ) {
		
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	
	if ( tid < n )  c[tid] = ( dtype ) ( a[tid] < fabsf ( m[tid] ) );
	
}

__global__ void kernel_elementwise_cmp ( dtype *__restrict__ c, dtype *__restrict__ a, dtype threshold, size_t n ) {

	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	
	if ( tid < n )  c[tid] = ( dtype ) ( a[tid] < threshold );
	
}

void cu_cmp_matrix ( dtype *__restrict__ data, dtype *__restrict__ src, dtype *__restrict__ matrix, size_t elements,
					 int stream_idx ) {
					 
	size_t num_blocks = ( elements + NUM_THREADS - 1 ) / NUM_THREADS;
	kernel_elementwise_cmp_matrix <<< num_blocks, NUM_THREADS, stream_idx>>> ( data, src, matrix, elements );
	
	
}

void cu_elementwise_zeros ( dtype *__restrict__ data, size_t elements, int stream_idx ) {

	size_t num_blocks = ( elements + NUM_THREADS - 1 ) / NUM_THREADS;
	kernel_elementwise_zeros <<< num_blocks, NUM_THREADS, stream_idx>>> ( data, elements );
	
	
}
__global__ void kernel_elementwise_zeros ( dtype *__restrict__ c, size_t n ) {

	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	
	if ( tid < n )
	
		c[tid] = 0;
		
}

//f'(x) = 1-(f(x))^2
//__device__ inline dtype tanh_prime ( dtype x ) { return ( dtype ) 1 - x * x; };

__global__ void kernel_elementwise_dtanh ( dtype *__restrict__ c, dtype *__restrict__ x, dtype *__restrict__ y,
		size_t n ) {
		
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	
	if ( tid < n )
	
		c[tid] = ( ( dtype ) 1 - x[tid] * x[tid] ) * y[tid];
		
}

void cu_dtanh ( dtype *__restrict__ data0, dtype *__restrict__ data1, dtype *__restrict__ data2, size_t elements,
				int stream_idx = 0 ) {
				
	size_t num_blocks = ( elements + NUM_THREADS - 1 ) / NUM_THREADS;
	kernel_elementwise_dtanh <<< num_blocks, NUM_THREADS, stream_idx>>> ( data0, data1, data2, elements );
	
}

__global__ void kernel_elementwise_mult ( dtype *__restrict__ c, dtype *__restrict__ x, dtype *__restrict__ y,
		size_t n ) {
		
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	
	if ( tid < n )
		c[tid] = x[tid] * y[tid];
		
}

void cu_elementwise_mult ( dtype *__restrict__ data0, dtype *__restrict__ data1, dtype *__restrict__ data2,
						   size_t elements, int stream_idx ) {
						   
	size_t num_blocks = ( elements + NUM_THREADS - 1 ) / NUM_THREADS;
	kernel_elementwise_mult <<< num_blocks, NUM_THREADS, stream_idx>>> ( data0, data1, data2, elements );
	
}

__global__ void kernel_elementwise_mult_add ( dtype *__restrict__ c, dtype *__restrict__ x, dtype *__restrict__ y,
		size_t n ) {
		
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	
	if ( tid < n )
		c[tid] += x[tid] * y[tid];
		
}

void cu_elementwise_mult_add ( dtype *__restrict__ data0, dtype *__restrict__ data1, dtype *__restrict__ data2,
							   size_t elements, int stream_idx ) {
							   
	size_t num_blocks = ( elements + NUM_THREADS - 1 ) / NUM_THREADS;
	kernel_elementwise_mult_add <<< num_blocks, NUM_THREADS, stream_idx>>> ( data0, data1, data2, elements );
	
}

void cu_elementwise_lstm_forward (
	dtype *__restrict__ g,
	dtype *__restrict__ g2,
	dtype *__restrict__ b,
	dtype *__restrict__ h,
	dtype *__restrict__ c,
	dtype *__restrict__ ct,
	dtype *__restrict__ prev_c,
	size_t N, size_t B, int stream_idx ) {
	
	size_t num_blocks = ( N * B + NUM_THREADS - 1 ) / NUM_THREADS;
	kernel_elementwise_lstm_forward <<< num_blocks, NUM_THREADS, stream_idx>>> ( g, g2, b, h, c, ct, prev_c, N, B );
	
}

void cu_elementwise_lstm_backward (
	dtype *__restrict__ dg,
	dtype *__restrict__ dh,
	dtype *__restrict__ c,
	dtype *__restrict__ ct,
	dtype *__restrict__ dc,
	dtype *__restrict__ g,
	dtype *__restrict__ prev_c,
	dtype *__restrict__ prev_dc,
	size_t N, size_t B, int stream_idx ) {
	
	
	size_t num_blocks = ( N * B + NUM_THREADS - 1 ) / NUM_THREADS;
	kernel_elementwise_lstm_backward <<< num_blocks, NUM_THREADS, stream_idx>>> ( dg, dh, c, ct, dc, g, prev_c, prev_dc,  N,
			B );
			
}

void cu_elementwise_gauss_lstm_forward (
	dtype *__restrict__ g,
	dtype *__restrict__ g2,
	dtype *__restrict__ b,
	dtype *__restrict__ h,
	dtype *__restrict__ c,
	dtype *__restrict__ ct,
	dtype *__restrict__ prev_c,
	dtype *__restrict__ rands,
	size_t N, size_t B, int stream_idx ) {
	
	size_t num_blocks = ( N * B + NUM_THREADS - 1 ) / NUM_THREADS;
	kernel_elementwise_gauss_lstm_forward <<< num_blocks, NUM_THREADS, stream_idx>>> ( g, g2, b, h, c, ct, prev_c, rands, N,
			B );
			
}

void cu_elementwise_gauss_lstm_backward (
	dtype *__restrict__ dg,
	dtype *__restrict__ dh,
	dtype *__restrict__ c,
	dtype *__restrict__ ct,
	dtype *__restrict__ dc,
	dtype *__restrict__ g,
	dtype *__restrict__ prev_c,
	dtype *__restrict__ prev_dc,
	size_t N, size_t B, int stream_idx ) {
	
	size_t num_blocks = ( N * B + NUM_THREADS - 1 ) / NUM_THREADS;
	kernel_elementwise_gauss_lstm_backward <<< num_blocks, NUM_THREADS, stream_idx>>> ( dg, dh, c, ct, dc, g, prev_c,
			prev_dc,  N,
			B );
			
}

void cu_elementwise_surprisal_lstm_forward (
	dtype *__restrict__ g, 
	dtype *__restrict__ g2, 
	dtype *__restrict__ b,
	dtype *__restrict__ h,
	dtype *__restrict__ c,
	dtype *__restrict__ ct,
	dtype *__restrict__ prev_c,
	size_t N, size_t B, int stream_idx ) {
	
	size_t num_blocks = ( N * B + NUM_THREADS - 1 ) / NUM_THREADS;
	kernel_elementwise_surprisal_lstm_forward <<< num_blocks, NUM_THREADS, stream_idx>>> ( g, g2, b, h, c, ct, prev_c, N,
			B );
			
}

void cu_elementwise_surprisal_lstm_backward (
	dtype *__restrict__ dg,
	dtype *__restrict__ dh,
	dtype *__restrict__ c,
	dtype *__restrict__ ct,
	dtype *__restrict__ dc,
	dtype *__restrict__ g,
	dtype *__restrict__ prev_c,
	dtype *__restrict__ prev_dc,
	size_t N, size_t B, int stream_idx ) {
	
	
	size_t num_blocks = ( N * B + NUM_THREADS - 1 ) / NUM_THREADS;
	kernel_elementwise_surprisal_lstm_backward <<< num_blocks, NUM_THREADS, stream_idx>>> ( dg, dh, c, ct, dc, g, prev_c,
			prev_dc, N,B );
			
}

void cu_elementwise_mlstm_forward (
	dtype *__restrict__ g,
	dtype *__restrict__ g2,
	dtype *__restrict__ b,
	dtype *__restrict__ h,
	dtype *__restrict__ c,
	dtype *__restrict__ prev_c,
	size_t N, size_t L, size_t B, int stream_idx ) {
	
	size_t num_blocks = ( N * B + NUM_THREADS - 1 ) / NUM_THREADS;
	kernel_elementwise_mlstm_forward <<< num_blocks, NUM_THREADS, stream_idx>>> ( g, g2, b, h, c, prev_c, N, L, B );
	
}

void cu_elementwise_mlstm_backward (
	dtype *__restrict__ dg,
	dtype *__restrict__ dh,
	dtype *__restrict__ c,
	dtype *__restrict__ dc,
	dtype *__restrict__ g,
	dtype *__restrict__ prev_c,
	dtype *__restrict__ prev_dc,
	size_t N, size_t L, size_t B, int stream_idx ) {
	
	size_t num_blocks = ( N * B + NUM_THREADS - 1 ) / NUM_THREADS;
	kernel_elementwise_mlstm_backward <<< num_blocks, NUM_THREADS, stream_idx>>> ( dg, dh, c, dc, g, prev_c, prev_dc,  N, L,
			B );
			
}

void cu_elementwise_clstm_forward ( dtype *__restrict__ g, dtype *__restrict__ g2, dtype *__restrict__ b,
									dtype *__restrict__ h,
									dtype *__restrict__ c,
									dtype *__restrict__ ct,
									dtype *__restrict__ prev_c,
									size_t N, size_t L, size_t B, int stream_idx ) {
									
	size_t num_blocks = ( N * B + NUM_THREADS - 1 ) / NUM_THREADS;
	kernel_elementwise_clstm_forward <<< num_blocks, NUM_THREADS, stream_idx>>> ( g, g2, b, h, c, ct, prev_c, N, L, B );
	
}

void cu_elementwise_clstm_backward ( dtype *__restrict__ dg,
									 dtype *__restrict__ dh,
									 dtype *__restrict__ c,
									 dtype *__restrict__ ct,
									 dtype *__restrict__ dc,
									 dtype *__restrict__ g,
									 dtype *__restrict__ prev_c,
									 dtype *__restrict__ prev_dc,
									 dtype *__restrict__ h,
									 size_t N, size_t L, size_t B, int stream_idx ) {
									 
									 
	size_t num_blocks = ( N * B + NUM_THREADS - 1 ) / NUM_THREADS;
	kernel_elementwise_clstm_backward <<< num_blocks, NUM_THREADS, stream_idx>>> ( dg, dh, c, ct, dc, g, prev_c, prev_dc, h,
			N, L, B );
			
}


void cu_elementwise_sparselstm_forward ( dtype *__restrict__ g, dtype *__restrict__ g2, dtype *__restrict__ b,
		dtype *__restrict__ h,
		dtype *__restrict__ c,
		dtype *__restrict__ ct,
		dtype *__restrict__ prev_c,
		dtype *__restrict__ prev_h,
		size_t N, size_t L, size_t B, int stream_idx ) {
		
	size_t num_blocks = ( N * B + NUM_THREADS - 1 ) / NUM_THREADS;
	kernel_elementwise_sparselstm_forward <<< num_blocks, NUM_THREADS, stream_idx>>> ( g, g2, b, h, c, ct, prev_c, prev_h,
			N,
			L, B );
			
}


void cu_elementwise_sparselstm_backward ( dtype *__restrict__ dg,
		dtype *__restrict__ dh,
		dtype *__restrict__ c,
		dtype *__restrict__ ct,
		dtype *__restrict__ dc,
		dtype *__restrict__ g,
		dtype *__restrict__ prev_c,
		dtype *__restrict__ prev_dc,
		dtype *__restrict__ h,
		dtype *__restrict__ prev_h,
		dtype *__restrict__ prev_dh,
		size_t N, size_t L, size_t B, int stream_idx ) {
		
		
	size_t num_blocks = ( N * B + NUM_THREADS - 1 ) / NUM_THREADS;
	kernel_elementwise_sparselstm_backward <<< num_blocks, NUM_THREADS, stream_idx>>> ( dg, dh, c, ct, dc, g, prev_c,
			prev_dc, h, prev_h, prev_dh, N, L, B );
			
}

/* v2 */
void cu_elementwise_hardattlstm_forward (	dtype *__restrict__ g, dtype *__restrict__ g2,
		dtype *__restrict__ G, dtype *__restrict__ b,
		dtype *__restrict__ h,
		dtype *__restrict__ max_o,
		dtype *__restrict__ c,
		dtype *__restrict__ ct,
		dtype *__restrict__ prev_c,
		dtype *__restrict__ prev_h,
		dtype *__restrict__ rands,
		size_t N, size_t L, size_t B, int stream_idx ) {
		
	size_t num_blocks = ( N * B + NUM_THREADS - 1 ) / NUM_THREADS;
	kernel_elementwise_hardattlstm_forward <<< num_blocks, NUM_THREADS, stream_idx>>> ( g, g2, G, b, h, max_o, c, ct,
			prev_c,
			prev_h, rands, N, L, B );
			
}

void cu_elementwise_hardattlstm_backward ( dtype *__restrict__ dg,
		dtype *__restrict__ dh,
		dtype *__restrict__ c,
		dtype *__restrict__ ct,
		dtype *__restrict__ dc,
		dtype *__restrict__ g,
		dtype *__restrict__ prev_c,
		dtype *__restrict__ prev_dc,
		dtype *__restrict__ h,
		dtype *__restrict__ max_o,
		dtype *__restrict__ prev_h,
		dtype *__restrict__ prev_dh,
		size_t N, size_t L, size_t B, int stream_idx ) {
		
		
	size_t num_blocks = ( N * B + NUM_THREADS - 1 ) / NUM_THREADS;
	kernel_elementwise_hardattlstm_backward <<< num_blocks, NUM_THREADS, stream_idx>>> ( dg, dh, c, ct, dc, g, prev_c,
			prev_dc, h, max_o, prev_h, prev_dh, N, L, B );
			
}

void cu_elementwise_attlstm_forward (	dtype *__restrict__ g, dtype *__restrict__ g2,
										dtype *__restrict__ G, dtype *__restrict__ b,
										dtype *__restrict__ h,
										dtype *__restrict__ c,
										dtype *__restrict__ ct,
										dtype *__restrict__ prev_c,
										dtype *__restrict__ prev_h,
										size_t N, size_t L, size_t B, int stream_idx ) {
										
	size_t num_blocks = ( N * B + NUM_THREADS - 1 ) / NUM_THREADS;
	kernel_elementwise_attlstm_forward <<< num_blocks, NUM_THREADS, stream_idx>>> ( g, g2, G, b, h, c, ct, prev_c, prev_h,
			N,
			L, B );
			
}

void cu_elementwise_attlstm_backward ( dtype *__restrict__ dg,
									   dtype *__restrict__ dh,
									   dtype *__restrict__ c,
									   dtype *__restrict__ ct,
									   dtype *__restrict__ dc,
									   dtype *__restrict__ g,
									   dtype *__restrict__ prev_c,
									   dtype *__restrict__ prev_dc,
									   dtype *__restrict__ h,
									   dtype *__restrict__ prev_h,
									   dtype *__restrict__ prev_dh,
									   size_t N, size_t L, size_t B, int stream_idx ) {
									   
									   
	size_t num_blocks = ( N * B + NUM_THREADS - 1 ) / NUM_THREADS;
	kernel_elementwise_attlstm_backward <<< num_blocks, NUM_THREADS, stream_idx>>> ( dg, dh, c, ct, dc, g, prev_c, prev_dc,
			h, prev_h, prev_dh, N, L, B );
			
}

void cu_elementwise_cmlstm_forward (	dtype *__restrict__ g, dtype *__restrict__ g2,
										dtype *__restrict__ G, dtype *__restrict__ b,
										dtype *__restrict__ h,
										dtype *__restrict__ c,
										dtype *__restrict__ prev_c,
										dtype *__restrict__ prev_h,
										dtype *__restrict__ rands,
										size_t N, size_t L, size_t B, int stream_idx ) {
										
	size_t num_blocks = ( N * B + NUM_THREADS - 1 ) / NUM_THREADS;
	kernel_elementwise_cmlstm_forward <<< num_blocks, NUM_THREADS, stream_idx>>> ( g, g2, G, b, h, c, prev_c, prev_h, rands,
			N, L, B );
			
}

void cu_elementwise_cmlstm_backward ( dtype *__restrict__ dg,
									  dtype *__restrict__ dh,
									  dtype *__restrict__ c,
									  dtype *__restrict__ dc,
									  dtype *__restrict__ g,
									  dtype *__restrict__ prev_c,
									  dtype *__restrict__ prev_dc,
									  dtype *__restrict__ h,
									  dtype *__restrict__ prev_h,
									  dtype *__restrict__ prev_dh,
									  size_t N, size_t L, size_t B, int stream_idx ) {
									  
									  
	size_t num_blocks = ( N * B + NUM_THREADS - 1 ) / NUM_THREADS;
	kernel_elementwise_cmlstm_backward <<< num_blocks, NUM_THREADS, stream_idx>>> ( dg, dh, c, dc, g, prev_c, prev_dc, h,
			prev_h, prev_dh, N, L, B );
			
}

void cu_elementwise_hlstm_forward ( dtype *__restrict__ g, dtype *__restrict__ g2, dtype *__restrict__ b,
									dtype *__restrict__ h,
									dtype *__restrict__ c,
									dtype *__restrict__ ct,
									dtype *__restrict__ prev_c,
									size_t N, size_t L, size_t B, int stream_idx ) {
									
	size_t num_blocks = ( N * B + NUM_THREADS - 1 ) / NUM_THREADS;
	kernel_elementwise_hlstm_forward <<< num_blocks, NUM_THREADS, stream_idx>>> ( g, g2, b, h, c, ct, prev_c, N, L, B );
	
}

void cu_elementwise_hlstm_backward ( dtype *__restrict__ dg,
									 dtype *__restrict__ dh,
									 dtype *__restrict__ c,
									 dtype *__restrict__ ct,
									 dtype *__restrict__ dc,
									 dtype *__restrict__ g,
									 dtype *__restrict__ prev_c,
									 dtype *__restrict__ prev_dc,
									 dtype *__restrict__ h,
									 size_t N, size_t L, size_t B, int stream_idx ) {
									 
									 
	size_t num_blocks = ( N * B + NUM_THREADS - 1 ) / NUM_THREADS;
	kernel_elementwise_hlstm_backward <<< num_blocks, NUM_THREADS, stream_idx>>> ( dg, dh, c, ct, dc, g, prev_c, prev_dc, h,
			N, L, B );
			
}

void cu_elementwise_hclstm_forward ( dtype *__restrict__ g, dtype *__restrict__ g2, dtype *__restrict__ b,
									 dtype *__restrict__ h,
									 dtype *__restrict__ c,
									 dtype *__restrict__ prev_c,
									 size_t N, size_t L, size_t B, int stream_idx ) {
									 
	size_t num_blocks = ( N * B + NUM_THREADS - 1 ) / NUM_THREADS;
	kernel_elementwise_hclstm_forward <<< num_blocks, NUM_THREADS, stream_idx>>> ( g, g2, b, h, c, prev_c, N, L, B );
	
}

void cu_elementwise_hclstm_backward ( dtype *__restrict__ dg,
									  dtype *__restrict__ dh,
									  dtype *__restrict__ c,
									  dtype *__restrict__ dc,
									  dtype *__restrict__ g,
									  dtype *__restrict__ prev_c,
									  dtype *__restrict__ prev_dc,
									  dtype *__restrict__ h,
									  size_t N, size_t L, size_t B, int stream_idx ) {
									  
									  
	size_t num_blocks = ( N * B + NUM_THREADS - 1 ) / NUM_THREADS;
	kernel_elementwise_hclstm_backward <<< num_blocks, NUM_THREADS, stream_idx>>> ( dg, dh, c, dc, g, prev_c, prev_dc, h,
			N,
			L, B );
			
}

void cu_elementwise_hmlstm_forward ( dtype *__restrict__ g, dtype *__restrict__ g2, dtype *__restrict__ b,
									 dtype *__restrict__ h,
									 dtype *__restrict__ c,
									 dtype *__restrict__ prev_c,
									 size_t N, size_t L, size_t B, int stream_idx ) {
									 
	size_t num_blocks = ( N * B + NUM_THREADS - 1 ) / NUM_THREADS;
	kernel_elementwise_hmlstm_forward <<< num_blocks, NUM_THREADS, stream_idx>>> ( g, g2, b, h, c, prev_c, N, L, B );
	
}

void cu_elementwise_hmlstm_backward ( dtype *__restrict__ dg,
									  dtype *__restrict__ dh,
									  dtype *__restrict__ c,
									  dtype *__restrict__ dc,
									  dtype *__restrict__ g,
									  dtype *__restrict__ prev_c,
									  dtype *__restrict__ prev_dc,
									  dtype *__restrict__ h,
									  size_t N, size_t L, size_t B, int stream_idx ) {
									  
									  
	size_t num_blocks = ( N * B + NUM_THREADS - 1 ) / NUM_THREADS;
	kernel_elementwise_hmlstm_backward <<< num_blocks, NUM_THREADS, stream_idx>>> ( dg, dh, c, dc, g, prev_c, prev_dc, h,
			N,
			L, B );
			
}

__global__ void kernel_elementwise_add_row_vector ( dtype *__restrict__ m,
		dtype *__restrict__ v,
		size_t N, size_t B ) {
		
	size_t elements = N * B;
	
	/* there are N * B threads */
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	
	/* in - gates after SGEMMs */
	
	if ( tid < elements ) {
	
		/* add vec */
		m[tid] 	+= 	v[tid / B];
		
	}
}

void cu_add_row_vector ( dtype *__restrict__ m, dtype *__restrict__ v, size_t N, size_t B, int stream_idx ) {

	size_t num_blocks = ( N * B + NUM_THREADS - 1 ) / NUM_THREADS;
	kernel_elementwise_add_row_vector <<< num_blocks, NUM_THREADS, stream_idx>>> ( m, v, N, B );
	
}

__global__ void kernel_elementwise_div_col_vector ( dtype *__restrict__ m,
		dtype *__restrict__ v,
		size_t N, size_t B ) {
		
	size_t elements = N * B;
	
	/* there are N * B threads */
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	
	/* in - gates after SGEMMs */
	
	if ( tid < elements ) {
	
		assert ( !isnan ( v[tid % B] ) );
		assert ( !isinf ( v[tid % B] ) );
		
		/* add vec */
		//if (!isnan(v[tid % B])  && !isinf(v[tid % B]))
		m[tid] 	/= 	v[tid % B];
		
		if ( isnan ( m[tid] ) || isinf ( m[tid] ) )
			printf ( "v %f m %f\n", v[tid % B], m[tid] );
			
		assert ( !isnan ( m[tid] ) );
		assert ( !isinf ( m[tid] ) );
	}
	
}

void cu_row_max ( dtype *__restrict__ v,  dtype *__restrict__ m, int N, int B, int stream_idx ) {


	//size_t num_blocks = (B + NUM_THREADS - 1) / NUM_THREADS;
	kernel_row_max <<< B, 1, stream_idx>>> ( v, m, N, B );
	
	
}

__global__ void kernel_row_max ( dtype *__restrict__ v,  dtype *__restrict__ m, int N, int B ) {

	//size_t elements = N * B;
	
	/* there are N * B threads */
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	
	//printf("%d\n", elements);
	
	v[tid] = -INFINITY;
	dtype local_max = -INFINITY;
	/* TODO make this faster */
#pragma unroll
	
	for ( int n = 0; n < 256; n++ ) {
	
		local_max = fmaxf ( local_max, m[tid + B * n] );
		//printf("%d %d %f %f\n", tid, tid + (int)B * n, v[tid], m[tid + (int)B * n]);
		
		
	}
	
	v[tid] = local_max;
}

__global__ void kernel_elementwise_sub_col_vector ( dtype *__restrict__ m,
		dtype *__restrict__ v,
		size_t N, size_t B ) {
		
	size_t elements = N * B;
	
	/* there are N * B threads */
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	
	if ( tid < elements ) {
	
		/* sub vec */
		m[tid] 	-= 	v[tid % ( int ) B];
		
	}
	
}

void cu_div_col_vector ( dtype *__restrict__ m, dtype *__restrict__ v, size_t N, size_t B, int stream_idx ) {

	size_t num_blocks = ( N * B + NUM_THREADS - 1 ) / NUM_THREADS;
	kernel_elementwise_div_col_vector <<< num_blocks, NUM_THREADS, stream_idx>>> ( m, v, N, B );
	
	
}

void cu_sub_col_vector ( dtype *__restrict__ m, dtype *__restrict__ v, size_t N, size_t B, int stream_idx ) {

	size_t num_blocks = ( N * B + NUM_THREADS - 1 ) / NUM_THREADS;
	kernel_elementwise_sub_col_vector <<< num_blocks, NUM_THREADS, stream_idx>>> ( m, v, N, B );
	
	
}


void cu_sub_max ( dtype *__restrict__ m, size_t N, int stream_idx ) {

	size_t num_blocks = ( N + NUM_THREADS - 1 ) / NUM_THREADS;
	kernel_sub_max <<< num_blocks, NUM_THREADS, stream_idx>>> ( m, N );
	
}

__global__ void kernel_sub_max ( dtype *__restrict__ m, size_t N ) {

	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	
	if ( tid < N ) {
	
		//TODO
	}
	
}

#define i_gates 0 * elements
#define o_gates 1 * elements
#define f_gates 2 * elements
#define c_gates 3 * elements

__global__ void kernel_elementwise_lstm_forward (	dtype *__restrict__ g,
		dtype *__restrict__ g2,
		dtype *__restrict__ b,
		dtype *__restrict__ h,
		dtype *__restrict__ c,
		dtype *__restrict__ ct,
		dtype *__restrict__ prev_c,
		size_t N, size_t B ) {
		
	size_t elements = N * B;
	
	/* there are N * B threads */
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	
	/* in - gates after SGEMMs */
	
	if ( tid < elements ) {
	
		/* add bias */
		g[i_gates + tid] 	+= g2[i_gates + tid] + b[ ( i_gates + tid ) / B];
		g[o_gates + tid] 	+= g2[o_gates + tid] + b[ ( o_gates + tid ) / B];
		g[f_gates + tid] 	+= g2[f_gates + tid] + b[ ( f_gates + tid ) / B];
		g[c_gates + tid] 	+= g2[c_gates + tid] + b[ ( c_gates + tid ) / B];
		
		/* there are 4 * N * B gate activation */
		
		g[i_gates + tid] 	= 	device_logistic	( g[i_gates + tid] );
		g[o_gates + tid] 	= 	device_logistic	( g[o_gates + tid] );
		g[f_gates + tid] 	= 	device_logistic	( g[f_gates + tid] );
		g[c_gates + tid] 	= 	device_tanh	( g[c_gates + tid] );
		
		c[tid] 				= 	g[f_gates + tid] * prev_c[tid];
		c[tid] 				+= 	g[i_gates + tid] * g[c_gates + tid];
		
		//fix
		ct[tid] 			= 	device_tanh ( c[tid] );
		
		h[tid] 				= 	g[o_gates + tid] * ct[tid];
		
	}
	
	/* out - updated c and h */
	
}

__global__ void kernel_elementwise_lstm_backward (
	dtype *__restrict__ dg,
	dtype *__restrict__ dh,
	dtype *__restrict__ c,
	dtype *__restrict__ ct,
	dtype *__restrict__ dc,
	dtype *__restrict__ g,
	dtype *__restrict__ prev_c,
	dtype *__restrict__ prev_dc,
	size_t N, size_t B ) {
	
	size_t elements = N * B;
	
	/* there are N * B threads */
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	
	if ( tid < elements ) {
	
		dc[tid] = dc[tid] + dh[tid] * g[o_gates + tid] * device_tanh_prime ( ct[tid] );
		prev_dc[tid] += dc[tid] * g[f_gates + tid];
		
		//fix
		//dc[tid] = dc[tid] * device_tanh_prime ( ct[tid] );
		
		//propagate error back through gates
		dg[o_gates + tid] = dh[tid] * ct[tid];
		dg[i_gates + tid] = dc[tid] * g[c_gates + tid];
		dg[f_gates + tid] = dc[tid] * prev_c[tid];
		dg[c_gates + tid] = dc[tid] * g[i_gates + tid];
		
		//propagate error back through gates, nonlinearities
		dg[i_gates + tid] 	= 	dg[i_gates + tid] * device_logistic_prime	( g[i_gates + tid] );
		dg[o_gates + tid] 	= 	dg[o_gates + tid] * device_logistic_prime	( g[o_gates + tid] );
		dg[f_gates + tid] 	= 	dg[f_gates + tid] * device_logistic_prime	( g[f_gates + tid] );
		dg[c_gates + tid] 	= 	dg[c_gates + tid] * device_tanh_prime	( g[c_gates + tid] );
		
		//fix
		//carry - c state
		//prev_dc[tid] = dc[tid] * g[f_gates + tid];
		
	}
	
	
}

#undef i_gates
#undef o_gates
#undef f_gates
#undef c_gates


#define i_gates 0 * elements
#define o_gates 1 * elements
#define f_gates 2 * elements
#define c_gates 3 * elements

__global__ void kernel_elementwise_gauss_lstm_forward (
	dtype *__restrict__ g,
	dtype *__restrict__ g2,
	dtype *__restrict__ b,
	dtype *__restrict__ h,
	dtype *__restrict__ c,
	dtype *__restrict__ ct,
	dtype *__restrict__ prev_c,
	dtype *__restrict__ rands,
	size_t N, size_t B ) {
	
	size_t elements = N * B;
	
	/* there are N * B threads */
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	
	/* in - gates after SGEMMs */
	
	if ( tid < elements ) {
	
		/* add bias */
		g[i_gates + tid] 	+= g2[i_gates + tid] + b[ ( i_gates + tid ) / B];
		g[o_gates + tid] 	+= g2[o_gates + tid] + b[ ( o_gates + tid ) / B];
		g[f_gates + tid] 	+= g2[f_gates + tid] + b[ ( f_gates + tid ) / B];
		g[c_gates + tid] 	+= g2[c_gates + tid] + b[ ( c_gates + tid ) / B];
		
		/* there are 4 * N * B gate activation */
		g[i_gates + tid] 	= 	device_logistic	( g[i_gates + tid] );
		g[o_gates + tid] 	= 	device_logistic	( g[o_gates + tid] );
		g[f_gates + tid] 	= 	device_logistic	( g[f_gates + tid] );
		g[c_gates + tid] 	= 	device_tanh	( g[c_gates + tid] );
		
		c[tid] 				= 	g[f_gates + tid] * prev_c[tid];
		c[tid] 				+= 	g[i_gates + tid] * g[c_gates + tid];
		
		//fix
		ct[tid] 			= 	device_tanh ( c[tid] );
		
		h[tid] 				= 	g[o_gates + tid] * ct[tid] + rands[tid];
		
	}
	
	/* out - updated c and h */
	
}

__global__ void kernel_elementwise_gauss_lstm_backward (

	dtype *__restrict__ dg,
	dtype *__restrict__ dh,
	dtype *__restrict__ c,
	dtype *__restrict__ ct,
	dtype *__restrict__ dc,
	dtype *__restrict__ g,
	dtype *__restrict__ prev_c,
	dtype *__restrict__ prev_dc,
	size_t N, size_t B ) {
	
	size_t elements = N * B;
	
	/* there are N * B threads */
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	
	if ( tid < elements ) {
	
		dc[tid] = dc[tid] + dh[tid] * g[o_gates + tid] * device_tanh_prime ( ct[tid] );
		prev_dc[tid] += dc[tid] * g[f_gates + tid];
		
		//propagate error back through gates
		dg[o_gates + tid] = dh[tid] * ct[tid];
		dg[i_gates + tid] = dc[tid] * g[c_gates + tid];
		dg[f_gates + tid] = dc[tid] * prev_c[tid];
		dg[c_gates + tid] = dc[tid] * g[i_gates + tid];
		
		//propagate error back through gates, nonlinearities
		dg[i_gates + tid] 	= 	dg[i_gates + tid] * device_logistic_prime	( g[i_gates + tid] );
		dg[o_gates + tid] 	= 	dg[o_gates + tid] * device_logistic_prime	( g[o_gates + tid] );
		dg[f_gates + tid] 	= 	dg[f_gates + tid] * device_logistic_prime	( g[f_gates + tid] );
		dg[c_gates + tid] 	= 	dg[c_gates + tid] * device_tanh_prime	( g[c_gates + tid] );
		
	}
	
	
}

#undef i_gates
#undef o_gates
#undef f_gates
#undef c_gates

#define i_gates 0 * elements
#define o_gates 1 * elements
#define f_gates 2 * elements
#define c_gates 3 * elements

__global__ void kernel_elementwise_surprisal_lstm_forward (	dtype *__restrict__ g,
		dtype *__restrict__ g2,
		dtype *__restrict__ b,
		dtype *__restrict__ h,
		dtype *__restrict__ c,
		dtype *__restrict__ ct,
		dtype *__restrict__ prev_c,
		size_t N, size_t B ) {
		
	size_t elements = N * B;
	
	/* there are N * B threads */
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	
	/* in - gates after SGEMMs */
	
	if ( tid < elements ) {
	
		/* add bias */
		g[i_gates + tid] 	+= g2[i_gates + tid] + b[ ( i_gates + tid ) / B];
		g[o_gates + tid] 	+= g2[o_gates + tid] + b[ ( o_gates + tid ) / B];
		g[f_gates + tid] 	+= g2[f_gates + tid] + b[ ( f_gates + tid ) / B];
		g[c_gates + tid] 	+= g2[c_gates + tid] + b[ ( c_gates + tid ) / B];
		
		/* there are 4 * N * B gate activation */
		
		g[i_gates + tid] 	= 	device_logistic	( g[i_gates + tid] );
		g[o_gates + tid] 	= 	device_logistic	( g[o_gates + tid] );
		g[f_gates + tid] 	= 	device_logistic	( g[f_gates + tid] );
		g[c_gates + tid] 	= 	device_tanh	( g[c_gates + tid] );
		
		c[tid] 				= 	g[f_gates + tid] * prev_c[tid];
		c[tid] 				+= 	g[i_gates + tid] * g[c_gates + tid];
		
		ct[tid] 			= 	device_tanh ( c[tid] );
		
		h[tid] 				= 	g[o_gates + tid] * ct[tid];
		
	}
	
	/* out - updated c and h */
	
}

__global__ void kernel_elementwise_surprisal_lstm_backward (
	dtype *__restrict__ dg,
	dtype *__restrict__ dh,
	dtype *__restrict__ c,
	dtype *__restrict__ ct,
	dtype *__restrict__ dc,
	dtype *__restrict__ g,
	dtype *__restrict__ prev_c,
	dtype *__restrict__ prev_dc,
	size_t N, size_t B ) {
	
	size_t elements = N * B;
	
	/* there are N * B threads */
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	
	if ( tid < elements ) {
	
		dc[tid] = dc[tid] + dh[tid] * g[o_gates + tid] * device_tanh_prime ( ct[tid] );
		prev_dc[tid] += dc[tid] * g[f_gates + tid];
		
		//fix
		//dc[tid] = dc[tid] * device_tanh_prime ( ct[tid] );
		
		//propagate error back through gates
		dg[o_gates + tid] = dh[tid] * ct[tid];
		dg[i_gates + tid] = dc[tid] * g[c_gates + tid];
		dg[f_gates + tid] = dc[tid] * prev_c[tid];
		dg[c_gates + tid] = dc[tid] * g[i_gates + tid];
		
		//propagate error back through gates, nonlinearities
		dg[i_gates + tid] 	= 	dg[i_gates + tid] * device_logistic_prime	( g[i_gates + tid] );
		dg[o_gates + tid] 	= 	dg[o_gates + tid] * device_logistic_prime	( g[o_gates + tid] );
		dg[f_gates + tid] 	= 	dg[f_gates + tid] * device_logistic_prime	( g[f_gates + tid] );
		dg[c_gates + tid] 	= 	dg[c_gates + tid] * device_tanh_prime	( g[c_gates + tid] );
		
		//fix
		//carry - c state
		//prev_dc[tid] = dc[tid] * g[f_gates + tid];
		
	}
	
	
}

#undef i_gates
#undef o_gates
#undef f_gates
#undef c_gates

#define i_gates 0 * N * B * L
#define o_gates 1 * N * B * L
#define f_gates 2 * N * B * L
#define c_gates 3 * N * B * L

__global__ void kernel_elementwise_mlstm_forward (	dtype *__restrict__ g,
		dtype *__restrict__ g2,
		dtype *__restrict__ b,
		dtype *__restrict__ h,
		dtype *__restrict__ c,
		dtype *__restrict__ prev_c,
		size_t N, size_t L, size_t B ) {
		
	size_t elements = N * B;
	
	/* there are N * B threads */
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	
	/* in - gates after SGEMMs */
	
	if ( tid < elements ) {
	
		h[tid] = 0;
#pragma unroll
		
		for ( int l = 0; l < L; l++ ) {
		
			int ltid = tid + l * N * B;
			
			/* add bias */
			g[i_gates + ltid] 	+= g2[i_gates + ltid] + b[ ( i_gates + ltid ) / B];
			g[o_gates + ltid] 	+= g2[o_gates + ltid] + b[ ( o_gates + ltid ) / B];
			g[f_gates + ltid] 	+= g2[f_gates + ltid] + b[ ( f_gates + ltid ) / B];
			g[c_gates + ltid] 	+= g2[c_gates + ltid] + b[ ( c_gates + ltid ) / B];
			
			/* there are 4 * N * B gate activation */
			
			g[i_gates + ltid] 	= 	device_logistic	( g[i_gates + ltid] );
			g[o_gates + ltid] 	= 	device_logistic	( g[o_gates + ltid] );
			g[f_gates + ltid] 	= 	device_logistic	( g[f_gates + ltid] );
			g[c_gates + ltid] 	= 	device_tanh	( g[c_gates + ltid] );
			
			c[ltid] 				= 	g[f_gates + ltid] * prev_c[ltid];
			c[ltid] 				+= 	g[i_gates + ltid] * g[c_gates + ltid];
			
			c[ltid] 				= 	device_tanh ( c[ltid] );
			
			h[tid] 				+= 	g[o_gates + ltid] * c[ltid];
			
		}
		
		
	}
	
	/* out - updated c and h */
	
}

__global__ void kernel_elementwise_mlstm_backward (
	dtype *__restrict__ dg,
	dtype *__restrict__ dh,
	dtype *__restrict__ c,
	dtype *__restrict__ dc,
	dtype *__restrict__ g,
	dtype *__restrict__ prev_c,
	dtype *__restrict__ prev_dc,
	size_t N, size_t L, size_t B ) {
	
	size_t elements = N * B;
	
	/* there are N * B threads */
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	
	if ( tid < elements ) {
	
#pragma unroll
	
		for ( int l = 0; l < L; l++ ) {
		
			int ltid = tid + l * N * B;
			dc[ltid] = dc[ltid] + dh[tid] * g[o_gates + ltid];
			dc[ltid] = dc[ltid] * device_tanh_prime ( c[ltid] );
			
			//propagate error back through gates
			dg[o_gates + ltid] = dh[tid] * c[ltid];
			dg[i_gates + ltid] = dc[ltid] * g[c_gates + ltid];
			dg[f_gates + ltid] = dc[ltid] * prev_c[ltid];
			dg[c_gates + ltid] = dc[ltid] * g[i_gates + ltid];
			
			//propagate error back through gates, nonlinearities
			dg[i_gates + ltid] 	= 	dg[i_gates + ltid] * device_logistic_prime	( g[i_gates + ltid] );
			dg[o_gates + ltid] 	= 	dg[o_gates + ltid] * device_logistic_prime	( g[o_gates + ltid] );
			dg[f_gates + ltid] 	= 	dg[f_gates + ltid] * device_logistic_prime	( g[f_gates + ltid] );
			dg[c_gates + ltid] 	= 	dg[c_gates + ltid] * device_tanh_prime	( g[c_gates + ltid] );
			
			//carry - c state
			prev_dc[ltid] = dc[ltid] * g[f_gates + ltid];
			
		}
	}
	
	
}

#undef i_gates
#undef o_gates
#undef f_gates
#undef c_gates

/*****************/

#define i_gates 0 * N * B * L
#define o_gates 1 * N * B * L
#define f_gates 2 * N * B * L
#define c_gates 3 * N * B * L

__device__ float sparse_penalty = 1.0f;
__device__ float i_activations = 0.0f;
__device__ float f_activations = 0.0f;
__device__ float o_activations = 0.0f;
__device__ float target = 0.1f;

void cu_elementwise_sparselstm_sparsity ( size_t N, size_t L, size_t B, size_t S, dtype *__restrict__ b, dtype corr ) {

	size_t num_blocks = ( N * B + NUM_THREADS - 1 ) / NUM_THREADS;
	kernel_elementwise_sparselstm_sparsity <<< num_blocks, NUM_THREADS>>> ( N, L, B, S, b, corr );
	
}

__global__ void kernel_elementwise_sparselstm_sparsity ( size_t N, size_t L, size_t B, size_t S, dtype *__restrict__ b,
		dtype corr ) {
		
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	
	size_t elements = ( S - 1 ) * L;
	
	// if (tid == 0) {
	
	// 	printf(" S %d g i %f o %f f %f ", S, i_activations/elements, o_activations/elements, f_activations/elements);
	// 	i_activations = 0;
	// 	f_activations = 0;
	// 	o_activations = 0;
	
	// }
	
	// float corr_factor_f = f_activations/elements - target;
	// float corr_factor_i = i_activations/elements - target;
	
	// __syncthreads();
	
	if ( tid < elements ) {
	
		for ( int l = 0; l < L; l++ ) {
		
			int ltid = tid + l * N * B;
			
			//b[(i_gates + ltid) / B] -= 0.005f * corr_factor_i;
			//b[(o_gates + ltid) / B] -= 0.0005f * sparse_penalty;
			b[ ( f_gates + ltid ) / B] += corr * l;
			
		}
	}
}

__global__ void kernel_elementwise_sparselstm_forward (	dtype *__restrict__ g,
		dtype *__restrict__ g2,
		dtype *__restrict__ b,
		dtype *__restrict__ h,
		dtype *__restrict__ c,
		dtype *__restrict__ ct,
		dtype *__restrict__ prev_c,
		dtype *__restrict__ prev_h,
		size_t N, size_t L, size_t B ) {
		
	size_t elements = N * B;
	
	/* there are N * B threads */
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	
	/* in - gates after SGEMMs */
	if ( tid < elements ) {
	
		h[tid] = 0;
#pragma unroll
		
		for ( int l = 0; l < L; l++ ) {
		
			int ltid = tid + l * N * B;
			
			/* add bias */
			g[i_gates + ltid] 	+= g2[i_gates + ltid] + b[ ( i_gates + ltid ) / B];
			g[o_gates + ltid] 	+= g2[o_gates + ltid] + b[ ( o_gates + ltid ) / B];
			g[f_gates + ltid] 	+= g2[f_gates + ltid] + b[ ( f_gates + ltid ) / B];
			g[c_gates + ltid] 	+= g2[c_gates + ltid] + b[ ( c_gates + ltid ) / B];
			
			/* there are 4 * N * B gate activation */
			
			g[i_gates + ltid] 	= 	device_logistic	( g[i_gates + ltid] );
			g[o_gates + ltid] 	= 	device_logistic	( g[o_gates + ltid] );
			g[f_gates + ltid] 	= 	device_logistic	( g[f_gates + ltid] );
			g[c_gates + ltid] 	= 	device_tanh	( g[c_gates + ltid] );
			
			// atomicAdd(&i_activations, (float)g[i_gates + ltid]/(float)(N * B));
			// atomicAdd(&o_activations, (float)g[o_gates + ltid]/(float)(N * B));
			// atomicAdd(&f_activations, (float)g[f_gates + ltid]/(float)(N * B));
			
			c[ltid] 				=	( ( dtype ) 1 - g[f_gates + ltid] ) * prev_c[ltid];
			c[ltid] 				+= 	g[i_gates + ltid] * g[c_gates + ltid];
			
			ct[ltid] 				= 	device_tanh ( c[ltid] );
			
			h[tid] 				+= 	g[o_gates + ltid] * ct[ltid];
			
		}
		
		h[tid] = device_tanh ( h[tid] );
	}
	
	// 		h[tid] = 0;
	// #pragma unroll
	// 		for (int l = 0; l < L; l++) {
	
	// 			int ltid = tid + l * N * B;
	
	// 			/* add bias */
	// 			g[i_gates + ltid] 	+= g2[i_gates + ltid] + b[(i_gates + ltid) / B];
	// 			g[o_gates + ltid] 	+= g2[o_gates + ltid] + b[(o_gates + ltid) / B];
	// 			g[f_gates + ltid] 	+= g2[f_gates + ltid] + b[(f_gates + ltid) / B];
	// 			g[c_gates + ltid] 	+= g2[c_gates + ltid] + b[(c_gates + ltid) / B];
	
	// 			/* there are 4 * N * B gate activation */
	// 			g[i_gates + ltid] 	= 	device_logistic	(g[i_gates+ltid]);
	// 			g[o_gates + ltid] 	= 	device_logistic	(g[o_gates+ltid]);
	// 			g[f_gates + ltid] 	= 	device_logistic	(g[f_gates+ltid]);
	// 			g[c_gates + ltid] 	= 	device_tanh		(g[c_gates+ltid]);
	
	// 			// atomicAdd(&i_activations, (float)g[i_gates + ltid]);
	// 			// atomicAdd(&o_activations, (float)g[o_gates + ltid]);
	// 			// atomicAdd(&f_activations, (float)g[f_gates + ltid]);
	
	// 			//printf("tid: %d, %d, i %f o %f f %f\n", tid, ltid, g[i_gates + ltid], g[o_gates + ltid], g[f_gates + ltid]);
	
	// 			c[ltid] 			= 	((dtype)1 - g[f_gates + ltid]) * prev_c[ltid];
	// 			c[ltid] 			+= 	g[i_gates + ltid] * g[c_gates + ltid];
	
	// 			c[ltid] 			= 	device_tanh(c[ltid]);
	
	// 			h[tid] 				+= 	g[o_gates + ltid] * c[ltid];
	
	// 			/****** sparse2 ******/
	// 			h[tid]				+= 	((dtype)1 - g[o_gates + ltid]) * prev_h[tid];
	
	// 		}
	
	// 		h[tid] = device_tanh(h[tid]);
	
	//	}
	
	/* out - updated c and h */
	
}

__global__ void kernel_elementwise_sparselstm_backward (
	dtype *__restrict__ dg,
	dtype *__restrict__ dh,
	dtype *__restrict__ c,
	dtype *__restrict__ ct,
	dtype *__restrict__ dc,
	dtype *__restrict__ g,
	dtype *__restrict__ prev_c,
	dtype *__restrict__ prev_dc,
	dtype *__restrict__ h,
	dtype *__restrict__ prev_h,
	dtype *__restrict__ prev_dh,
	size_t N, size_t L, size_t B ) {
	
	size_t elements = N * B;
	
	/* there are N * B threads */
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	
	if ( tid < elements ) {
	
		dh[tid] = dh[tid] * device_tanh_prime	( h[tid] );
#pragma unroll
		
		for ( int l = 0; l < L; l++ ) {
		
			int ltid = tid + l * N * B;
			dc[ltid] = dc[ltid] + dh[tid] * g[o_gates + ltid] * device_tanh_prime ( ct[ltid] );
			prev_dc[ltid] += dc[ltid] * ( ( dtype ) 1 - g[f_gates + ltid] );
			
			//dc[ltid] = dc[ltid] * device_tanh_prime ( c[ltid] );
			
			//propagate error back through gates
			dg[o_gates + ltid] = dh[tid] * ct[ltid];
			dg[i_gates + ltid] = dc[ltid] * g[c_gates + ltid];
			dg[f_gates + ltid] = -dc[ltid] * prev_c[ltid];
			dg[c_gates + ltid] = dc[ltid] * g[i_gates + ltid];
			
			//propagate error back through gates, nonlinearities
			dg[i_gates + ltid] 	= 	dg[i_gates + ltid] * device_logistic_prime	( g[i_gates + ltid] );
			dg[o_gates + ltid] 	= 	dg[o_gates + ltid] * device_logistic_prime	( g[o_gates + ltid] );
			dg[f_gates + ltid] 	= 	dg[f_gates + ltid] * device_logistic_prime	( g[f_gates + ltid] );
			dg[c_gates + ltid] 	= 	dg[c_gates + ltid] * device_tanh_prime	( g[c_gates + ltid] );
			
			//carry - c state
			//prev_dc[ltid] = dc[ltid] * g[f_gates + ltid];
			
		}
	}
	
	// 		dh[tid] = dh[tid] * device_tanh_prime (h[tid]);
	
	// #pragma unroll
	// 		for (int l = 0; l < L; l++) {
	
	// 			int ltid = tid + l * N * B;
	// 			dc[ltid] = dc[ltid] + dh[tid] * g[o_gates + ltid];
	// 			dc[ltid] = dc[ltid] * device_tanh_prime ( c[ltid] );
	
	// 			//propagate error back through gates
	// 			dg[o_gates + ltid] = dh[tid] * c[ltid] - dh[ltid] * prev_h[ltid];
	// 			dg[i_gates + ltid] = dc[ltid] * g[c_gates + ltid];
	// 			dg[f_gates + ltid] = -dc[ltid] * prev_c[ltid];
	// 			dg[c_gates + ltid] = dc[ltid] * g[i_gates + ltid];
	
	// 			//propagate error back through gates, nonlinearities
	// 			dg[i_gates + ltid] 	= 	dg[i_gates + ltid] * device_logistic_prime	(g[i_gates+ltid]);
	// 			dg[o_gates + ltid] 	= 	dg[o_gates + ltid] * device_logistic_prime	(g[o_gates+ltid]);
	// 			dg[f_gates + ltid] 	= 	dg[f_gates + ltid] * device_logistic_prime	(g[f_gates+ltid]);
	// 			dg[c_gates + ltid] 	= 	dg[c_gates + ltid] * device_tanh_prime		(g[c_gates+ltid]);
	
	// 			//carry - c state
	// 			prev_dc[ltid] = dc[ltid] * ((dtype)1 - g[f_gates + ltid]);
	
	// 			/****** sparse2 ******/
	// 			// carry - h state
	// 			prev_dh[tid] += dh[tid] * ((dtype)1 - g[o_gates + ltid]);
	
	// 		}
	//	}
	
}

#undef i_gates
#undef o_gates
#undef f_gates
#undef c_gates

/*******************/

#define i_gates 0 * N * B * L
#define o_gates 1 * N * B * L
#define f_gates 2 * N * B * L
#define c_gates 3 * N * B * L

__global__ void kernel_elementwise_clstm_forward (	dtype *__restrict__ g,
		dtype *__restrict__ g2,
		dtype *__restrict__ b,
		dtype *__restrict__ h,
		dtype *__restrict__ c,
		dtype *__restrict__ ct,
		dtype *__restrict__ prev_c,
		size_t N, size_t L, size_t B ) {
		
	size_t elements = N * B;
	
	/* there are N * B threads */
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	
	/* in - gates after SGEMMs */
	
	if ( tid < elements ) {
	
		h[tid] = 0;
#pragma unroll
		
		for ( int l = 0; l < L; l++ ) {
		
			int ltid = tid + l * N * B;
			
			/* add bias */
			g[i_gates + ltid] 	+= g2[i_gates + ltid] + b[ ( i_gates + ltid ) / B];
			g[o_gates + ltid] 	+= g2[o_gates + ltid] + b[ ( o_gates + ltid ) / B];
			g[f_gates + ltid] 	+= g2[f_gates + ltid] + b[ ( f_gates + ltid ) / B];
			g[c_gates + ltid] 	+= g2[c_gates + ltid] + b[ ( c_gates + ltid ) / B];
			
			/* there are 4 * N * B gate activation */
			
			g[i_gates + ltid] 	= 	device_logistic	( g[i_gates + ltid] );
			g[o_gates + ltid] 	= 	device_logistic	( g[o_gates + ltid] );
			g[f_gates + ltid] 	= 	device_logistic	( g[f_gates + ltid] );
			g[c_gates + ltid] 	= 	device_tanh	( g[c_gates + ltid] );
			
			c[ltid] 				= 	g[f_gates + ltid] * prev_c[ltid];
			c[ltid] 				+= 	g[i_gates + ltid] * g[c_gates + ltid];
			
			ct[ltid] 				= 	device_tanh ( c[ltid] );
			
			h[tid] 				+= 	g[o_gates + ltid] * ct[ltid];
			
		}
		
		h[tid] = device_tanh ( h[tid] );
	}
	
	/* out - updated c and h */
	
}

__global__ void kernel_elementwise_clstm_backward (
	dtype *__restrict__ dg,
	dtype *__restrict__ dh,
	dtype *__restrict__ c,
	dtype *__restrict__ ct,
	dtype *__restrict__ dc,
	dtype *__restrict__ g,
	dtype *__restrict__ prev_c,
	dtype *__restrict__ prev_dc,
	dtype *__restrict__ h,
	size_t N, size_t L, size_t B ) {
	
	size_t elements = N * B;
	
	/* there are N * B threads */
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	
	if ( tid < elements ) {
	
		dh[tid] = dh[tid] * device_tanh_prime	( h[tid] );
#pragma unroll
		
		for ( int l = 0; l < L; l++ ) {
		
			int ltid = tid + l * N * B;
			dc[ltid] = dc[ltid] + dh[tid] * g[o_gates + ltid] * device_tanh_prime ( ct[ltid] );
			prev_dc[ltid] += dc[ltid] * g[f_gates + ltid];
			
			//dc[ltid] = dc[ltid] * device_tanh_prime ( c[ltid] );
			
			//propagate error back through gates
			dg[o_gates + ltid] = dh[tid] * ct[ltid];
			dg[i_gates + ltid] = dc[ltid] * g[c_gates + ltid];
			dg[f_gates + ltid] = dc[ltid] * prev_c[ltid];
			dg[c_gates + ltid] = dc[ltid] * g[i_gates + ltid];
			
			//propagate error back through gates, nonlinearities
			dg[i_gates + ltid] 	= 	dg[i_gates + ltid] * device_logistic_prime	( g[i_gates + ltid] );
			dg[o_gates + ltid] 	= 	dg[o_gates + ltid] * device_logistic_prime	( g[o_gates + ltid] );
			dg[f_gates + ltid] 	= 	dg[f_gates + ltid] * device_logistic_prime	( g[f_gates + ltid] );
			dg[c_gates + ltid] 	= 	dg[c_gates + ltid] * device_tanh_prime	( g[c_gates + ltid] );
			
			//carry - c state
			//prev_dc[ltid] = dc[ltid] * g[f_gates + ltid];
			
		}
	}
	
	
}

#undef i_gates
#undef o_gates
#undef f_gates
#undef c_gates

/**** cmlstm */

#define i_gates 0 * N * B * L
#define o_gates 1 * N * B * L
#define f_gates 2 * N * B * L
#define c_gates 3 * N * B * L
#define s_gates 4 * N * B * L

// __global__ void kernel_elementwise_cmlstm_forward ( dtype * __restrict__ g,
// 													dtype * __restrict__ g2,
// 													dtype * __restrict__ G,
// 													dtype * __restrict__ b,
// 													dtype * __restrict__ h,
// 													dtype * __restrict__ c,
// 													dtype * __restrict__ prev_c,
// 													dtype * __restrict__ prev_h,
// 													dtype * __restrict__ rands,
// 													size_t N, size_t L, size_t B ) {

// 	size_t elements = N * B;

// 	/* there are N * B threads */
// 	int tid = blockDim.x * blockIdx.x + threadIdx.x;

// 	/* in - gates after SGEMMs */

// 	if ( tid < elements ) {

// 		h[tid] = 0;
// 		dtype total_prob = (dtype)0;
// 		int max_o_idx = tid;
// 		dtype cumsum[64];

// 		for (int l = 0; l < L; l++)
// 			cumsum[l] = 0;

// 		for (int l = 0; l < L; l++) {

// 			int ltid = tid + l * N * B;

// 			// if not selected, just do nothing
// 			c[ltid] = prev_c[ltid];

// 		}

// 		for (int l = 0; l < L; l++) {

// 			int ltid = tid + l * N * B;

// 			/* add bias */
// 			g[i_gates + ltid] 	+= g2[i_gates + ltid] + b[(i_gates + ltid) / B];
// 			g[o_gates + ltid] 	+= g2[o_gates + ltid] + b[(o_gates + ltid) / B];
// 			g[f_gates + ltid] 	+= g2[f_gates + ltid] + b[(f_gates + ltid) / B];
// 			g[c_gates + ltid] 	+= g2[c_gates + ltid] + b[(c_gates + ltid) / B];
// 			g[s_gates + ltid] 	+= g2[s_gates + ltid] + b[(s_gates + ltid) / B];

// 			/* there are 4 * N * B gate activations */
// 			g[i_gates + ltid] 	= 	device_logistic	(g[i_gates+ltid]);
// 			g[o_gates + ltid] 	= 	device_logistic	(g[o_gates+ltid]);
// 			g[f_gates + ltid] 	= 	device_logistic	(g[f_gates+ltid]);
// 			g[c_gates + ltid] 	= 	device_tanh		(g[c_gates+ltid]);

// 			//for softmax
// 			total_prob += device_exp(g[s_gates + ltid]);

// 		}

// 		for (int l = 0; l < L; l++) {
// 			int ltid = tid + l * N * B;
// 			g[s_gates + ltid] = device_exp(g[s_gates + ltid])/total_prob;
// 		}

// 		cumsum[0] = g[s_gates + tid];

// 		for (int l = 1; l < L; l++) {

// 			int ltid = tid + l * N * B;

// 			cumsum[l] = g[s_gates + ltid] + cumsum[l-1];

// 		}

// 		for (int l = 0; l < L; l++) {

// 			if (rands[tid] <= cumsum[l]) {

// 				int ltid = tid + l * N * B;
// 				max_o_idx = ltid;
// 				// printf("l = %d, rand %f, total %f, cumsum %f, max_o = %d\n",
// 				//  		l, rands[tid], total_prob, cumsum[l], max_o_idx);
// 				break;

// 			}

// 		}

// 		//for (int l = 0; l < L; l++) {

// 			//int ltid = tid + l * N * B;

// 			//softmax

// 			//stochastic pass
// 			// G[i_gates + ltid] 	= 	(g[i_gates+ltid] > rands[i_gates+ltid]);
// 			// G[o_gates + ltid] 	= 	(g[o_gates+ltid] > rands[o_gates+ltid]);
// 			// G[f_gates + ltid] 	= 	(g[f_gates+ltid] > rands[f_gates+ltid]);

// 			//G[s_gates + ltid] 	= 	(g[s_gates+ltid]);

// 			G[i_gates + max_o_idx] 	= 	(g[s_gates+max_o_idx]) * (g[i_gates+max_o_idx]);
// 			G[o_gates + max_o_idx] 	= 	(g[s_gates+max_o_idx]) * (g[o_gates+max_o_idx]);
// 			G[f_gates + max_o_idx] 	= 	(g[s_gates+max_o_idx]) * (g[f_gates+max_o_idx]);

// 			c[max_o_idx] 				= 	((dtype)1 - G[f_gates + max_o_idx]) * prev_c[max_o_idx];
// 			c[max_o_idx] 				+= 	G[i_gates + max_o_idx] * g[c_gates + max_o_idx];

// 			c[max_o_idx] 				= 	device_tanh(c[max_o_idx]);

// 			h[tid] 				+= 	(G[o_gates + max_o_idx]) * c[max_o_idx];

// 		//}

// 		//h[tid] 	+= 	prev_h[tid];

// 		h[tid] = device_tanh(h[tid]);

// 	}

// 	/* out - updated c and h */

// }

// __global__ void kernel_elementwise_cmlstm_backward (
// 		dtype* __restrict__ dg,
// 		dtype* __restrict__ dh,
// 		dtype* __restrict__ c,
// 	 	dtype* __restrict__ dc,
// 	  	dtype* __restrict__ g,
// 	  	dtype* __restrict__ prev_c,
// 	  	dtype* __restrict__ prev_dc,
// 	  	dtype* __restrict__ h,
// 	  	dtype * __restrict__ prev_h,
// 	  	dtype * __restrict__ prev_dh,
// 		size_t N, size_t L, size_t B ) {

// 	size_t elements = N * B;

// 	/* there are N * B threads */
// 	int tid = blockDim.x * blockIdx.x + threadIdx.x;

// 	dtype total_prob = (dtype)0;

// 	if ( tid < elements ) {

// 		dh[tid] = dh[tid] * device_tanh_prime		(h[tid]);
// #pragma unroll
// 		for (int l = 0; l < L; l++) {

// 			int ltid = tid + l * N * B;
// 			dc[ltid] = dc[ltid] + dh[tid] * g[o_gates + ltid] * g[s_gates + ltid];
// 			dc[ltid] = dc[ltid] * device_tanh_prime ( c[ltid] );

// 			//propagate error back through gates
// 			dg[o_gates + ltid] = dh[tid] * c[ltid] * g[s_gates + ltid];
// 			dg[i_gates + ltid] = dc[ltid] * g[c_gates + ltid] * g[s_gates + ltid];
// 			dg[f_gates + ltid] = -dc[ltid] * prev_c[ltid] * g[s_gates + ltid];
// 			dg[c_gates + ltid] = dc[ltid] * g[i_gates + ltid] * g[s_gates + ltid];
// 			dg[s_gates + ltid] = dc[ltid] * g[i_gates + ltid] * g[c_gates + ltid] +
// 								 dh[tid] * c[ltid] * g[o_gates + ltid] +
// 								 -dc[ltid] * prev_c[ltid] * g[f_gates + ltid];

// 			//propagate error back through gates, nonlinearities
// 			dg[i_gates + ltid] 	= 	dg[i_gates + ltid] * device_logistic_prime	(g[i_gates+ltid]);
// 			dg[o_gates + ltid] 	= 	dg[o_gates + ltid] * device_logistic_prime	(g[o_gates+ltid]);
// 			dg[f_gates + ltid] 	= 	dg[f_gates + ltid] * device_logistic_prime	(g[f_gates+ltid]);
// 			dg[c_gates + ltid] 	= 	dg[c_gates + ltid] * device_tanh_prime		(g[c_gates+ltid]);

// 			//softmax derivative
// 			dg[s_gates + ltid] = dg[s_gates + ltid] * (g[s_gates+ltid]);
// 			total_prob += dg[s_gates + ltid];

// 			//carry - c state
// 			prev_dc[ltid] = dc[ltid] * ((dtype)1 - g[f_gates + ltid] * g[s_gates + ltid]) + (1 - g[s_gates + ltid]) * dc[ltid];

// 		}

// 		//softmax derivative
// 		for (int l = 0; l < L; l++) {

// 			int ltid = tid + l * N * B;
// 			dg[s_gates + ltid] -= g[s_gates+ltid] * total_prob;

// 		}
// 			/****** sparse2 ******/
// 			// carry - h state
// 			//prev_dh[tid] += dh[tid];
// 	}


// }

#define i_gates 0 * N * B * L
#define o_gates 1 * N * B * L
#define f_gates 2 * N * B * L
#define c_gates 3 * N * B * L
#define s_gates 4 * N * B * L

__global__ void kernel_elementwise_attlstm_forward ( dtype *__restrict__ g,
		dtype *__restrict__ g2,
		dtype *__restrict__ G,
		dtype *__restrict__ b,
		dtype *__restrict__ h,
		dtype *__restrict__ c,
		dtype *__restrict__ ct,
		dtype *__restrict__ prev_c,
		dtype *__restrict__ prev_h,
		size_t N, size_t L, size_t B ) {
		
	size_t elements = N * B;
	
	/* there are N * B threads */
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	
	/* in - gates after SGEMMs */
	
	if ( tid < elements ) {
	
		h[tid] = 0;
		dtype total_prob = ( dtype ) 0;
		
#pragma unroll
		
		for ( int l = 0; l < L; l++ ) {
		
			int ltid = tid + l * N * B;
			
			/* add bias */
			g[i_gates + ltid] 	+= g2[i_gates + ltid] + b[ ( i_gates + ltid ) / B];
			g[o_gates + ltid] 	+= g2[o_gates + ltid] + b[ ( o_gates + ltid ) / B];
			g[f_gates + ltid] 	+= g2[f_gates + ltid] + b[ ( f_gates + ltid ) / B];
			g[c_gates + ltid] 	+= g2[c_gates + ltid] + b[ ( c_gates + ltid ) / B];
			g[s_gates + ltid] 	+= g2[s_gates + ltid] + b[ ( s_gates + ltid ) / B];
			
			/* there are 4 * N * B gate activations */
			g[i_gates + ltid] 	= 	device_logistic	( g[i_gates + ltid] );
			g[o_gates + ltid] 	= 	device_logistic	( g[o_gates + ltid] );
			g[f_gates + ltid] 	= 	device_logistic	( g[f_gates + ltid] );
			g[c_gates + ltid] 	= 	device_tanh	( g[c_gates + ltid] );
			
			//for softmax
			total_prob += device_exp ( g[s_gates + ltid] );
			
		}
		
		for ( int l = 0; l < L; l++ ) {
		
			int ltid = tid + l * N * B;
			
			//softmax
			g[s_gates + ltid] = device_exp ( g[s_gates + ltid] ) / total_prob;
			
			G[i_gates + ltid] 	=	( g[s_gates + ltid] ) * ( g[i_gates + ltid] );
			G[o_gates + ltid] 	=	( g[s_gates + ltid] ) * ( g[o_gates + ltid] );
			G[f_gates + ltid] 	=	( g[s_gates + ltid] ) * ( g[f_gates + ltid] );
			
			c[ltid] 				=	( ( dtype ) 1 - G[f_gates + ltid] ) * prev_c[ltid];
			c[ltid] 				+= 	G[i_gates + ltid] * g[c_gates + ltid];
			
			ct[ltid] 				= 	device_tanh ( c[ltid] );
			
			h[tid] 				+=	( G[o_gates + ltid] ) * ct[ltid];
			
		}
		
		//h[tid] = device_tanh(h[tid]);
		
	}
	
	/* out - updated c and h */
	
}

__global__ void kernel_elementwise_attlstm_backward (
	dtype *__restrict__ dg,
	dtype *__restrict__ dh,
	dtype *__restrict__ c,
	dtype *__restrict__ ct,
	dtype *__restrict__ dc,
	dtype *__restrict__ g,
	dtype *__restrict__ prev_c,
	dtype *__restrict__ prev_dc,
	dtype *__restrict__ h,
	dtype *__restrict__ prev_h,
	dtype *__restrict__ prev_dh,
	size_t N, size_t L, size_t B ) {
	
	size_t elements = N * B;
	
	/* there are N * B threads */
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	
	dtype total_prob = ( dtype ) 0;
	
	if ( tid < elements ) {
	
		//dh[tid] = dh[tid] * device_tanh_prime		(h[tid]);
#pragma unroll
		for ( int l = 0; l < L; l++ ) {
		
			int ltid = tid + l * N * B;
			dc[ltid] = dc[ltid] + dh[tid] * g[o_gates + ltid] * g[s_gates + ltid] * device_tanh_prime ( ct[ltid] );
			//dc[ltid] = dc[ltid] * device_tanh_prime ( c[ltid] );
			
			//propagate error back through gates
			dg[o_gates + ltid] = dh[tid] * ct[ltid] * g[s_gates + ltid];
			dg[i_gates + ltid] = dc[ltid] * g[c_gates + ltid] * g[s_gates + ltid];
			dg[f_gates + ltid] = -dc[ltid] * prev_c[ltid] * g[s_gates + ltid];
			dg[c_gates + ltid] = dc[ltid] * g[i_gates + ltid] * g[s_gates + ltid];
			dg[s_gates + ltid] = dc[ltid] * g[i_gates + ltid] * g[c_gates + ltid] +
								 dh[tid] * ct[ltid] * g[o_gates + ltid] +
								 -dc[ltid] * prev_c[ltid] * g[f_gates + ltid];
								 
			//propagate error back through gates, nonlinearities
			dg[i_gates + ltid] 	= 	dg[i_gates + ltid] * device_logistic_prime	( g[i_gates + ltid] );
			dg[o_gates + ltid] 	= 	dg[o_gates + ltid] * device_logistic_prime	( g[o_gates + ltid] );
			dg[f_gates + ltid] 	= 	dg[f_gates + ltid] * device_logistic_prime	( g[f_gates + ltid] );
			dg[c_gates + ltid] 	= 	dg[c_gates + ltid] * device_tanh_prime	( g[c_gates + ltid] );
			
			//softmax derivative
			dg[s_gates + ltid] = dg[s_gates + ltid] * ( g[s_gates + ltid] );
			total_prob += dg[s_gates + ltid];
			
			//carry - c state
			prev_dc[ltid] = dc[ltid] * ( ( dtype ) 1 - g[f_gates + ltid] * g[s_gates + ltid] );
			
		}
		
		//softmax derivative
		for ( int l = 0; l < L; l++ ) {
		
			int ltid = tid + l * N * B;
			dg[s_gates + ltid] -= g[s_gates + ltid] * total_prob;
			
		}
		
		/****** sparse2 ******/
		// carry - h state
		//prev_dh[tid] += dh[tid];
	}
	
	
}

#undef i_gates
#undef o_gates
#undef f_gates
#undef c_gates
#undef s_gates

#define i_gates 0 * N * B * L
#define o_gates 1 * N * B * L
#define f_gates 2 * N * B * L
#define c_gates 3 * N * B * L
#define s_gates 4 * N * B * L

__global__ void kernel_elementwise_hardattlstm_forward ( dtype *__restrict__ g,
		dtype *__restrict__ g2,
		dtype *__restrict__ G,
		dtype *__restrict__ b,
		dtype *__restrict__ h,
		dtype *__restrict__ max_o,
		dtype *__restrict__ c,
		dtype *__restrict__ ct,
		dtype *__restrict__ prev_c,
		dtype *__restrict__ prev_h,
		dtype *__restrict__ rands,
		size_t N, size_t L, size_t B ) {
		
	size_t elements = N * B;
	
	/* there are N * B threads */
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	
	/* in - gates after SGEMMs */
	
	if ( tid < elements ) {
	
		dtype cumsum[64];
		
		for ( int l = 0; l < L; l++ )
			cumsum[l] = 0;
			
		h[tid] = 0;
		dtype total_prob = ( dtype ) 0;
		
#pragma unroll
		
		for ( int l = 0; l < L; l++ ) {
		
			int ltid = tid + l * N * B;
			
			/* add bias */
			g[i_gates + ltid] 	+= g2[i_gates + ltid] + b[ ( i_gates + ltid ) / B];
			g[o_gates + ltid] 	+= g2[o_gates + ltid] + b[ ( o_gates + ltid ) / B];
			g[f_gates + ltid] 	+= g2[f_gates + ltid] + b[ ( f_gates + ltid ) / B];
			g[c_gates + ltid] 	+= g2[c_gates + ltid] + b[ ( c_gates + ltid ) / B];
			g[s_gates + ltid] 	+= g2[s_gates + ltid] + b[ ( s_gates + ltid ) / B];
			
			/* there are 4 * N * B gate activations */
			g[i_gates + ltid] 	= 	device_logistic	( g[i_gates + ltid] );
			g[o_gates + ltid] 	= 	device_logistic	( g[o_gates + ltid] );
			g[f_gates + ltid] 	= 	device_logistic	( g[f_gates + ltid] );
			g[c_gates + ltid] 	= 	device_tanh	( g[c_gates + ltid] );
			
			c[ltid] = prev_c[ltid];
			
			//for softmax
			total_prob += device_exp ( g[s_gates + ltid] );
			
		}
		
		int max_o_idx = tid;
		//dtype max_o_val = ( dtype ) - 1;
		
#pragma unroll
		
		//choose max
		for ( int l = 0; l < L; l++ ) {
		
			int ltid = tid + l * N * B;
			
			//softmax
			g[s_gates + ltid] = device_exp ( g[s_gates + ltid] ) / total_prob;
			
			//max
			// if (max_o_val < g[s_gates + ltid]) {
			
			// 	max_o_val = g[s_gates + ltid];
			// 	max_o_idx = ltid;
			
			// }
			
		}
		
		/* stoch */
		cumsum[0] = g[s_gates + tid];
		
		for ( int l = 1; l < L; l++ ) {
		
			int ltid = tid + l * N * B;
			
			cumsum[l] = g[s_gates + ltid] + cumsum[l - 1];
			
		}
		
		for ( int l = 0; l < L; l++ ) {
		
			if ( rands[tid] <= ( dtype ) ( l + 1 ) / ( dtype ) L ) {
			
				int ltid = tid + l * N * B;
				max_o_idx = ltid;
				break;
				
			}
		}
		
		max_o[tid] = ( dtype ) max_o_idx;
		
		// #pragma unroll
		// 		for (int l = 0; l < L; l++) {
		
		int ltid = max_o_idx;
		
		G[i_gates + ltid] 	=	( g[s_gates + ltid] ) * ( g[i_gates + ltid] );
		G[o_gates + ltid] 	=	( g[s_gates + ltid] ) * ( g[o_gates + ltid] );
		G[f_gates + ltid] 	=	( g[s_gates + ltid] ) * ( g[f_gates + ltid] );
		
		c[ltid] 				=	( ( dtype ) 1 - G[f_gates + ltid] ) * prev_c[ltid];
		c[ltid] 				+= 	G[i_gates + ltid] * g[c_gates + ltid];
		
		ct[ltid] 				= 	device_tanh ( c[ltid] );
		
		h[tid] 				+=	( G[o_gates + ltid] ) * ct[ltid];
		
		// }
		
		//h[tid] = device_tanh(h[tid]);
		
	}
	
	/* out - updated c and h */
	
	
}

__global__ void kernel_elementwise_hardattlstm_backward (
	dtype *__restrict__ dg,
	dtype *__restrict__ dh,
	dtype *__restrict__ c,
	dtype *__restrict__ ct,
	dtype *__restrict__ dc,
	dtype *__restrict__ g,
	dtype *__restrict__ prev_c,
	dtype *__restrict__ prev_dc,
	dtype *__restrict__ h,
	dtype *__restrict__ max_o,
	dtype *__restrict__ prev_h,
	dtype *__restrict__ prev_dh,
	size_t N, size_t L, size_t B ) {
	
	size_t elements = N * B;
	
	/* there are N * B threads */
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	
	dtype total_prob = ( dtype ) 0;
	
	if ( tid < elements ) {
	
		//dh[tid] = dh[tid] * device_tanh_prime		(h[tid]);
		// #pragma unroll
		// 		for (int l = 0; l < L; l++) {
		
		for ( int l = 0; l < L; l++ ) {
		
			int ltid = tid + l * N * B;
			prev_dc[ltid] = dc[ltid];
			
		}
		
		int ltid = ( int ) max_o[tid];
		
		//int ltid = tid + l * N * B;
		dc[ltid] = dc[ltid] + dh[tid] * g[o_gates + ltid] * g[s_gates + ltid] * device_tanh_prime ( ct[ltid] );
		//dc[ltid] = dc[ltid] * device_tanh_prime ( c[ltid] );
		
		//propagate error back through gates
		dg[o_gates + ltid] = dh[tid] * ct[ltid] * g[s_gates + ltid];
		dg[i_gates + ltid] = dc[ltid] * g[c_gates + ltid] * g[s_gates + ltid];
		dg[f_gates + ltid] = -dc[ltid] * prev_c[ltid] * g[s_gates + ltid];
		dg[c_gates + ltid] = dc[ltid] * g[i_gates + ltid] * g[s_gates + ltid];
		dg[s_gates + ltid] = dc[ltid] * g[i_gates + ltid] * g[c_gates + ltid] +
							 dh[tid] * ct[ltid] * g[o_gates + ltid] +
							 -dc[ltid] * prev_c[ltid] * g[f_gates + ltid];
							 
		//propagate error back through gates, nonlinearities
		dg[i_gates + ltid] 	= 	dg[i_gates + ltid] * device_logistic_prime	( g[i_gates + ltid] );
		dg[o_gates + ltid] 	= 	dg[o_gates + ltid] * device_logistic_prime	( g[o_gates + ltid] );
		dg[f_gates + ltid] 	= 	dg[f_gates + ltid] * device_logistic_prime	( g[f_gates + ltid] );
		dg[c_gates + ltid] 	= 	dg[c_gates + ltid] * device_tanh_prime	( g[c_gates + ltid] );
		
		//softmax derivative
		dg[s_gates + ltid] = dg[s_gates + ltid] * ( g[s_gates + ltid] );
		total_prob += dg[s_gates + ltid];
		
		//carry - c state
		prev_dc[ltid] = dc[ltid] * ( ( dtype ) 1 - g[f_gates + ltid] * g[s_gates + ltid] );
		
		//	}
		
		//softmax derivative
		for ( int l = 0; l < L; l++ ) {
		
			int ltid = tid + l * N * B;
			dg[s_gates + ltid] -= g[s_gates + ltid] * total_prob;
			
		}
		
		/****** sparse2 ******/
		// carry - h state
		//prev_dh[tid] += dh[tid];
	}
	
}

#undef i_gates
#undef o_gates
#undef f_gates
#undef c_gates
#undef s_gates

/***************/
/***************/

#define i_gates 0 * N * B * L
#define o_gates 1 * N * B * L
#define f_gates 2 * N * B * L
#define c_gates 3 * N * B * L
#define s_gates 4 * N * B * L

__global__ void kernel_elementwise_cmlstm_forward ( dtype *__restrict__ g,
		dtype *__restrict__ g2,
		dtype *__restrict__ G,
		dtype *__restrict__ b,
		dtype *__restrict__ h,
		dtype *__restrict__ c,
		dtype *__restrict__ prev_c,
		dtype *__restrict__ prev_h,
		dtype *__restrict__ rands,
		size_t N, size_t L, size_t B ) {
		
	size_t elements = N * B;
	
	/* there are N * B threads */
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	
	/* in - gates after SGEMMs */
	
	if ( tid < elements ) {
	
		h[tid] = 0;
		dtype total_prob = ( dtype ) 0;
		// dtype cumsum[64];
		
		// for (int l = 0; l < L; l++)
		// 	cumsum[l] = 0;
		
#pragma unroll
		
		for ( int l = 0; l < L; l++ ) {
		
			int ltid = tid + l * N * B;
			
			/* add bias */
			g[i_gates + ltid] 	+= g2[i_gates + ltid] + b[ ( i_gates + ltid ) / B];
			g[o_gates + ltid] 	+= g2[o_gates + ltid] + b[ ( o_gates + ltid ) / B];
			g[f_gates + ltid] 	+= g2[f_gates + ltid] + b[ ( f_gates + ltid ) / B];
			g[c_gates + ltid] 	+= g2[c_gates + ltid] + b[ ( c_gates + ltid ) / B];
			g[s_gates + ltid] 	+= g2[s_gates + ltid] + b[ ( s_gates + ltid ) / B];
			
			/* there are 4 * N * B gate activations */
			g[i_gates + ltid] 	= 	device_logistic	( g[i_gates + ltid] );
			g[o_gates + ltid] 	= 	device_logistic	( g[o_gates + ltid] );
			g[f_gates + ltid] 	= 	device_logistic	( g[f_gates + ltid] );
			g[c_gates + ltid] 	= 	device_tanh	( g[c_gates + ltid] );
			
			//for softmax
			total_prob += device_exp ( g[s_gates + ltid] );
			
		}
		
		// cumsum[0] = exp(g[s_gates + tid])/total_prob;
		
		// for (int l = 1; l < L; l++) {
		
		// 	int ltid = tid + l * N * B;
		
		// 	cumsum[l] = exp(g[s_gates + ltid])/total_prob + cumsum[l-1];
		
		// }
		
		for ( int l = 0; l < L; l++ ) {
		
			int ltid = tid + l * N * B;
			
			//softmax
			g[s_gates + ltid] = device_exp ( g[s_gates + ltid] ) / total_prob;
			
			//stochastic pass
			// G[i_gates + ltid] 	= 	(g[i_gates+ltid] > rands[i_gates+ltid]);
			// G[o_gates + ltid] 	= 	(g[o_gates+ltid] > rands[o_gates+ltid]);
			// G[f_gates + ltid] 	= 	(g[f_gates+ltid] > rands[f_gates+ltid]);
			
			// G[s_gates + ltid] 	= 	(g[s_gates+ltid]);
			
			G[i_gates + ltid] 	=	( g[s_gates + ltid] ) * ( g[i_gates + ltid] );
			G[o_gates + ltid] 	=	( g[s_gates + ltid] ) * ( g[o_gates + ltid] );
			G[f_gates + ltid] 	=	( g[s_gates + ltid] ) * ( g[f_gates + ltid] );
			
			c[ltid] 				=	( ( dtype ) 1 - G[f_gates + ltid] ) * prev_c[ltid];
			c[ltid] 				+= 	G[i_gates + ltid] * g[c_gates + ltid];
			
			c[ltid] 				= 	device_tanh ( c[ltid] );
			
			h[tid] 				+=	( G[o_gates + ltid] ) * c[ltid];
			
		}
		
		//h[tid] 	+= 	prev_h[tid];
		
		h[tid] = device_tanh ( h[tid] );
		
	}
	
	/* out - updated c and h */
	
}

__global__ void kernel_elementwise_cmlstm_backward (
	dtype *__restrict__ dg,
	dtype *__restrict__ dh,
	dtype *__restrict__ c,
	dtype *__restrict__ dc,
	dtype *__restrict__ g,
	dtype *__restrict__ prev_c,
	dtype *__restrict__ prev_dc,
	dtype *__restrict__ h,
	dtype *__restrict__ prev_h,
	dtype *__restrict__ prev_dh,
	size_t N, size_t L, size_t B ) {
	
	size_t elements = N * B;
	
	/* there are N * B threads */
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	
	dtype total_prob = ( dtype ) 0;
	
	if ( tid < elements ) {
	
		dh[tid] = dh[tid] * device_tanh_prime	( h[tid] );
#pragma unroll
		
		for ( int l = 0; l < L; l++ ) {
		
			int ltid = tid + l * N * B;
			dc[ltid] = dc[ltid] + dh[tid] * g[o_gates + ltid] * g[s_gates + ltid];
			dc[ltid] = dc[ltid] * device_tanh_prime ( c[ltid] );
			
			//propagate error back through gates
			dg[o_gates + ltid] = dh[tid] * c[ltid] * g[s_gates + ltid];
			dg[i_gates + ltid] = dc[ltid] * g[c_gates + ltid] * g[s_gates + ltid];
			dg[f_gates + ltid] = -dc[ltid] * prev_c[ltid] * g[s_gates + ltid];
			dg[c_gates + ltid] = dc[ltid] * g[i_gates + ltid] * g[s_gates + ltid];
			dg[s_gates + ltid] = dc[ltid] * g[i_gates + ltid] * g[c_gates + ltid] +
								 dh[tid] * c[ltid] * g[o_gates + ltid] +
								 -dc[ltid] * prev_c[ltid] * g[f_gates + ltid];
								 
			//propagate error back through gates, nonlinearities
			dg[i_gates + ltid] 	= 	dg[i_gates + ltid] * device_logistic_prime	( g[i_gates + ltid] );
			dg[o_gates + ltid] 	= 	dg[o_gates + ltid] * device_logistic_prime	( g[o_gates + ltid] );
			dg[f_gates + ltid] 	= 	dg[f_gates + ltid] * device_logistic_prime	( g[f_gates + ltid] );
			dg[c_gates + ltid] 	= 	dg[c_gates + ltid] * device_tanh_prime	( g[c_gates + ltid] );
			
			//softmax derivative
			dg[s_gates + ltid] = dg[s_gates + ltid] * ( g[s_gates + ltid] );
			total_prob += dg[s_gates + ltid];
			
			//carry - c state
			prev_dc[ltid] = dc[ltid] * ( ( dtype ) 1 - g[f_gates + ltid] * g[s_gates + ltid] );
			
		}
		
		//softmax derivative
		for ( int l = 0; l < L; l++ ) {
		
			int ltid = tid + l * N * B;
			dg[s_gates + ltid] -= g[s_gates + ltid] * total_prob;
			
		}
		
		/****** sparse2 ******/
		// carry - h state
		//prev_dh[tid] += dh[tid];
	}
	
	
}

#undef i_gates
#undef o_gates
#undef f_gates
#undef c_gates
#undef s_gates

/***************/

#define i_gates 0 * N * B * L
#define o_gates 1 * N * B * L
#define f_gates 2 * N * B * L
#define c_gates 3 * N * B * L

__global__ void kernel_elementwise_hlstm_forward (	dtype *__restrict__ g,
		dtype *__restrict__ g2,
		dtype *__restrict__ b,
		dtype *__restrict__ h,
		dtype *__restrict__ c,
		dtype *__restrict__ ct,
		dtype *__restrict__ prev_c,
		size_t N, size_t L, size_t B ) {
		
	size_t elements = N * B;
	
	/* there are N * B threads */
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	
	/* in - gates after SGEMMs */
	
	if ( tid < elements ) {
	
		h[tid] = 0;
		int l;
#pragma unroll
		
		for ( int l = 0; l < L; l++ ) {
		
			int ltid = tid + l * N * B;
			
			/* add bias */
			g[i_gates + ltid] 	+= g2[i_gates + ltid] + b[ ( i_gates + ltid ) / B];
			g[o_gates + ltid] 	+= g2[o_gates + ltid] + b[ ( o_gates + ltid ) / B];
			g[f_gates + ltid] 	+= g2[f_gates + ltid] + b[ ( f_gates + ltid ) / B];
			g[c_gates + ltid] 	+= g2[c_gates + ltid] + b[ ( c_gates + ltid ) / B];
			
			/* there are 4 * N * B gate activation */
			
			g[i_gates + ltid] 	= 	device_logistic	( g[i_gates + ltid] );
			g[o_gates + ltid] 	= 	device_logistic	( g[o_gates + ltid] );
			g[f_gates + ltid] 	= 	device_logistic	( g[f_gates + ltid] );
			g[c_gates + ltid] 	= 	device_tanh	( g[c_gates + ltid] );
			
			c[ltid] 			= 	g[f_gates + ltid] * prev_c[ltid];
			
		}
		
		// up
		for ( int l = 1; l < L; l++ ) {
		
			int ltid = tid + l * N * B;
			int ltid_prev = tid + ( l - 1 ) * N * B;
			c[ltid] += 	g[i_gates + ltid] * prev_c[ltid_prev];
			
		}
		
		// lowest level
		l = 0;
		int ltid = tid + l * N * B;
		c[ltid] += 	g[i_gates + ltid] * g[c_gates + ltid];
		
		
		// down
		for ( int l = 0; l < L - 1; l++ ) {
		
			int ltid = tid + l * N * B;
			int ltid_prev = tid + ( l + 1 ) * N * B;
			c[ltid] += 	g[o_gates + ltid_prev] * prev_c[ltid_prev];
		}
		
		for ( int l = 0; l < L; l++ ) {
			int ltid = tid + l * N * B;
			c[ltid] = 	device_tanh ( c[ltid] );
		}
		
		l = 0;
		ltid = tid + l * N * B;
		
		// lowest level
		h[tid] 					= 	g[o_gates + ltid] * c[ltid];
		
		
		
	}
	
	/* out - updated c and h */
	
}

__global__ void kernel_elementwise_hlstm_backward (
	dtype *__restrict__ dg,
	dtype *__restrict__ dh,
	dtype *__restrict__ c,
	dtype *__restrict__ ct,
	dtype *__restrict__ dc,
	dtype *__restrict__ g,
	dtype *__restrict__ prev_c,
	dtype *__restrict__ prev_dc,
	dtype *__restrict__ h,
	size_t N, size_t L, size_t B ) {
	
	size_t elements = N * B;
	
	/* there are N * B threads */
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	
	if ( tid < elements ) {
	
		//dh[tid] = dh[tid] * device_tanh_prime		(h[tid]);
		//#pragma unroll
		
		int l;
		
		l = 0;
		int ltid = tid + l * N * B;
		dc[ltid] = dc[ltid] + dh[tid] * g[o_gates + ltid];
		
		for ( l = 0; l < L; l++ ) {
		
			int ltid = tid + l * N * B;
			dc[ltid] = dc[ltid] * device_tanh_prime ( c[ltid] );
			
		}
		
		l = 0;
		ltid = tid + l * N * B;
		dg[c_gates + ltid] = dc[ltid] * g[i_gates + ltid];
		dg[i_gates + ltid] = dc[ltid] * g[c_gates + ltid];
		dg[o_gates + ltid] = dh[ltid] * c[ltid];
		
		for ( l = 1; l < L; l++ ) {
			int ltid = tid + l * N * B;
			int ltid_prev = tid + ( l - 1 ) * N * B;
			dg[i_gates + ltid] = prev_c[ltid_prev] * dc[ltid];
			dg[o_gates + ltid] = prev_c[ltid] * dc[ltid_prev];
		}
		
		// for (int l = 0; l < L-1; l++) {
		
		// 	int ltid = tid + l * N * B;
		// 	int ltid_prev = tid + (l+1) * N * B;
		// 	dg[o_gates + ltid_prev] = prev_c[ltid_prev] * dc[ltid];
		// }
		
		for ( l = 0; l < L; l++ ) {
			int ltid = tid + l * N * B;
			
			//propagate error back through gates
			
			dg[f_gates + ltid] = dc[ltid] * prev_c[ltid];
			
			//propagate error back through gates, nonlinearities
			dg[i_gates + ltid] 	= 	dg[i_gates + ltid] * device_logistic_prime	( g[i_gates + ltid] );
			dg[o_gates + ltid] 	= 	dg[o_gates + ltid] * device_logistic_prime	( g[o_gates + ltid] );
			dg[f_gates + ltid] 	= 	dg[f_gates + ltid] * device_logistic_prime	( g[f_gates + ltid] );
			dg[c_gates + ltid] 	= 	dg[c_gates + ltid] * device_tanh_prime	( g[c_gates + ltid] );
			
			//carry - c state
			prev_dc[ltid] = dc[ltid] * g[f_gates + ltid];
			
		}
		
		for ( l = 1; l < L; l++ ) {
		
			int ltid = tid + l * N * B;
			int ltid_prev = tid + ( l - 1 ) * N * B;
			prev_dc[ltid] += dc[ltid_prev] * g[o_gates + ltid];
			
		}
		
		for ( l = 0; l < L - 1; l++ ) {
		
			int ltid = tid + l * N * B;
			int ltid_prev = tid + ( l + 1 ) * N * B;
			prev_dc[ltid] += dc[ltid_prev] * g[i_gates + ltid_prev];
			
		}
	}
	
	
}

#undef i_gates
#undef o_gates
#undef f_gates
#undef c_gates

/*** hclstm */

#define i_gates 0 * N * B * L
#define o_gates 1 * N * B * L
#define f_gates 2 * N * B * L
#define c_gates 3 * N * B * L

__global__ void kernel_elementwise_hclstm_forward (	dtype *__restrict__ g,
		dtype *__restrict__ g2,
		dtype *__restrict__ b,
		dtype *__restrict__ h,
		dtype *__restrict__ c,
		dtype *__restrict__ prev_c,
		size_t N, size_t L, size_t B ) {
		
	size_t elements = N * B;
	
	/* there are N * B threads */
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	
	/* in - gates after SGEMMs */
	
	if ( tid < elements ) {
	
		h[tid] = 0;
		int l;
#pragma unroll
		
		for ( int l = 0; l < L; l++ ) {
		
			int ltid = tid + l * N * B;
			
			/* add bias */
			g[i_gates + ltid] 	+= g2[i_gates + ltid] + b[ ( i_gates + ltid ) / B];
			g[o_gates + ltid] 	+= g2[o_gates + ltid] + b[ ( o_gates + ltid ) / B];
			g[f_gates + ltid] 	+= g2[f_gates + ltid] + b[ ( f_gates + ltid ) / B];
			g[c_gates + ltid] 	+= g2[c_gates + ltid] + b[ ( c_gates + ltid ) / B];
			
			/* there are 4 * N * B gate activation */
			
			g[i_gates + ltid] 	= 	device_logistic	( g[i_gates + ltid] );
			g[o_gates + ltid] 	= 	device_logistic	( g[o_gates + ltid] );
			g[f_gates + ltid] 	= 	device_logistic	( g[f_gates + ltid] );
			g[c_gates + ltid] 	= 	device_tanh	( g[c_gates + ltid] );
			
			c[ltid] 			= 	g[f_gates + ltid] * prev_c[ltid];
			
		}
		
		// up
		for ( int l = 1; l < L; l++ ) {
		
			int ltid = tid + l * N * B;
			int ltid_prev = tid + ( l - 1 ) * N * B;
			c[ltid] += 	g[i_gates + ltid] * prev_c[ltid_prev];
			
		}
		
		// lowest level
		l = 0;
		int ltid = tid + l * N * B;
		c[ltid] += 	g[i_gates + ltid] * g[c_gates + ltid];
		
		
		// down
		// for (int l = 0; l < L-1; l++) {
		
		// 	int ltid = tid + l * N * B;
		// 	int ltid_prev = tid + (l+1) * N * B;
		// 	c[ltid] += 	g[o_gates + ltid_prev] * prev_c[ltid_prev];
		// }
		
		for ( int l = 0; l < L; l++ ) {
			int ltid = tid + l * N * B;
			c[ltid] = 	device_tanh ( c[ltid] );
		}
		
		h[tid] = 0;
		
		for ( int l = 0; l < L; l++ ) {
		
			int ltid = tid + l * N * B;
			h[tid] 				+= 	g[o_gates + ltid] * c[ltid];
		}
		
		h[tid] = device_tanh ( h[tid] );
		
		// l = 0;
		// ltid = tid + l * N * B;
		// // lowest level
		// h[tid] 					= 	g[o_gates + ltid] * c[ltid];
		
		
		
	}
	
	/* out - updated c and h */
	
}

__global__ void kernel_elementwise_hclstm_backward (
	dtype *__restrict__ dg,
	dtype *__restrict__ dh,
	dtype *__restrict__ c,
	dtype *__restrict__ dc,
	dtype *__restrict__ g,
	dtype *__restrict__ prev_c,
	dtype *__restrict__ prev_dc,
	dtype *__restrict__ h,
	size_t N, size_t L, size_t B ) {
	
	size_t elements = N * B;
	
	/* there are N * B threads */
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	
	if ( tid < elements ) {
	
		dh[tid] = dh[tid] * device_tanh_prime	( h[tid] );
		//#pragma unroll
		
		int l;
		
		// l = 0;
		// int ltid = tid + l * N * B;
		// dc[ltid] = dc[ltid] + dh[tid] * g[o_gates + ltid];
		
		for ( l = 0; l < L; l++ ) {
		
			int ltid = tid + l * N * B;
			dc[ltid] = dc[ltid] + dh[tid] * g[o_gates + ltid];
			dc[ltid] = dc[ltid] * device_tanh_prime ( c[ltid] );
			dg[o_gates + ltid] = dh[tid] * c[ltid];
			
		}
		
		l = 0;
		int ltid = tid + l * N * B;
		dg[c_gates + ltid] = dc[ltid] * g[i_gates + ltid];
		dg[i_gates + ltid] = dc[ltid] * g[c_gates + ltid];
		//dg[o_gates + ltid] = dh[ltid] * c[ltid];
		
		for ( l = 1; l < L; l++ ) {
			int ltid = tid + l * N * B;
			int ltid_prev = tid + ( l - 1 ) * N * B;
			dg[i_gates + ltid] = prev_c[ltid_prev] * dc[ltid];
			// dg[o_gates + ltid] = prev_c[ltid] * dc[ltid_prev];
		}
		
		// for (int l = 0; l < L-1; l++) {
		
		// 	int ltid = tid + l * N * B;
		// 	int ltid_prev = tid + (l+1) * N * B;
		// 	dg[o_gates + ltid_prev] = prev_c[ltid_prev] * dc[ltid];
		// }
		
		for ( l = 0; l < L; l++ ) {
			int ltid = tid + l * N * B;
			
			//propagate error back through gates
			
			dg[f_gates + ltid] = dc[ltid] * prev_c[ltid];
			
			//propagate error back through gates, nonlinearities
			dg[i_gates + ltid] 	= 	dg[i_gates + ltid] * device_logistic_prime	( g[i_gates + ltid] );
			dg[o_gates + ltid] 	= 	dg[o_gates + ltid] * device_logistic_prime	( g[o_gates + ltid] );
			dg[f_gates + ltid] 	= 	dg[f_gates + ltid] * device_logistic_prime	( g[f_gates + ltid] );
			dg[c_gates + ltid] 	= 	dg[c_gates + ltid] * device_tanh_prime	( g[c_gates + ltid] );
			
			//carry - c state
			prev_dc[ltid] = dc[ltid] * g[f_gates + ltid];
			
		}
		
		// for (l = 1; l < L; l++) {
		
		// 	int ltid = tid + l * N * B;
		// 	int ltid_prev = tid + (l-1) * N * B;
		// 	prev_dc[ltid] += dc[ltid_prev] * g[o_gates + ltid];
		
		// }
		
		for ( l = 0; l < L - 1; l++ ) {
		
			int ltid = tid + l * N * B;
			int ltid_prev = tid + ( l + 1 ) * N * B;
			prev_dc[ltid] += dc[ltid_prev] * g[i_gates + ltid_prev];
			
		}
	}
	
	
}

#undef i_gates
#undef o_gates
#undef f_gates
#undef c_gates

/****************************************/

/*hmlstm */

#define i_gates 0 * N * B * L
#define o_gates 1 * N * B * L
#define f_gates 2 * N * B * L
#define c_gates 3 * N * B * L

__global__ void kernel_elementwise_hmlstm_forward (	dtype *__restrict__ g,
		dtype *__restrict__ g2,
		dtype *__restrict__ b,
		dtype *__restrict__ h,
		dtype *__restrict__ c,
		dtype *__restrict__ prev_c,
		size_t N, size_t L, size_t B ) {
		
	size_t elements = N * B;
	
	/* there are N * B threads */
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	
	/* in - gates after SGEMMs */
	
	if ( tid < elements ) {
	
		h[tid] = 1;
		int l;
#pragma unroll
		
		for ( int l = 0; l < L; l++ ) {
		
			int ltid = tid + l * N * B;
			
			/* add bias */
			g[i_gates + ltid] 	+= g2[i_gates + ltid] + b[ ( i_gates + ltid ) / B];
			g[o_gates + ltid] 	+= g2[o_gates + ltid] + b[ ( o_gates + ltid ) / B];
			g[f_gates + ltid] 	+= g2[f_gates + ltid] + b[ ( f_gates + ltid ) / B];
			g[c_gates + ltid] 	+= g2[c_gates + ltid] + b[ ( c_gates + ltid ) / B];
			
			/* there are 4 * N * B gate activation */
			
			g[i_gates + ltid] 	= 	device_logistic	( g[i_gates + ltid] );
			g[o_gates + ltid] 	= 	device_logistic	( g[o_gates + ltid] );
			g[f_gates + ltid] 	= 	device_logistic	( g[f_gates + ltid] );
			g[c_gates + ltid] 	= 	device_tanh	( g[c_gates + ltid] );
			
			c[ltid] 			= 	g[f_gates + ltid] * prev_c[ltid];
			
		}
		
		// up
		for ( int l = 1; l < L; l++ ) {
		
			int ltid = tid + l * N * B;
			int ltid_prev = tid + ( l - 1 ) * N * B;
			c[ltid] += 	g[i_gates + ltid] * prev_c[ltid_prev];
			
		}
		
		// lowest level
		l = 0;
		int ltid = tid + l * N * B;
		c[ltid] += 	g[i_gates + ltid] * g[c_gates + ltid];
		
		
		// down
		// for (int l = 0; l < L-1; l++) {
		
		// 	int ltid = tid + l * N * B;
		// 	int ltid_prev = tid + (l+1) * N * B;
		// 	c[ltid] += 	g[o_gates + ltid_prev] * prev_c[ltid_prev];
		// }
		
		for ( int l = 0; l < L; l++ ) {
			int ltid = tid + l * N * B;
			c[ltid] = 	device_tanh ( c[ltid] );
		}
		
		//h[tid] = 0;
		
		for ( int l = 0; l < L; l++ ) {
		
			int ltid = tid + l * N * B;
			h[tid] *= 	g[o_gates + ltid] * c[ltid];
		}
		
		//h[tid] = device_tanh(h[tid]);
		
		// l = 0;
		// ltid = tid + l * N * B;
		// // lowest level
		// h[tid] 					= 	g[o_gates + ltid] * c[ltid];
		
		
		
	}
	
	/* out - updated c and h */
	
}

__global__ void kernel_elementwise_hmlstm_backward (
	dtype *__restrict__ dg,
	dtype *__restrict__ dh,
	dtype *__restrict__ c,
	dtype *__restrict__ dc,
	dtype *__restrict__ g,
	dtype *__restrict__ prev_c,
	dtype *__restrict__ prev_dc,
	dtype *__restrict__ h,
	size_t N, size_t L, size_t B ) {
	
	size_t elements = N * B;
	
	/* there are N * B threads */
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	
	if ( tid < elements ) {
	
		//dh[tid] = dh[tid] * device_tanh_prime		(h[tid]);
		//#pragma unroll
		
		int l;
		
		// l = 0;
		// int ltid = tid + l * N * B;
		// dc[ltid] = dc[ltid] + dh[tid] * g[o_gates + ltid];
		
		l = 0;
		
		int ltid = tid + l * N * B;
		int otid = tid + 1 * N * B;
		
		dc[ltid] = dc[ltid] + dh[tid] * g[o_gates + ltid] * c[otid] * g[o_gates + otid];
		dg[o_gates + ltid] = dh[tid] * c[ltid] * c[otid] * g[o_gates + otid];
		
		l = 1;
		
		ltid = tid + l * N * B;
		otid = tid + 0 * N * B;
		dc[ltid] = dc[ltid] + dh[tid] * g[o_gates + ltid] * c[otid] * g[o_gates + otid];
		dg[o_gates + ltid] = dh[tid] * c[ltid] * c[otid] * g[o_gates + otid];
		
		for ( l = 0; l < L; l++ ) {
		
			int ltid = tid + l * N * B;
			//dc[ltid] = dc[ltid] + dh[tid] * g[o_gates + ltid];
			dc[ltid] = dc[ltid] * device_tanh_prime ( c[ltid] );
			//dg[o_gates + ltid] = dh[tid] * c[ltid];
			
		}
		
		l = 0;
		ltid = tid + l * N * B;
		dg[c_gates + ltid] = dc[ltid] * g[i_gates + ltid];
		dg[i_gates + ltid] = dc[ltid] * g[c_gates + ltid];
		//dg[o_gates + ltid] = dh[ltid] * c[ltid];
		
		for ( l = 1; l < L; l++ ) {
			int ltid = tid + l * N * B;
			int ltid_prev = tid + ( l - 1 ) * N * B;
			dg[i_gates + ltid] = prev_c[ltid_prev] * dc[ltid];
			// dg[o_gates + ltid] = prev_c[ltid] * dc[ltid_prev];
		}
		
		// for (int l = 0; l < L-1; l++) {
		
		// 	int ltid = tid + l * N * B;
		// 	int ltid_prev = tid + (l+1) * N * B;
		// 	dg[o_gates + ltid_prev] = prev_c[ltid_prev] * dc[ltid];
		// }
		
		for ( l = 0; l < L; l++ ) {
			int ltid = tid + l * N * B;
			
			//propagate error back through gates
			
			dg[f_gates + ltid] = dc[ltid] * prev_c[ltid];
			
			//propagate error back through gates, nonlinearities
			dg[i_gates + ltid] 	= 	dg[i_gates + ltid] * device_logistic_prime	( g[i_gates + ltid] );
			dg[o_gates + ltid] 	= 	dg[o_gates + ltid] * device_logistic_prime	( g[o_gates + ltid] );
			dg[f_gates + ltid] 	= 	dg[f_gates + ltid] * device_logistic_prime	( g[f_gates + ltid] );
			dg[c_gates + ltid] 	= 	dg[c_gates + ltid] * device_tanh_prime	( g[c_gates + ltid] );
			
			//carry - c state
			prev_dc[ltid] = dc[ltid] * g[f_gates + ltid];
			
		}
		
		// for (l = 1; l < L; l++) {
		
		// 	int ltid = tid + l * N * B;
		// 	int ltid_prev = tid + (l-1) * N * B;
		// 	prev_dc[ltid] += dc[ltid_prev] * g[o_gates + ltid];
		
		// }
		
		for ( l = 0; l < L - 1; l++ ) {
		
			int ltid = tid + l * N * B;
			int ltid_prev = tid + ( l + 1 ) * N * B;
			prev_dc[ltid] += dc[ltid_prev] * g[i_gates + ltid_prev];
			
		}
	}
	
	
}

#undef i_gates
#undef o_gates
#undef f_gates
#undef c_gates

/*******/

void cu_elementwise_hhlstm_forward ( dtype *__restrict__ g, dtype *__restrict__ g2, dtype *__restrict__ b,
									 dtype *__restrict__ h,
									 dtype *__restrict__ c,
									 dtype *__restrict__ prev_c,
									 size_t N, size_t L, size_t B, int stream_idx ) {
									 
	size_t num_blocks = ( N * B + NUM_THREADS - 1 ) / NUM_THREADS;
	kernel_elementwise_hhlstm_forward <<< num_blocks, NUM_THREADS, stream_idx>>> ( g, g2, b, h, c, prev_c, N, L, B );
	
}

void cu_elementwise_hhlstm_backward ( dtype *__restrict__ dg,
									  dtype *__restrict__ dh,
									  dtype *__restrict__ c,
									  dtype *__restrict__ dc,
									  dtype *__restrict__ g,
									  dtype *__restrict__ prev_c,
									  dtype *__restrict__ prev_dc,
									  dtype *__restrict__ h,
									  size_t N, size_t L, size_t B, int stream_idx ) {
									  
									  
	size_t num_blocks = ( N * B + NUM_THREADS - 1 ) / NUM_THREADS;
	kernel_elementwise_hhlstm_backward <<< num_blocks, NUM_THREADS, stream_idx>>> ( dg, dh, c, dc, g, prev_c, prev_dc, h,
			N,
			L, B );
			
}

#define i_gates 0 * N * B
#define o_gates 1 * N * B
#define f_gates 2 * N * B
#define c_gates 3 * N * B

__global__ void kernel_elementwise_hhlstm_forward (	dtype *__restrict__ g,
		dtype *__restrict__ g2,
		dtype *__restrict__ b,
		dtype *__restrict__ h,
		dtype *__restrict__ c,
		dtype *__restrict__ prev_c,
		size_t N, size_t L, size_t B ) {
		
	size_t elements = N * B;
	
	/* there are N * B threads */
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	
	/* in - gates after SGEMMs */
	
	if ( tid < elements ) {
	
		h[tid] = 0;
		
		/* add bias */
		g[i_gates + tid] 	+= g2[i_gates + tid] + b[ ( i_gates + tid ) / B];
		g[o_gates + tid] 	+= g2[o_gates + tid] + b[ ( o_gates + tid ) / B];
		g[f_gates + tid] 	+= g2[f_gates + tid] + b[ ( f_gates + tid ) / B];
		g[c_gates + tid] 	+= g2[c_gates + tid] + b[ ( c_gates + tid ) / B];
		
		/* there are 4 * N * B gate activation */
		
		g[i_gates + tid] 	= 	device_logistic	( g[i_gates + tid] );
		g[o_gates + tid] 	= 	device_logistic	( g[o_gates + tid] );
		g[f_gates + tid] 	= 	device_logistic	( g[f_gates + tid] );
		g[c_gates + tid] 	= 	device_tanh	( g[c_gates + tid] );
		
		int l;
		
		for ( l = 0; l < L; l++ ) {
		
			int ltid = tid + l * N * B;
			
			c[ltid] 			= 	g[f_gates + tid] * prev_c[ltid];
			
		}
		
		// up
		for ( int l = 1; l < L; l++ ) {
		
			int ltid = tid + l * N * B;
			int ltid_prev = tid + ( l - 1 ) * N * B;
			c[ltid] += 	g[i_gates + tid] * prev_c[ltid_prev];
			
		}
		
		// lowest level
		c[tid] += 	g[i_gates + tid] * g[c_gates + tid];
		
		
		// down
		for ( int l = 0; l < L - 1; l++ ) {
		
			int ltid = tid + l * N * B;
			int ltid_prev = tid + ( l + 1 ) * N * B;
			c[ltid] += 	g[o_gates + tid] * prev_c[ltid_prev];
		}
		
		for ( int l = 0; l < L; l++ ) {
			int ltid = tid + l * N * B;
			c[ltid] = 	device_tanh ( c[ltid] );
		}
		
		// lowest level
		h[tid] 					= 	g[o_gates + tid] * c[tid];
		
	}
	
	/* out - updated c and h */
	
}

__global__ void kernel_elementwise_hhlstm_backward (
	dtype *__restrict__ dg,
	dtype *__restrict__ dh,
	dtype *__restrict__ c,
	dtype *__restrict__ dc,
	dtype *__restrict__ g,
	dtype *__restrict__ prev_c,
	dtype *__restrict__ prev_dc,
	dtype *__restrict__ h,
	size_t N, size_t L, size_t B ) {
	
	size_t elements = N * B;
	
	/* there are N * B threads */
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	
	if ( tid < elements ) {
	
		dc[tid] = dc[tid] + dh[tid] * g[o_gates + tid];
		
		for ( int l = 0; l < L; l++ ) {
		
			int ltid = tid + l * N * B;
			dc[ltid] = dc[ltid] * device_tanh_prime ( c[ltid] );
			
		}
		
		dg[c_gates + tid] = dc[tid] * g[i_gates + tid];
		dg[i_gates + tid] = dc[tid] * g[c_gates + tid];
		dg[o_gates + tid] = dh[tid] * c[tid];
		
		for ( int l = 1; l < L; l++ ) {
			int ltid = tid + l * N * B;
			int ltid_prev = tid + ( l - 1 ) * N * B;
			dg[i_gates + tid] += prev_c[ltid_prev] * dc[ltid];
			dg[o_gates + tid] += prev_c[ltid] * dc[ltid_prev];
		}
		
		dg[f_gates + tid] = 0;
		
		for ( int l = 0; l < L; l++ ) {
			int ltid = tid + l * N * B;
			dg[f_gates + tid] += dc[ltid] * prev_c[ltid];
		}
		
		//propagate error back through gates, nonlinearities
		dg[i_gates + tid] 	= 	dg[i_gates + tid] * device_logistic_prime	( g[i_gates + tid] );
		dg[o_gates + tid] 	= 	dg[o_gates + tid] * device_logistic_prime	( g[o_gates + tid] );
		dg[f_gates + tid] 	= 	dg[f_gates + tid] * device_logistic_prime	( g[f_gates + tid] );
		dg[c_gates + tid] 	= 	dg[c_gates + tid] * device_tanh_prime	( g[c_gates + tid] );
		
		for ( int l = 0; l < L; l++ ) {
			int ltid = tid + l * N * B;
			
			//carry - c state
			prev_dc[ltid] = dc[ltid] * g[f_gates + tid];
			
		}
		
		for ( int l = 1; l < L; l++ ) {
		
			int ltid = tid + l * N * B;
			int ltid_prev = tid + ( l - 1 ) * N * B;
			prev_dc[ltid] += dc[ltid_prev] * g[o_gates + tid];
			
		}
		
		for ( int l = 0; l < L - 1; l++ ) {
		
			int ltid = tid + l * N * B;
			int ltid_prev = tid + ( l + 1 ) * N * B;
			prev_dc[ltid] += dc[ltid_prev] * g[i_gates + tid];
			
		}
	}
	
	
}

#undef i_gates
#undef o_gates
#undef f_gates
#undef c_gates

/****************************************/

void cu_elementwise_plstm_forward ( dtype *__restrict__ g, dtype *__restrict__ g2, dtype *__restrict__ b,
									dtype *__restrict__ h,
									dtype *__restrict__ max_o,
									dtype *__restrict__ c,
									dtype *__restrict__ prev_c,
									size_t N, size_t L, size_t B, int stream_idx ) {
									
	size_t num_blocks = ( N * B + NUM_THREADS - 1 ) / NUM_THREADS;
	kernel_elementwise_plstm_forward <<< num_blocks, NUM_THREADS, stream_idx>>> ( g, g2, b, h, max_o, c, prev_c, N, L, B );
	
}

void cu_elementwise_plstm_backward ( dtype *__restrict__ dg,
									 dtype *__restrict__ dh,
									 dtype *__restrict__ c,
									 dtype *__restrict__ dc,
									 dtype *__restrict__ g,
									 dtype *__restrict__ prev_c,
									 dtype *__restrict__ prev_dc,
									 dtype *__restrict__ h,
									 dtype *__restrict__ max_o,
									 size_t N, size_t L, size_t B, int stream_idx ) {
									 
									 
	size_t num_blocks = ( N * B + NUM_THREADS - 1 ) / NUM_THREADS;
	kernel_elementwise_plstm_backward <<< num_blocks, NUM_THREADS, stream_idx>>> ( dg, dh, c, dc, g, prev_c, prev_dc, h,
			max_o,  N, L, B );
			
}

#define i_gates 0 * N * B * L
#define o_gates 1 * N * B * L
#define f_gates 2 * N * B * L
#define c_gates 3 * N * B * L

__global__ void kernel_elementwise_plstm_forward (	dtype *__restrict__ g,
		dtype *__restrict__ g2,
		dtype *__restrict__ b,
		dtype *__restrict__ h,
		dtype *__restrict__ max_o,
		dtype *__restrict__ c,
		dtype *__restrict__ prev_c,
		size_t N, size_t L, size_t B ) {
		
	size_t elements = N * B;
	
	/* there are N * B threads */
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	
	/* in - gates after SGEMMs */
	
	if ( tid < elements ) {
	
		h[tid] = 0;
		
		int max_o_idx = tid;
		dtype max_o_val = ( dtype ) - 1;
		
#pragma unroll
		
		for ( int l = 0; l < L; l++ ) {
		
			int ltid = tid + l * N * B;
			
			/* add bias */
			g[i_gates + ltid] 	+= g2[i_gates + ltid] + b[ ( i_gates + ltid ) / B];
			g[o_gates + ltid] 	+= g2[o_gates + ltid] + b[ ( o_gates + ltid ) / B];
			g[f_gates + ltid] 	+= g2[f_gates + ltid] + b[ ( f_gates + ltid ) / B];
			g[c_gates + ltid] 	+= g2[c_gates + ltid] + b[ ( c_gates + ltid ) / B];
			
			/* there are 4 * N * B gate activation */
			
			g[i_gates + ltid] 	= 	device_logistic	( g[i_gates + ltid] );
			g[o_gates + ltid] 	= 	device_logistic	( g[o_gates + ltid] );
			g[f_gates + ltid] 	= 	device_logistic	( g[f_gates + ltid] );
			g[c_gates + ltid] 	= 	device_tanh	( g[c_gates + ltid] );
			
			c[ltid] 				= 	g[f_gates + ltid] * prev_c[ltid];
			c[ltid] 				+= 	g[i_gates + ltid] * g[c_gates + ltid];
			
			c[ltid] 				= 	device_tanh ( c[ltid] );
			
			//h[tid] 				+= 	g[o_gates + ltid] * c[ltid];
			if ( max_o_val < g[o_gates + ltid] ) {
			
				max_o_val = g[o_gates + ltid];
				max_o_idx = ltid;
				
			}
		}
		
		max_o[tid] = ( dtype ) max_o_idx;
		h[tid] = g[o_gates + max_o_idx] * c[max_o_idx];
		//h[tid] = device_tanh(h[tid]);
	}
	
	/* out - updated c and h */
	
}

__global__ void kernel_elementwise_plstm_backward (
	dtype *__restrict__ dg,
	dtype *__restrict__ dh,
	dtype *__restrict__ c,
	dtype *__restrict__ dc,
	dtype *__restrict__ g,
	dtype *__restrict__ prev_c,
	dtype *__restrict__ prev_dc,
	dtype *__restrict__ h,
	dtype *__restrict__ max_o,
	size_t N, size_t L, size_t B ) {
	
	size_t elements = N * B;
	
	/* there are N * B threads */
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	
	if ( tid < elements ) {
	
		//dh[tid] = dh[tid] * device_tanh_prime		(h[tid]);
		//#pragma unroll
		//for (int l = 0; l < L; l++) {
		
		int ltid = ( int ) max_o[tid];
		dc[ltid] = dc[ltid] + dh[tid] * g[o_gates + ltid];
		dg[o_gates + ltid] = dh[tid] * c[ltid];
		
		for ( int l = 0; l < L; l++ ) {
		
			int ltid = tid + l * N * B;
			dc[ltid] = dc[ltid] * device_tanh_prime ( c[ltid] );
			
			//propagate error back through gates
			
			dg[i_gates + ltid] = dc[ltid] * g[c_gates + ltid];
			dg[f_gates + ltid] = dc[ltid] * prev_c[ltid];
			dg[c_gates + ltid] = dc[ltid] * g[i_gates + ltid];
			
			//propagate error back through gates, nonlinearities
			dg[i_gates + ltid] 	= 	dg[i_gates + ltid] * device_logistic_prime	( g[i_gates + ltid] );
			dg[o_gates + ltid] 	= 	dg[o_gates + ltid] * device_logistic_prime	( g[o_gates + ltid] );
			dg[f_gates + ltid] 	= 	dg[f_gates + ltid] * device_logistic_prime	( g[f_gates + ltid] );
			dg[c_gates + ltid] 	= 	dg[c_gates + ltid] * device_tanh_prime	( g[c_gates + ltid] );
			
			//carry - c state
			prev_dc[ltid] = dc[ltid] * g[f_gates + ltid];
			
		}
	}
	
	
}

#undef i_gates
#undef o_gates
#undef f_gates
#undef c_gates

/****************************************/

/****************************************/

void cu_elementwise_alstm_forward ( dtype *__restrict__ g, dtype *__restrict__ g2, dtype *__restrict__ b,
									dtype *__restrict__ h,
									dtype *__restrict__ max_o,
									dtype *__restrict__ c,
									dtype *__restrict__ ct,
									dtype *__restrict__ prev_c,
									dtype *__restrict__ rands,
									size_t N, size_t L, size_t B, int stream_idx ) {
									
	size_t num_blocks = ( N * B + NUM_THREADS - 1 ) / NUM_THREADS;
	kernel_elementwise_alstm_forward <<< num_blocks, NUM_THREADS, stream_idx>>> ( g, g2, b, h, max_o, c, ct, prev_c, rands,
			N, L, B );
			
}

void cu_elementwise_alstm_backward ( dtype *__restrict__ dg,
									 dtype *__restrict__ dh,
									 dtype *__restrict__ c,
									 dtype *__restrict__ ct,
									 dtype *__restrict__ dc,
									 dtype *__restrict__ g,
									 dtype *__restrict__ prev_c,
									 dtype *__restrict__ prev_dc,
									 dtype *__restrict__ h,
									 dtype *__restrict__ max_o,
									 size_t N, size_t L, size_t B, int stream_idx ) {
									 
									 
	size_t num_blocks = ( N * B + NUM_THREADS - 1 ) / NUM_THREADS;
	kernel_elementwise_alstm_backward <<< num_blocks, NUM_THREADS, stream_idx>>> ( dg, dh, c, ct, dc, g, prev_c, prev_dc, h,
			max_o,  N, L, B );
			
}

void cu_elementwise_dolstm_forward ( dtype *__restrict__ g, dtype *__restrict__ g2, dtype *__restrict__ b,
									 dtype *__restrict__ h,
									 dtype *__restrict__ max_o,
									 dtype *__restrict__ c,
									 dtype *__restrict__ ct,
									 dtype *__restrict__ prev_c,
									 dtype *__restrict__ rands,
									 size_t N, size_t L, size_t B, int stream_idx ) {
									 
	size_t num_blocks = ( N * B + NUM_THREADS - 1 ) / NUM_THREADS;
	kernel_elementwise_dolstm_forward <<< num_blocks, NUM_THREADS, stream_idx>>> ( g, g2, b, h, max_o, c, ct, prev_c, rands,
			N, L, B );
			
}

void cu_elementwise_dolstm_backward ( dtype *__restrict__ dg,
									  dtype *__restrict__ dh,
									  dtype *__restrict__ c,
									  dtype *__restrict__ ct,
									  dtype *__restrict__ dc,
									  dtype *__restrict__ g,
									  dtype *__restrict__ prev_c,
									  dtype *__restrict__ prev_dc,
									  dtype *__restrict__ h,
									  dtype *__restrict__ max_o,
									  size_t N, size_t L, size_t B, int stream_idx ) {
									  
									  
	size_t num_blocks = ( N * B + NUM_THREADS - 1 ) / NUM_THREADS;
	kernel_elementwise_dolstm_backward <<< num_blocks, NUM_THREADS, stream_idx>>> ( dg, dh, c, ct, dc, g, prev_c, prev_dc,
			h,
			max_o,  N, L, B );
			
}

/************************/

#define i_gates 0 * N * B * L
#define o_gates 1 * N * B * L
#define f_gates 2 * N * B * L
#define c_gates 3 * N * B * L

__global__ void kernel_elementwise_dolstm_forward ( dtype *__restrict__ g,
		dtype *__restrict__ g2,
		dtype *__restrict__ b,
		dtype *__restrict__ h,
		dtype *__restrict__ max_o,
		dtype *__restrict__ c,
		dtype *__restrict__ ct,
		dtype *__restrict__ prev_c,
		dtype *__restrict__ rands,
		size_t N, size_t L, size_t B ) {
		
	size_t elements = N * B;
	
	/* there are N * B threads */
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	
	/* in - gates after SGEMMs */
	
	if ( tid < elements ) {
	
		h[tid] = 0;
		
		for ( int l = 0; l < L; l++ ) {
		
			int ltid = tid + l * N * B;
			
			c[ltid] = prev_c[ltid];
		}
		
		for ( int l = 0; l < L / 2; l++ ) {
		
			int ltid = tid + l * N * B;
			
			//max_o[ltid] = tid;
			if ( rands[ltid] <= ( dtype ) 0.5 ) {
			
				max_o[ltid] = ( dtype ) ( tid + 2 * l * N * B );
				
				//printf("1 tid = %d l = %d, rand %f, max_o = %d\n", tid, l, rands[tid],  (int)max_o[ltid]);
				
			} else {
			
				max_o[ltid] = ( dtype ) ( tid + ( 2 * l + 1 ) * N * B );
				
				//printf("2 tid %d, l = %d, rand %f, max_o = %d\n", tid, l, rands[tid],  (int)max_o[ltid]);
				
			}
			
			
		}
		
		for ( int l = 0; l < L / 2; l++ ) {
		
			int ltid = ( int ) max_o[tid + l * N * B];
			
			//printf("tid %d l %d ltid %d\n", tid, l, ltid );
			
			/* add bias */
			g[i_gates + ltid] 	+= g2[i_gates + ltid] + b[ ( i_gates + ltid ) / B];
			g[o_gates + ltid] 	+= g2[o_gates + ltid] + b[ ( o_gates + ltid ) / B];
			g[f_gates + ltid] 	+= g2[f_gates + ltid] + b[ ( f_gates + ltid ) / B];
			g[c_gates + ltid] 	+= g2[c_gates + ltid] + b[ ( c_gates + ltid ) / B];
			
			/* there are 4 * N * B gate activation */
			
			g[i_gates + ltid] 	= 	device_logistic	( g[i_gates + ltid] );
			g[o_gates + ltid] 	= 	device_logistic	( g[o_gates + ltid] );
			g[f_gates + ltid] 	= 	device_logistic	( g[f_gates + ltid] );
			g[c_gates + ltid] 	= 	device_tanh	( g[c_gates + ltid] );
			
			c[ltid] 				= 	g[f_gates + ltid] * prev_c[ltid];
			c[ltid] 				+= 	g[i_gates + ltid] * g[c_gates + ltid];
			
			ct[ltid] 				= 	device_tanh ( c[ltid] );
			
			h[tid] += g[o_gates + ltid] * ct[ltid];
			
		}
		
		
		
		h[tid] = device_tanh ( h[tid] );
	}
	
	/* out - updated c and h */
	
}

__global__ void kernel_elementwise_dolstm_backward (
	dtype *__restrict__ dg,
	dtype *__restrict__ dh,
	dtype *__restrict__ c,
	dtype *__restrict__ ct,
	dtype *__restrict__ dc,
	dtype *__restrict__ g,
	dtype *__restrict__ prev_c,
	dtype *__restrict__ prev_dc,
	dtype *__restrict__ h,
	dtype *__restrict__ max_o,
	size_t N, size_t L, size_t B ) {
	
	size_t elements = N * B;
	
	/* there are N * B threads */
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	
	if ( tid < elements ) {
	
		dh[tid] = dh[tid] * device_tanh_prime	( h[tid] );
		
		for ( int l = 0; l < L; l++ ) {
		
			int ltid = tid + l * N * B;
			prev_dc[ltid] = dc[ltid];
			
		}
		
		for ( int l = 0; l < L / 2; l++ ) {
		
			int ltid = ( int ) max_o[tid + l * N * B];
			
			dc[ltid] = dc[ltid] + dh[tid] * g[o_gates + ltid] * device_tanh_prime ( ct[ltid] );
			dg[o_gates + ltid] = dh[tid] * ct[ltid];
			
			//dc[ltid] = dc[ltid] * device_tanh_prime ( c[ltid] );
			
			dg[i_gates + ltid] = dc[ltid] * g[c_gates + ltid];
			dg[f_gates + ltid] = dc[ltid] * prev_c[ltid];
			dg[c_gates + ltid] = dc[ltid] * g[i_gates + ltid];
			
			dg[i_gates + ltid] 	= 	dg[i_gates + ltid] * device_logistic_prime	( g[i_gates + ltid] );
			dg[o_gates + ltid] 	= 	dg[o_gates + ltid] * device_logistic_prime	( g[o_gates + ltid] );
			dg[f_gates + ltid] 	= 	dg[f_gates + ltid] * device_logistic_prime	( g[f_gates + ltid] );
			dg[c_gates + ltid] 	= 	dg[c_gates + ltid] * device_tanh_prime	( g[c_gates + ltid] );
			
			prev_dc[ltid] = dc[ltid] * g[f_gates + ltid];
			
		}
		
	}
	
	
}

#undef i_gates
#undef o_gates
#undef f_gates
#undef c_gates

/****************************************/

#define i_gates 0 * N * B * L
#define o_gates 1 * N * B * L
#define f_gates 2 * N * B * L
#define c_gates 3 * N * B * L
#define s_gates 4 * N * B * L

__global__ void kernel_elementwise_alstm_forward (	dtype *__restrict__ g,
		dtype *__restrict__ g2,
		dtype *__restrict__ b,
		dtype *__restrict__ h,
		dtype *__restrict__ max_o,
		dtype *__restrict__ c,
		dtype *__restrict__ ct,
		dtype *__restrict__ prev_c,
		dtype *__restrict__ rands,
		size_t N, size_t L, size_t B ) {
		
	size_t elements = N * B;
	
	/* there are N * B threads */
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	
	/* in - gates after SGEMMs */
	
	if ( tid < elements ) {
	
		h[tid] = 0;
		
		int max_o_idx = tid;
		//dtype max_o_val = (dtype)-1;
		
		// dtype cumsum[64];
		// dtype total_prob = (dtype)0;
		
		// for (int l = 0; l < L; l++)
		// 	cumsum[l] = 0;
		
		// //selector gates
		// #pragma unroll
		for ( int l = 0; l < L; l++ ) {
		
			int ltid = tid + l * N * B;
			
			// 	g[s_gates + ltid] 	+= g2[s_gates + ltid] + b[(s_gates + ltid) / B];
			// 	g[s_gates + ltid] 	= 	device_logistic	(g[s_gates+ltid]);
			
			// 	total_prob += expf(g[s_gates + ltid]);
			
			// if not selected, just do nothing
			c[ltid] = prev_c[ltid];
		}
		
		// cumsum[0] = expf(g[s_gates + tid])/total_prob;
		
		// for (int l = 1; l < L; l++) {
		
		// 	int ltid = tid + l * N * B;
		
		// 	cumsum[l] = expf(g[s_gates + ltid])/total_prob + cumsum[l-1];
		
		// }
		
		//max_o_idx = tid + 1 * N * B;
		
		for ( int l = 0; l < L; l++ ) {
		
			// 	if (rands[tid] <= cumsum[l]) {
			
			// 		int ltid = tid + l * N * B;
			// 		max_o_idx = ltid;
			// 		printf("l = %d, rand %f, total %f, cumsum %f, max_o = %d\n",
			// 		 		l, rands[tid], total_prob, cumsum[l], max_o_idx);
			// 		break;
			
			// 	}
			if ( rands[tid] <= ( dtype ) ( l + 1 ) / ( dtype ) L ) {
			
				int ltid = tid + l * N * B;
				max_o_idx = ltid;
				// printf("l = %d, rand %f, total %f, cumsum %f, max_o = %d\n",
				//  		l, rands[tid], total_prob, (dtype)(l+1)/(dtype)L, max_o_idx);
				break;
				
			}
		}
		
		max_o[tid] = ( dtype ) max_o_idx;
		
		//for (int l = 0; l < L; l++) {
		
		//int ltid = tid + l * N * B;
		
		/* add bias */
		g[i_gates + max_o_idx] 	+= g2[i_gates + max_o_idx] + b[ ( i_gates + max_o_idx ) / B];
		g[o_gates + max_o_idx] 	+= g2[o_gates + max_o_idx] + b[ ( o_gates + max_o_idx ) / B];
		g[f_gates + max_o_idx] 	+= g2[f_gates + max_o_idx] + b[ ( f_gates + max_o_idx ) / B];
		g[c_gates + max_o_idx] 	+= g2[c_gates + max_o_idx] + b[ ( c_gates + max_o_idx ) / B];
		
		/* there are 4 * N * B gate activation */
		
		g[i_gates + max_o_idx] 	= 	device_logistic	( g[i_gates + max_o_idx] );
		g[o_gates + max_o_idx] 	= 	device_logistic	( g[o_gates + max_o_idx] );
		g[f_gates + max_o_idx] 	= 	device_logistic	( g[f_gates + max_o_idx] );
		g[c_gates + max_o_idx] 	= 	device_tanh	( g[c_gates + max_o_idx] );
		
		c[max_o_idx] 				= 	g[f_gates + max_o_idx] * prev_c[max_o_idx];
		c[max_o_idx] 				+= 	g[i_gates + max_o_idx] * g[c_gates + max_o_idx];
		
		ct[max_o_idx] 				= 	device_tanh ( c[max_o_idx] );
		
		h[tid] = g[o_gates + max_o_idx] * ct[max_o_idx];
		
		//h[tid] 				+= 	g[o_gates + ltid] * c[ltid];
		// if (max_o_val < g[o_gates + ltid]) {
		
		// 	max_o_val = g[o_gates + ltid];
		// 	max_o_idx = ltid;
		
		// }
		// }
		
		
		
		//h[tid] = device_tanh(h[tid]);
	}
	
	/* out - updated c and h */
	
}

__global__ void kernel_elementwise_alstm_backward (
	dtype *__restrict__ dg,
	dtype *__restrict__ dh,
	dtype *__restrict__ c,
	dtype *__restrict__ ct,
	dtype *__restrict__ dc,
	dtype *__restrict__ g,
	dtype *__restrict__ prev_c,
	dtype *__restrict__ prev_dc,
	dtype *__restrict__ h,
	dtype *__restrict__ max_o,
	size_t N, size_t L, size_t B ) {
	
	size_t elements = N * B;
	
	/* there are N * B threads */
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	
	if ( tid < elements ) {
	
		//dh[tid] = dh[tid] * device_tanh_prime		(h[tid]);
		//#pragma unroll
		//for (int l = 0; l < L; l++) {
		
		int ltid = ( int ) max_o[tid];
		
		dc[ltid] = dc[ltid] + dh[tid] * g[o_gates + ltid] * device_tanh_prime ( ct[ltid] );
		dg[o_gates + ltid] = dh[tid] * ct[ltid];
		
		//dc[ltid] = dc[ltid] * device_tanh_prime ( c[ltid] );
		
		dg[i_gates + ltid] = dc[ltid] * g[c_gates + ltid];
		dg[f_gates + ltid] = dc[ltid] * prev_c[ltid];
		dg[c_gates + ltid] = dc[ltid] * g[i_gates + ltid];
		
		dg[i_gates + ltid] 	= 	dg[i_gates + ltid] * device_logistic_prime	( g[i_gates + ltid] );
		dg[o_gates + ltid] 	= 	dg[o_gates + ltid] * device_logistic_prime	( g[o_gates + ltid] );
		dg[f_gates + ltid] 	= 	dg[f_gates + ltid] * device_logistic_prime	( g[f_gates + ltid] );
		dg[c_gates + ltid] 	= 	dg[c_gates + ltid] * device_tanh_prime	( g[c_gates + ltid] );
		
		for ( int l = 0; l < L; l++ ) {
		
			int ltid2 = tid + l * N * B;
			
			//carry - c state
			if ( ltid2 != ltid )
				prev_dc[ltid2] = dc[ltid2];
			else
				prev_dc[ltid2] = dc[ltid2] * g[f_gates + ltid2];
				
		}
	}
	
	
}

#undef i_gates
#undef o_gates
#undef f_gates
#undef c_gates
#undef s_gates

/****************************************/

void cu_elementwise_aclstm_forward ( dtype *__restrict__ g, dtype *__restrict__ g2, dtype *__restrict__ b,
									 dtype *__restrict__ h,
									 dtype *__restrict__ max_o,
									 dtype *__restrict__ c,
									 dtype *__restrict__ prev_c,
									 dtype *__restrict__ rands,
									 size_t N, size_t L, size_t B, int stream_idx ) {
									 
	size_t num_blocks = ( N * B + NUM_THREADS - 1 ) / NUM_THREADS;
	kernel_elementwise_aclstm_forward <<< num_blocks, NUM_THREADS, stream_idx>>> ( g, g2, b, h, max_o, c, prev_c, rands, N,
			L, B );
			
}

void cu_elementwise_aclstm_backward ( dtype *__restrict__ dg,
									  dtype *__restrict__ dh,
									  dtype *__restrict__ c,
									  dtype *__restrict__ dc,
									  dtype *__restrict__ g,
									  dtype *__restrict__ prev_c,
									  dtype *__restrict__ prev_dc,
									  dtype *__restrict__ h,
									  dtype *__restrict__ max_o,
									  size_t N, size_t L, size_t B, int stream_idx ) {
									  
									  
	size_t num_blocks = ( N * B + NUM_THREADS - 1 ) / NUM_THREADS;
	kernel_elementwise_aclstm_backward <<< num_blocks, NUM_THREADS, stream_idx>>> ( dg, dh, c, dc, g, prev_c, prev_dc, h,
			max_o,  N, L, B );
			
}

#define i_gates 0 * N * B
#define o_gates 1 * N * B
#define f_gates 2 * N * B
#define c_gates 3 * N * B
#define s_gates 4 * N * B

__global__ void kernel_elementwise_aclstm_forward ( dtype *__restrict__ g,
		dtype *__restrict__ g2,
		dtype *__restrict__ b,
		dtype *__restrict__ h,
		dtype *__restrict__ max_o,
		dtype *__restrict__ c,
		dtype *__restrict__ prev_c,
		dtype *__restrict__ rands,
		size_t N, size_t L, size_t B ) {
		
	size_t elements = N * B;
	
	/* there are N * B threads */
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	
	/* in - gates after SGEMMs */
	
	if ( tid < elements ) {
	
		h[tid] = 0;
		
		//dtype total_prob = (dtype)0;
		
		// // #pragma unroll
		for ( int l = 0; l < L; l++ ) {
		
			int ltid = tid + l * N * B;
			
			//selector gates
			g[s_gates + ltid] 	+= g2[s_gates + ltid] + b[ ( s_gates + ltid ) / B];
			g[s_gates + ltid] 	=  device_logistic	( g[s_gates + ltid] );
			
			// total_prob += expf(g[s_gates + ltid]);
			
		}
		
		
		/*		for (int l = 0; l < L; l++) {
		
						int ltid = tid + l * N * B;
		
						//max_o_idx = ltid;
						// g[s_gates + ltid] = expf(g[s_gates + ltid])/total_prob;
		
						// printf("l = %d, rand %f, total %f, cumsum %f, max_o = %d\n",
						//  		l, rands[tid], total_prob, cumsum[l], max_o_idx);
						//break;
		
						// g[s_gates + ltid] = (dtype)0.25 * l;
						g[s_gates + ltid] 	= 	device_logistic	(g[s_gates + ltid]);
		
				}*/
		
		/* add bias */
		g[i_gates + tid] 	+= g2[i_gates + tid] + b[ ( i_gates + tid ) / B];
		g[o_gates + tid] 	+= g2[o_gates + tid] + b[ ( o_gates + tid ) / B];
		g[f_gates + tid] 	+= g2[f_gates + tid] + b[ ( f_gates + tid ) / B];
		g[c_gates + tid] 	+= g2[c_gates + tid] + b[ ( c_gates + tid ) / B];
		
		/* there are 4 * N * B gate activation */
		g[i_gates + tid] 	= 	device_logistic	( g[i_gates + tid] );
		g[o_gates + tid] 	= 	device_logistic	( g[o_gates + tid] );
		g[f_gates + tid] 	= 	device_logistic	( g[f_gates + tid] );
		g[c_gates + tid] 	= 	device_tanh	( g[c_gates + tid] );
		
		//soft selection
		for ( int l = 0; l < L; l++ ) {
		
			//if (rands[tid] <= cumsum[l]) {
			
			int ltid = tid + l * N * B;
			
			c[ltid] 				= 	g[s_gates + ltid] * g[f_gates + tid] * prev_c[ltid];
			c[ltid] 				+= 	g[s_gates + ltid] * g[i_gates + tid] * g[c_gates + tid];
			c[ltid] 				= 	device_tanh ( c[ltid] );
			h[tid] 					+= 	g[o_gates + tid] * c[ltid];
			
			/*
							c[ltid] = ((dtype)1 - g[s_gates + ltid]) * prev_c[ltid];
							c[ltid] += (g[s_gates + ltid]) * (g[f_gates + tid] * prev_c[ltid]);
							c[ltid] += (g[s_gates + ltid]) * (g[i_gates + tid] * g[c_gates + tid]);
							c[ltid] = 	device_tanh(c[ltid]);
							h[tid] += (g[s_gates + ltid]) * g[o_gates + tid] * c[ltid];*/
			
			
		}
		
		h[tid] = device_tanh ( h[tid] );
	}
	
	/* out - updated c and h */
	
}

__global__ void kernel_elementwise_aclstm_backward (
	dtype *__restrict__ dg,
	dtype *__restrict__ dh,
	dtype *__restrict__ c,
	dtype *__restrict__ dc,
	dtype *__restrict__ g,
	dtype *__restrict__ prev_c,
	dtype *__restrict__ prev_dc,
	dtype *__restrict__ h,
	dtype *__restrict__ max_o,
	size_t N, size_t L, size_t B ) {
	
	size_t elements = N * B;
	
	/* there are N * B threads */
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	
	if ( tid < elements ) {
	
		dh[tid] = dh[tid] * device_tanh_prime	( h[tid] );
		
#pragma unroll
		
		for ( int l = 0; l < L; l++ ) {
		
			/*			int ltid = tid + l * N * B;
						dc[ltid] = dc[ltid] + dh[tid] * g[o_gates + ltid] * g[s_gates + ltid];
						dc[ltid] = dc[ltid] * device_tanh_prime ( c[ltid] );
			
						dg[o_gates + tid] += dh[tid] * c[ltid] * g[s_gates + ltid];
						dg[i_gates + tid] += dc[ltid] * g[c_gates + tid] * g[s_gates + ltid];
						dg[f_gates + tid] += dc[ltid] * prev_c[ltid] * g[s_gates + ltid];
						dg[c_gates + tid] += dc[ltid] * g[i_gates + tid] * g[s_gates + ltid];
			
						dg[s_gates + ltid] = (g[o_gates + tid] * c[ltid]) * dh[tid];
						dg[s_gates + ltid] += (g[i_gates + tid] * g[c_gates + tid]) * dc[ltid];
						dg[s_gates + ltid] += (g[f_gates + tid] * prev_c[ltid]);
						dg[s_gates + ltid] += prev_c[ltid] * dc[ltid];
			
						//softmax derivative
						dg[s_gates + ltid] = dg[s_gates + ltid] - g[s_gates + ltid];
			*/
			int ltid = tid + l * N * B;
			dc[ltid] = dc[ltid] + dh[tid] * g[o_gates + tid];
			dc[ltid] = dc[ltid] * device_tanh_prime ( c[ltid] );
			
			//propagate error back through gates
			dg[o_gates + tid] += dh[tid] * c[ltid];
			dg[i_gates + tid] += dc[ltid] * g[c_gates + tid] * g[s_gates + ltid];
			dg[f_gates + tid] += dc[ltid] * prev_c[ltid] * g[s_gates + ltid];
			dg[c_gates + tid] += dc[ltid] * g[i_gates + tid] * g[s_gates + ltid];
			
		}
		
		dg[i_gates + tid] 	= 	dg[i_gates + tid] * device_logistic_prime	( g[i_gates + tid] );
		dg[o_gates + tid] 	= 	dg[o_gates + tid] * device_logistic_prime	( g[o_gates + tid] );
		dg[f_gates + tid] 	= 	dg[f_gates + tid] * device_logistic_prime	( g[f_gates + tid] );
		dg[c_gates + tid] 	= 	dg[c_gates + tid] * device_tanh_prime	( g[c_gates + tid] );
		
		for ( int l = 0; l < L; l++ ) {
		
			int ltid = tid + l * N * B;
			
			dg[s_gates + ltid]  = dc[ltid] * ( g[f_gates + tid] * prev_c[ltid] +
											   g[i_gates + tid] * g[c_gates + tid] );
											   
			dg[s_gates + ltid] 	= 	dg[s_gates + ltid] * device_logistic_prime	( g[s_gates + ltid] );
			/*				prev_dc[ltid] = dc[ltid] * g[f_gates + tid] * g[s_gates + ltid];
							prev_dc[ltid] += dc[ltid] * ((dtype)1 - g[s_gates + ltid]);
			*/
			prev_dc[ltid] = dc[ltid] * g[f_gates + tid] * g[s_gates + ltid];
			
		}
	}
	
	
}

#undef i_gates
#undef o_gates
#undef f_gates
#undef c_gates

/****************************************/

void cu_elementwise_aslstm_forward ( dtype *__restrict__ g, dtype *__restrict__ g2, dtype *__restrict__ b,
									 dtype *__restrict__ h,
									 dtype *__restrict__ max_o,
									 dtype *__restrict__ c,
									 dtype *__restrict__ prev_c,
									 dtype *__restrict__ rands,
									 size_t N, size_t L, size_t B, int stream_idx ) {
									 
	size_t num_blocks = ( N * B + NUM_THREADS - 1 ) / NUM_THREADS;
	kernel_elementwise_aslstm_forward <<< num_blocks, NUM_THREADS, stream_idx>>> ( g, g2, b, h, max_o, c, prev_c, rands, N,
			L, B );
			
}

void cu_elementwise_aslstm_backward ( dtype *__restrict__ dg,
									  dtype *__restrict__ dh,
									  dtype *__restrict__ c,
									  dtype *__restrict__ dc,
									  dtype *__restrict__ g,
									  dtype *__restrict__ prev_c,
									  dtype *__restrict__ prev_dc,
									  dtype *__restrict__ h,
									  dtype *__restrict__ max_o,
									  size_t N, size_t L, size_t B, int stream_idx ) {
									  
									  
	size_t num_blocks = ( N * B + NUM_THREADS - 1 ) / NUM_THREADS;
	kernel_elementwise_aslstm_backward <<< num_blocks, NUM_THREADS, stream_idx>>> ( dg, dh, c, dc, g, prev_c, prev_dc, h,
			max_o,  N, L, B );
			
}

#define i_gates 0 * N * B
#define o_gates 1 * N * B
#define f_gates 2 * N * B
#define c_gates 3 * N * B

__global__ void kernel_elementwise_aslstm_forward ( dtype *__restrict__ g,
		dtype *__restrict__ g2,
		dtype *__restrict__ b,
		dtype *__restrict__ h,
		dtype *__restrict__ max_o,
		dtype *__restrict__ c,
		dtype *__restrict__ prev_c,
		dtype *__restrict__ rands,
		size_t N, size_t L, size_t B ) {
		
	size_t elements = N * B;
	
	/* there are N * B threads */
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	
	/* in - gates after SGEMMs */
	
	if ( tid < elements ) {
	
		h[tid] = 0;
		
		int max_o_idx = tid;
		//dtype max_o_val = (dtype)-1;
		
		// dtype cumsum[64];
		// dtype total_prob = (dtype)0;
		
		// for (int l = 0; l < L; l++)
		// 	cumsum[l] = 0;
		
		// //selector gates
		// #pragma unroll
		for ( int l = 0; l < L; l++ ) {
		
			int ltid = tid + l * N * B;
			
			// 	g[s_gates + ltid] 	+= g2[s_gates + ltid] + b[(s_gates + ltid) / B];
			// 	g[s_gates + ltid] 	= 	device_logistic	(g[s_gates+ltid]);
			
			// 	total_prob += expf(g[s_gates + ltid]);
			
			// if not selected, just do nothing
			c[ltid] = prev_c[ltid];
		}
		
		// cumsum[0] = expf(g[s_gates + tid])/total_prob;
		
		// for (int l = 1; l < L; l++) {
		
		// 	int ltid = tid + l * N * B;
		
		// 	cumsum[l] = expf(g[s_gates + ltid])/total_prob + cumsum[l-1];
		
		// }
		
		//max_o_idx = tid + 1 * N * B;
		
		for ( int l = 0; l < L; l++ ) {
		
			// 	if (rands[tid] <= cumsum[l]) {
			
			// 		int ltid = tid + l * N * B;
			// 		max_o_idx = ltid;
			// 		printf("l = %d, rand %f, total %f, cumsum %f, max_o = %d\n",
			// 		 		l, rands[tid], total_prob, cumsum[l], max_o_idx);
			// 		break;
			
			// 	}
			if ( rands[tid] <= ( dtype ) ( l + 1 ) / ( dtype ) L ) {
			
				int ltid = tid + l * N * B;
				max_o_idx = ltid;
				// printf("l = %d, rand %f, total %f, cumsum %f, max_o = %d\n",
				//  		l, rands[tid], total_prob, (dtype)(l+1)/(dtype)L, max_o_idx);
				break;
				
			}
		}
		
		max_o[tid] = ( dtype ) max_o_idx;
		
		//for (int l = 0; l < L; l++) {
		
		//int ltid = tid + l * N * B;
		
		/* add bias */
		g[i_gates + tid] 	+= g2[i_gates + tid] + b[ ( i_gates + tid ) / B];
		g[o_gates + tid] 	+= g2[o_gates + tid] + b[ ( o_gates + tid ) / B];
		g[f_gates + tid] 	+= g2[f_gates + tid] + b[ ( f_gates + tid ) / B];
		g[c_gates + tid] 	+= g2[c_gates + tid] + b[ ( c_gates + tid ) / B];
		
		/* there are 4 * N * B gate activation */
		
		g[i_gates + tid] 	= 	device_logistic	( g[i_gates + tid] );
		g[o_gates + tid] 	= 	device_logistic	( g[o_gates + tid] );
		g[f_gates + tid] 	= 	device_logistic	( g[f_gates + tid] );
		g[c_gates + tid] 	= 	device_tanh	( g[c_gates + tid] );
		
		c[max_o_idx] 				= 	g[f_gates + tid] * prev_c[max_o_idx];
		c[max_o_idx] 				+= 	g[i_gates + tid] * g[c_gates + tid];
		
		c[max_o_idx] 				= 	device_tanh ( c[max_o_idx] );
		
		h[tid] = g[o_gates + tid] * c[max_o_idx];
		
		//h[tid] 				+= 	g[o_gates + ltid] * c[ltid];
		// if (max_o_val < g[o_gates + ltid]) {
		
		// 	max_o_val = g[o_gates + ltid];
		// 	max_o_idx = ltid;
		
		// }
		// }
		
		
		
		//h[tid] = device_tanh(h[tid]);
	}
	
	/* out - updated c and h */
	
}

__global__ void kernel_elementwise_aslstm_backward (
	dtype *__restrict__ dg,
	dtype *__restrict__ dh,
	dtype *__restrict__ c,
	dtype *__restrict__ dc,
	dtype *__restrict__ g,
	dtype *__restrict__ prev_c,
	dtype *__restrict__ prev_dc,
	dtype *__restrict__ h,
	dtype *__restrict__ max_o,
	size_t N, size_t L, size_t B ) {
	
	size_t elements = N * B;
	
	/* there are N * B threads */
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	
	if ( tid < elements ) {
	
		//dh[tid] = dh[tid] * device_tanh_prime		(h[tid]);
		//#pragma unroll
		//for (int l = 0; l < L; l++) {
		
		int ltid = ( int ) max_o[tid];
		
		dc[ltid] = dc[ltid] + dh[tid] * g[o_gates + tid];
		dg[o_gates + tid] = dh[tid] * c[ltid];
		
		dc[ltid] = dc[ltid] * device_tanh_prime ( c[ltid] );
		
		dg[i_gates + tid] = dc[ltid] * g[c_gates + tid];
		dg[f_gates + tid] = dc[ltid] * prev_c[ltid];
		dg[c_gates + tid] = dc[ltid] * g[i_gates + tid];
		
		dg[i_gates + tid] 	= 	dg[i_gates + tid] * device_logistic_prime	( g[i_gates + tid] );
		dg[o_gates + tid] 	= 	dg[o_gates + tid] * device_logistic_prime	( g[o_gates + tid] );
		dg[f_gates + tid] 	= 	dg[f_gates + tid] * device_logistic_prime	( g[f_gates + tid] );
		dg[c_gates + tid] 	= 	dg[c_gates + tid] * device_tanh_prime	( g[c_gates + tid] );
		
		for ( int l = 0; l < L; l++ ) {
		
			int ltid2 = tid + l * N * B;
			
			//carry - c state
			if ( ltid2 != ltid )
				prev_dc[ltid2] = dc[ltid2];
			else
				prev_dc[ltid2] = dc[ltid2] * g[f_gates + tid];
				
		}
	}
	
	
}

#undef i_gates
#undef o_gates
#undef f_gates
#undef c_gates

/****************************************/


void cu_elementwise_splstm_forward ( dtype *__restrict__ g, dtype *__restrict__ g2, dtype *__restrict__ b,
									 dtype *__restrict__ h,
									 dtype *__restrict__ max_o,
									 dtype *__restrict__ c,
									 dtype *__restrict__ ct,
									 dtype *__restrict__ prev_c,
									 dtype *__restrict__ rands,
									 size_t N, size_t L, size_t B, int stream_idx ) {
									 
	size_t num_blocks = ( N * B + NUM_THREADS - 1 ) / NUM_THREADS;
	kernel_elementwise_splstm_forward <<< num_blocks, NUM_THREADS, stream_idx>>> ( g, g2, b, h, max_o, c, ct, prev_c, rands,
			N, L, B );
			
}

void cu_elementwise_splstm_backward ( dtype *__restrict__ dg,
									  dtype *__restrict__ dh,
									  dtype *__restrict__ c,
									  dtype *__restrict__ ct,
									  dtype *__restrict__ dc,
									  dtype *__restrict__ g,
									  dtype *__restrict__ prev_c,
									  dtype *__restrict__ prev_dc,
									  dtype *__restrict__ h,
									  dtype *__restrict__ max_o,
									  size_t N, size_t L, size_t B, int stream_idx ) {
									  
									  
	size_t num_blocks = ( N * B + NUM_THREADS - 1 ) / NUM_THREADS;
	kernel_elementwise_splstm_backward <<< num_blocks, NUM_THREADS, stream_idx>>> ( dg, dh, c, ct, dc, g, prev_c, prev_dc,
			h,
			max_o,  N, L, B );
			
}

#define i_gates 0 * N * B * L
#define o_gates 1 * N * B * L
#define f_gates 2 * N * B * L
#define c_gates 3 * N * B * L

__global__ void kernel_elementwise_splstm_forward (	dtype *__restrict__ g,
		dtype *__restrict__ g2,
		dtype *__restrict__ b,
		dtype *__restrict__ h,
		dtype *__restrict__ max_o,
		dtype *__restrict__ c,
		dtype *__restrict__ ct,
		dtype *__restrict__ prev_c,
		dtype *__restrict__ rands,
		size_t N, size_t L, size_t B ) {
		
	size_t elements = N * B;
	
	/* there are N * B threads */
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	
	/* in - gates after SGEMMs */
	
	if ( tid < elements ) {
	
		h[tid] = 0;
		
		int max_o_idx = tid;
		//dtype max_o_val = (dtype)-1;
		
		dtype cumsum[64];
		dtype total_prob = ( dtype ) 0;
		
		for ( int l = 0; l < L; l++ )
			cumsum[l] = 0;
			
#pragma unroll
			
		for ( int l = 0; l < L; l++ ) {
		
			int ltid = tid + l * N * B;
			
			/* add bias */
			g[i_gates + ltid] 	+= g2[i_gates + ltid] + b[ ( i_gates + ltid ) / B];
			g[o_gates + ltid] 	+= g2[o_gates + ltid] + b[ ( o_gates + ltid ) / B];
			g[f_gates + ltid] 	+= g2[f_gates + ltid] + b[ ( f_gates + ltid ) / B];
			g[c_gates + ltid] 	+= g2[c_gates + ltid] + b[ ( c_gates + ltid ) / B];
			
			/* there are 4 * N * B gate activation */
			
			g[i_gates + ltid] 	= 	device_logistic	( g[i_gates + ltid] );
			g[o_gates + ltid] 	= 	device_logistic	( g[o_gates + ltid] );
			g[f_gates + ltid] 	= 	device_logistic	( g[f_gates + ltid] );
			g[c_gates + ltid] 	= 	device_tanh	( g[c_gates + ltid] );
			
			c[ltid] 				= 	g[f_gates + ltid] * prev_c[ltid];
			c[ltid] 				+= 	g[i_gates + ltid] * g[c_gates + ltid];
			
			ct[ltid] 				= 	device_tanh ( c[ltid] );
			
			//h[tid] 				+= 	g[o_gates + ltid] * c[ltid];
			
			total_prob += expf ( g[o_gates + ltid] );
			
			// if (max_o_val < g[o_gates + ltid]) {
			
			// 	max_o_val = g[o_gates + ltid];
			// 	max_o_idx = ltid;
			
			// }
		}
		
		cumsum[0] = expf ( g[o_gates + tid] ) / total_prob;
		
		for ( int l = 1; l < L; l++ ) {
		
			int ltid = tid + l * N * B;
			
			cumsum[l] = expf ( g[o_gates + ltid] ) / total_prob + cumsum[l - 1];
			
		}
		
		//max_o_idx = tid;
		
		for ( int l = 0; l < L; l++ ) {
			int ltid = tid + l * N * B;
			//printf("l = %d, rand %f, total %f, cumsum %f\n", l, rands[tid], total_prob, cumsum[l]);
			
			if ( rands[tid] <= cumsum[l] ) {
			
				//max_o_val = g[o_gates + ltid];
				max_o_idx = ltid;
				//printf("%d\n", l);
				break;
				
			}
		}
		
		//printf("max_o_idx = %d\n", max_o_idx);
		
		max_o[tid] = ( dtype ) max_o_idx;
		h[tid] = g[o_gates + max_o_idx] * ct[max_o_idx];
		//h[tid] = device_tanh(h[tid]);
	}
	
	/* out - updated c and h */
	
}

__global__ void kernel_elementwise_splstm_backward (
	dtype *__restrict__ dg,
	dtype *__restrict__ dh,
	dtype *__restrict__ c,
	dtype *__restrict__ ct,
	dtype *__restrict__ dc,
	dtype *__restrict__ g,
	dtype *__restrict__ prev_c,
	dtype *__restrict__ prev_dc,
	dtype *__restrict__ h,
	dtype *__restrict__ max_o,
	size_t N, size_t L, size_t B ) {
	
	size_t elements = N * B;
	
	/* there are N * B threads */
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	
	if ( tid < elements ) {
	
		//dh[tid] = dh[tid] * device_tanh_prime		(h[tid]);
		//#pragma unroll
		//for (int l = 0; l < L; l++) {
		
		int ltid = ( int ) max_o[tid];
		dc[ltid] = dc[ltid] + dh[tid] * g[o_gates + ltid] * device_tanh_prime ( ct[ltid] );
		dg[o_gates + ltid] = dh[tid] * ct[ltid];
		
		for ( int l = 0; l < L; l++ ) {
		
			int ltid = tid + l * N * B;
			prev_dc[ltid] += dc[ltid] * g[f_gates + ltid];
			//dc[ltid] = dc[ltid] * device_tanh_prime ( c[ltid] );
			
			//propagate error back through gates
			
			dg[i_gates + ltid] = dc[ltid] * g[c_gates + ltid];
			dg[f_gates + ltid] = dc[ltid] * prev_c[ltid];
			dg[c_gates + ltid] = dc[ltid] * g[i_gates + ltid];
			
			//propagate error back through gates, nonlinearities
			dg[i_gates + ltid] 	= 	dg[i_gates + ltid] * device_logistic_prime	( g[i_gates + ltid] );
			dg[o_gates + ltid] 	= 	dg[o_gates + ltid] * device_logistic_prime	( g[o_gates + ltid] );
			dg[f_gates + ltid] 	= 	dg[f_gates + ltid] * device_logistic_prime	( g[f_gates + ltid] );
			dg[c_gates + ltid] 	= 	dg[c_gates + ltid] * device_tanh_prime	( g[c_gates + ltid] );
			
			//carry - c state
			
			
		}
	}
	
	
}

#undef i_gates
#undef o_gates
#undef f_gates
#undef c_gates

/******************************************/
void cu_elementwise_adagrad (	dtype learning_rate,
								dtype *__restrict__ p,
								dtype *__restrict__ d,
								dtype *__restrict__ m,
								size_t N, int stream_idx ) {
								
								
	size_t num_blocks = ( N + NUM_THREADS - 1 ) / NUM_THREADS;
	kernel_elementwise_adagrad <<< num_blocks, NUM_THREADS, stream_idx>>> ( learning_rate, p, d, m, N );
	
}

void cu_elementwise_adadelta (	dtype learning_rate, dtype rho,
								dtype *__restrict__ p,
								dtype *__restrict__ d,
								dtype *__restrict__ m,
								dtype *__restrict__ u,
								size_t N, int stream_idx ) {
								
								
	size_t num_blocks = ( N + NUM_THREADS - 1 ) / NUM_THREADS;
	kernel_elementwise_adadelta <<< num_blocks, NUM_THREADS, stream_idx>>> ( learning_rate, rho, p, d, m, u, N );
	
}

void cu_elementwise_adadelta_decay (	dtype learning_rate, dtype rho,
										dtype *__restrict__ p,
										dtype *__restrict__ d,
										dtype *__restrict__ m,
										dtype *__restrict__ u,
										size_t N, dtype decay, int stream_idx ) {
										
										
	size_t num_blocks = ( N + NUM_THREADS - 1 ) / NUM_THREADS;
	kernel_elementwise_adadelta_decay <<< num_blocks, NUM_THREADS, stream_idx>>> ( learning_rate, rho, p, d, m, u, N,
			decay );
			
}

__global__ void kernel_elementwise_adagrad (
	dtype learning_rate,
	dtype *__restrict__ p,
	dtype *__restrict__ d,
	dtype *__restrict__ m,
	size_t N ) {
	
	
	size_t elements = N;
	
	/* there are N * B threads */
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	
	if ( tid < elements ) {
	
		m[tid] += d[tid] * d[tid];
		p[tid] -= learning_rate * d[tid] / device_sqrt_eps ( m[tid], ( dtype ) 1e-6 );
		
		
	}
	
}

__global__ void kernel_elementwise_adadelta (
	dtype learning_rate, dtype rho,
	dtype *__restrict__ p,
	dtype *__restrict__ d,
	dtype *__restrict__ m,
	dtype *__restrict__ u,
	size_t N ) {
	
	size_t elements = N;
	
	/* there are N * B threads */
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	
	if ( tid < elements ) {
	
		assert ( !isinf ( d[tid] ) );
		assert ( !isnan ( d[tid] ) );
		
		//clip
		d[tid] = fminf ( d[tid], 1.0f );
		d[tid] = fmaxf ( d[tid], -1.0f );
		
		m[tid] = rho * m[tid] + ( ( dtype ) 1 - rho ) * d[tid] * d[tid];
		
		p[tid] -= learning_rate * d[tid] / device_sqrt_eps ( m[tid], ( dtype ) 1e-4 );
		
		
	}
	
}

__global__ void kernel_elementwise_adadelta_decay (
	dtype learning_rate, dtype rho,
	dtype *__restrict__ p,
	dtype *__restrict__ d,
	dtype *__restrict__ m,
	dtype *__restrict__ u,
	size_t N, dtype decay ) {
	
	size_t elements = N;
	
	/* there are N * B threads */
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	
	if ( tid < elements ) {
	
		assert ( !isinf ( d[tid] ) );
		assert ( !isnan ( d[tid] ) );
		
		m[tid] = rho * m[tid] + ( ( dtype ) 1 - rho ) * d[tid] * d[tid];
		
		p[tid] = ( ( dtype ) 1 - decay ) * p[tid] - learning_rate * d[tid] / device_sqrt_eps ( m[tid], ( dtype ) 1e-4 );
		
		
	}
	
}