/*
*
* Author: Kamil Rocki
*
*/

#ifdef __PRECISE_MATH__
	#define dtype double
#else
	#define dtype float
#endif

#define NUM_THREADS 256

#ifndef __KERNELS_H__
#define __KERNELS_H__

#include <curand.h>
extern curandGenerator_t prng;

void cu_sub (
	dtype *__restrict__ out,
	dtype *__restrict__ data,
	dtype *__restrict__ other,
	size_t elements,
	int stream_idx = 0 );

__global__ void kernel_elementwise_sub (
	dtype *__restrict__ out,
	dtype *__restrict__a,
	dtype *__restrict__b,
	size_t n );

void cu_exp (
	dtype *__restrict__ data,
	size_t elements,
	int stream_idx = 0 );

__global__ void kernel_elementwise_exp (
	dtype *__restrict__c,
	size_t n );

void cu_submax (
	dtype *__restrict__ data,
	size_t elements,
	dtype maxval,
	int stream_idx = 0 );

__global__ void kernel_elementwise_submax (
	dtype *__restrict__c,
	size_t n,
	dtype maxval );

void cu_tanh (
	dtype *__restrict__ data,
	size_t elements,
	int stream_idx = 0 );

__global__ void kernel_elementwise_tanh (
	dtype *__restrict__c,
	size_t n );

void cu_logistic ( dtype *__restrict__ data, size_t elements, int stream_idx = 0 );
__global__ void kernel_elementwise_logistic ( dtype *__restrict__c, size_t n );

void cu_dtanh ( dtype *__restrict__ c, dtype *__restrict__ x, dtype *__restrict__ y, size_t elements );
__global__ void kernel_elementwise_dtanh ( dtype *__restrict__ c, dtype *__restrict__ x, dtype *__restrict__ y,
		size_t n );

void cu_elementwise_mult ( dtype *__restrict__ z, dtype *__restrict__ x, dtype *__restrict__ y, size_t elements,
						   int stream_idx = 0 );
__global__ void kernel_elementwise_mult ( dtype *__restrict__ z, dtype *__restrict__ x, dtype *__restrict__ y,
		size_t n );

void cu_elementwise_mult_add ( dtype *__restrict__ z, dtype *__restrict__ x, dtype *__restrict__ y, size_t elements,
							   int stream_idx = 0 );
__global__ void kernel_elementwise_mult_add ( dtype *__restrict__z, dtype *__restrict__x, dtype *__restrict__ y,
		size_t n );

__global__ void kernel_elementwise_add_row_vector ( dtype *__restrict__ m,
		dtype *__restrict__ v,
		size_t N, size_t B );


__global__ void kernel_elementwise_sub_col_vector (
	dtype *__restrict__ m,
	dtype *__restrict__ v,
	size_t N, size_t B );

void cu_div_col_vector ( dtype *__restrict__ m, dtype *__restrict__ v, size_t N, size_t B, int stream_idx = 0 );
void cu_sub_col_vector ( dtype *__restrict__ m, dtype *__restrict__ v, size_t N, size_t B, int stream_idx = 0 );

void cu_add_row_vector ( dtype *__restrict__ m, dtype *__restrict__ v, size_t N, size_t B, int stream_idx = 0 );

void cu_row_max ( dtype *__restrict__ v,  dtype *__restrict__ m, int N, int B, int stream_idx = 0 );
__global__ void kernel_row_max ( dtype *__restrict__ v,  dtype *__restrict__ m, int N, int B );

__global__ void kernel_elementwise_div_col_vector (
	dtype *__restrict__ m,
	dtype *__restrict__ v,
	size_t N, size_t B );

__global__ void kernel_sub_max ( dtype *__restrict__ m, size_t N );

void cu_elementwise_zeros (
	dtype *__restrict__ data,
	size_t elements,
	int stream_idx = 0 );

__global__ void kernel_elementwise_zeros ( dtype *__restrict__c, size_t n );

void cu_sub_max ( dtype *__restrict__ m, size_t N, int stream_idx = 0 );

/* lstm */
void cu_elementwise_lstm_forward (
	dtype *__restrict__ g,
	dtype *__restrict__ g2,
	dtype *__restrict__ b,
	dtype *__restrict__ h,
	dtype *__restrict__ c,
	dtype *__restrict__ ct,
	dtype *__restrict__ prev_c,
	size_t N, size_t B, int stream_idx = 0 );


__global__ void kernel_elementwise_lstm_forward (
	dtype *__restrict__ gc,
	dtype *__restrict__ g2,
	dtype *__restrict__ b,
	dtype *__restrict__ h,
	dtype *__restrict__ c,
	dtype *__restrict__ ct,
	dtype *__restrict__ prev_c,
	size_t N, size_t B );

void cu_elementwise_lstm_backward (
	dtype *__restrict__ dg,
	dtype *__restrict__ dh,
	dtype *__restrict__ c,
	dtype *__restrict__ ct,
	dtype *__restrict__ dc,
	dtype *__restrict__ g,
	dtype *__restrict__ prev_c,
	dtype *__restrict__ prev_dc,
	size_t N, size_t B, int stream_idx = 0 );

__global__ void kernel_elementwise_lstm_backward (
	dtype *__restrict__ dg,
	dtype *__restrict__ dh,
	dtype *__restrict__ c,
	dtype *__restrict__ ct,
	dtype *__restrict__ dc,
	dtype *__restrict__ g,
	dtype *__restrict__ prev_c,
	dtype *__restrict__ prev_dc,
	size_t N, size_t B );

/* gauss lstm */

void cu_elementwise_gauss_lstm_forward (
	dtype *__restrict__ g,
	dtype *__restrict__ g2,
	dtype *__restrict__ b,
	dtype *__restrict__ h,
	dtype *__restrict__ c,
	dtype *__restrict__ ct,
	dtype *__restrict__ prev_c,
	dtype *__restrict__ rands,
	size_t N, size_t B, int stream_idx = 0 );


__global__ void kernel_elementwise_gauss_lstm_forward (
	dtype *__restrict__ gc,
	dtype *__restrict__ g2,
	dtype *__restrict__ b,
	dtype *__restrict__ h,
	dtype *__restrict__ c,
	dtype *__restrict__ ct,
	dtype *__restrict__ prev_c,
	dtype *__restrict__ rands,
	size_t N, size_t B );

void cu_elementwise_gauss_lstm_backward (
	dtype *__restrict__ dg,
	dtype *__restrict__ dh,
	dtype *__restrict__ c,
	dtype *__restrict__ ct,
	dtype *__restrict__ dc,
	dtype *__restrict__ g,
	dtype *__restrict__ prev_c,
	dtype *__restrict__ prev_dc,
	size_t N, size_t B, int stream_idx = 0 );

__global__ void kernel_elementwise_gauss_lstm_backward (
	dtype *__restrict__ dg,
	dtype *__restrict__ dh,
	dtype *__restrict__ c,
	dtype *__restrict__ ct,
	dtype *__restrict__ dc,
	dtype *__restrict__ g,
	dtype *__restrict__ prev_c,
	dtype *__restrict__ prev_dc,
	size_t N, size_t B );

/* surprisal lstm */

void cu_elementwise_surprisal_lstm_forward (
	dtype *__restrict__ g,
	dtype *__restrict__ g2,
	dtype *__restrict__ b,
	dtype *__restrict__ h,
	dtype *__restrict__ c,
	dtype *__restrict__ ct,
	dtype *__restrict__ prev_c,
	size_t N, size_t B, int stream_idx = 0 );

__global__ void kernel_elementwise_surprisal_lstm_forward (
	dtype *__restrict__ gc,
	dtype *__restrict__ g2,
	dtype *__restrict__ b,
	dtype *__restrict__ h,
	dtype *__restrict__ c,
	dtype *__restrict__ ct,
	dtype *__restrict__ prev_c,
	size_t N, size_t B );

void cu_elementwise_surprisal_lstm_backward (
	dtype *__restrict__ dg,
	dtype *__restrict__ dh,
	dtype *__restrict__ c,
	dtype *__restrict__ ct,
	dtype *__restrict__ dc,
	dtype *__restrict__ g,
	dtype *__restrict__ prev_c,
	dtype *__restrict__ prev_dc,
	size_t N, size_t B, int stream_idx = 0 );

__global__ void kernel_elementwise_surprisal_lstm_backward (
	dtype *__restrict__ dg,
	dtype *__restrict__ dh,
	dtype *__restrict__ c,
	dtype *__restrict__ ct,
	dtype *__restrict__ dc,
	dtype *__restrict__ g,
	dtype *__restrict__ prev_c,
	dtype *__restrict__ prev_dc,
	size_t N, size_t B );

void cu_elementwise_mlstm_forward (
	dtype *__restrict__ g,
	dtype *__restrict__ g2,
	dtype *__restrict__ b,
	dtype *__restrict__ h,
	dtype *__restrict__ c,
	dtype *__restrict__ prev_c,
	size_t N, size_t L, size_t B, int stream_idx = 0 );


__global__ void kernel_elementwise_mlstm_forward (
	dtype *__restrict__ gc,
	dtype *__restrict__ g2,
	dtype *__restrict__ b,
	dtype *__restrict__ h,
	dtype *__restrict__ c,
	dtype *__restrict__ prev_c,
	size_t N, size_t L, size_t B );

void cu_elementwise_mlstm_backward (
	dtype *__restrict__ dg,
	dtype *__restrict__ dh,
	dtype *__restrict__ c,
	dtype *__restrict__ dc,
	dtype *__restrict__ g,
	dtype *__restrict__ prev_c,
	dtype *__restrict__ prev_dc,
	size_t N, size_t L, size_t B, int stream_idx = 0 );

__global__ void kernel_elementwise_mlstm_backward (
	dtype *__restrict__ dg,
	dtype *__restrict__ dh,
	dtype *__restrict__ c,
	dtype *__restrict__ dc,
	dtype *__restrict__ g,
	dtype *__restrict__ prev_c,
	dtype *__restrict__ prev_dc,
	size_t N, size_t L, size_t B );

void cu_elementwise_clstm_forward (
	dtype *__restrict__ g,
	dtype *__restrict__ g2,
	dtype *__restrict__ b,
	dtype *__restrict__ h,
	dtype *__restrict__ c,
	dtype *__restrict__ ct,
	dtype *__restrict__ prev_c,
	size_t N, size_t L, size_t B, int stream_idx = 0 );


__global__ void kernel_elementwise_clstm_forward (
	dtype *__restrict__ gc,
	dtype *__restrict__ g2,
	dtype *__restrict__ b,
	dtype *__restrict__ h,
	dtype *__restrict__ c,
	dtype *__restrict__ ct,
	dtype *__restrict__ prev_c,
	size_t N, size_t L, size_t B );

void cu_elementwise_clstm_backward (
	dtype *__restrict__ dg,
	dtype *__restrict__ dh,
	dtype *__restrict__ c,
	dtype *__restrict__ ct,
	dtype *__restrict__ dc,
	dtype *__restrict__ g,
	dtype *__restrict__ prev_c,
	dtype *__restrict__ prev_dc,
	dtype *__restrict__ h,
	size_t N, size_t L, size_t B, int stream_idx = 0 );

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
	size_t N, size_t L, size_t B );

/*************/


void cu_elementwise_sparselstm_forward (
	dtype *__restrict__ g,
	dtype *__restrict__ g2,
	dtype *__restrict__ b,
	dtype *__restrict__ h,
	dtype *__restrict__ c,
	dtype *__restrict__ ct,
	dtype *__restrict__ prev_c,
	dtype *__restrict__ prev_h,
	size_t N, size_t L, size_t B, int stream_idx = 0 );


__global__ void kernel_elementwise_sparselstm_forward (
	dtype *__restrict__ gc,
	dtype *__restrict__ g2,
	dtype *__restrict__ b,
	dtype *__restrict__ h,
	dtype *__restrict__ c,
	dtype *__restrict__ ct,
	dtype *__restrict__ prev_c,
	dtype *__restrict__ prev_h,
	size_t N, size_t L, size_t B );


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
		size_t N, size_t L, size_t B, int stream_idx = 0 );


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
	size_t N, size_t L, size_t B );


/*************/

void cu_elementwise_cmlstm_forward (
	dtype *__restrict__ g,
	dtype *__restrict__ g2,
	dtype *__restrict__ G,
	dtype *__restrict__ b,
	dtype *__restrict__ h,
	dtype *__restrict__ c,
	dtype *__restrict__ prev_c,
	dtype *__restrict__ prev_h,
	dtype *__restrict__ rands,
	size_t N, size_t L, size_t B, int stream_idx = 0 );


__global__ void kernel_elementwise_cmlstm_forward (
	dtype *__restrict__ gc,
	dtype *__restrict__ g2,
	dtype *__restrict__ G,
	dtype *__restrict__ b,
	dtype *__restrict__ h,
	dtype *__restrict__ c,
	dtype *__restrict__ prev_c,
	dtype *__restrict__ prev_h,
	dtype *__restrict__ rands,
	size_t N, size_t L, size_t B );

void cu_elementwise_cmlstm_backward (
	dtype *__restrict__ dg,
	dtype *__restrict__ dh,
	dtype *__restrict__ c,
	dtype *__restrict__ dc,
	dtype *__restrict__ g,
	dtype *__restrict__ prev_c,
	dtype *__restrict__ prev_dc,
	dtype *__restrict__ prev_h,
	dtype *__restrict__ prev_dh,
	dtype *__restrict__ h,
	size_t N, size_t L, size_t B, int stream_idx = 0 );

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
	size_t N, size_t L, size_t B );

/*************/

void cu_elementwise_attlstm_forward (
	dtype *__restrict__ g,
	dtype *__restrict__ g2,
	dtype *__restrict__ G,
	dtype *__restrict__ b,
	dtype *__restrict__ h,
	dtype *__restrict__ c,
	dtype *__restrict__ ct,
	dtype *__restrict__ prev_c,
	dtype *__restrict__ prev_h,
	size_t N, size_t L, size_t B, int stream_idx = 0 );


__global__ void kernel_elementwise_attlstm_forward (
	dtype *__restrict__ gc,
	dtype *__restrict__ g2,
	dtype *__restrict__ G,
	dtype *__restrict__ b,
	dtype *__restrict__ h,
	dtype *__restrict__ c,
	dtype *__restrict__ ct,
	dtype *__restrict__ prev_c,
	dtype *__restrict__ prev_h,
	size_t N, size_t L, size_t B );

void cu_elementwise_attlstm_backward (
	dtype *__restrict__ dg,
	dtype *__restrict__ dh,
	dtype *__restrict__ c,
	dtype *__restrict__ ct,
	dtype *__restrict__ dc,
	dtype *__restrict__ g,
	dtype *__restrict__ prev_c,
	dtype *__restrict__ prev_dc,
	dtype *__restrict__ prev_h,
	dtype *__restrict__ prev_dh,
	dtype *__restrict__ h,
	size_t N, size_t L, size_t B, int stream_idx = 0 );

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
	size_t N, size_t L, size_t B );

/*************/

/* v1 */

// void cu_elementwise_hardattlstm_forward (	dtype *__restrict__ g,
// 		dtype *__restrict__ g2,
// 		dtype *__restrict__ G,
// 		dtype *__restrict__ b,
// 		dtype *__restrict__ h,
// 		dtype *__restrict__ c,
// 		dtype *__restrict__ ct,
// 		dtype *__restrict__ prev_c,
// 		dtype *__restrict__ prev_h,
// 		dtype *__restrict__ rands,
// 		size_t N, size_t L, size_t B, int stream_idx = 0 );


// __global__ void kernel_elementwise_hardattlstm_forward (
// 	dtype *__restrict__ gc,
// 	dtype *__restrict__ g2,
// 	dtype *__restrict__ G,
// 	dtype *__restrict__ b,
// 	dtype *__restrict__ h,
// 	dtype *__restrict__ c,
// 	dtype *__restrict__ ct,
// 	dtype *__restrict__ prev_c,
// 	dtype *__restrict__ prev_h,
// 	dtype *__restrict__ rands,
// 	size_t N, size_t L, size_t B );

// void cu_elementwise_hardattlstm_backward ( dtype *__restrict__ dg,
// 		dtype *__restrict__ dh,
// 		dtype *__restrict__ c,
// 		dtype *__restrict__ ct,
// 		dtype *__restrict__ dc,
// 		dtype *__restrict__ g,
// 		dtype *__restrict__ prev_c,
// 		dtype *__restrict__ prev_dc,
// 		dtype *__restrict__ prev_h,
// 		dtype *__restrict__ prev_dh,
// 		dtype *__restrict__ h,
// 		size_t N, size_t L, size_t B, int stream_idx = 0 );

// __global__ void kernel_elementwise_hardattlstm_backward (
// 	dtype *__restrict__ dg,
// 	dtype *__restrict__ dh,
// 	dtype *__restrict__ c,
// 	dtype *__restrict__ ct,
// 	dtype *__restrict__ dc,
// 	dtype *__restrict__ g,
// 	dtype *__restrict__ prev_c,
// 	dtype *__restrict__ prev_dc,
// 	dtype *__restrict__ h,
// 	dtype *__restrict__ prev_h,
// 	dtype *__restrict__ prev_dh,
// 	size_t N, size_t L, size_t B );

/* v2 */
void cu_elementwise_hardattlstm_forward (
	dtype *__restrict__ g,
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
	size_t N, size_t L, size_t B, int stream_idx = 0 );


__global__ void kernel_elementwise_hardattlstm_forward (
	dtype *__restrict__ gc,
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
	size_t N, size_t L, size_t B );

void cu_elementwise_hardattlstm_backward (
	dtype *__restrict__ dg,
	dtype *__restrict__ dh,
	dtype *__restrict__ c,
	dtype *__restrict__ ct,
	dtype *__restrict__ dc,
	dtype *__restrict__ g,
	dtype *__restrict__ prev_c,
	dtype *__restrict__ prev_dc,
	dtype *__restrict__ prev_h,
	dtype *__restrict__ prev_dh,
	dtype *__restrict__ h,
	dtype *__restrict__ max_o,
	size_t N, size_t L, size_t B, int stream_idx = 0 );

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
	size_t N, size_t L, size_t B );


void cu_elementwise_hlstm_forward (
	dtype *__restrict__ g,
	dtype *__restrict__ g2,
	dtype *__restrict__ b,
	dtype *__restrict__ h,
	dtype *__restrict__ c,
	dtype *__restrict__ ct,
	dtype *__restrict__ prev_c,
	size_t N, size_t L, size_t B, int stream_idx = 0 );


__global__ void kernel_elementwise_hlstm_forward (
	dtype *__restrict__ gc,
	dtype *__restrict__ g2,
	dtype *__restrict__ b,
	dtype *__restrict__ h,
	dtype *__restrict__ c,
	dtype *__restrict__ ct,
	dtype *__restrict__ prev_c,
	size_t N, size_t L, size_t B );

void cu_elementwise_hlstm_backward (
	dtype *__restrict__ dg,
	dtype *__restrict__ dh,
	dtype *__restrict__ c,
	dtype *__restrict__ ct,
	dtype *__restrict__ dc,
	dtype *__restrict__ g,
	dtype *__restrict__ prev_c,
	dtype *__restrict__ prev_dc,
	dtype *__restrict__ h,
	size_t N, size_t L, size_t B, int stream_idx = 0 );

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
	size_t N, size_t L, size_t B );


void cu_elementwise_hclstm_forward (
	dtype *__restrict__ g,
	dtype *__restrict__ g2,
	dtype *__restrict__ b,
	dtype *__restrict__ h,
	dtype *__restrict__ c,
	dtype *__restrict__ prev_c,
	size_t N, size_t L, size_t B, int stream_idx = 0 );


__global__ void kernel_elementwise_hclstm_forward (
	dtype *__restrict__ gc,
	dtype *__restrict__ g2,
	dtype *__restrict__ b,
	dtype *__restrict__ h,
	dtype *__restrict__ c,
	dtype *__restrict__ prev_c,
	size_t N, size_t L, size_t B );

void cu_elementwise_hclstm_backward (
	dtype *__restrict__ dg,
	dtype *__restrict__ dh,
	dtype *__restrict__ c,
	dtype *__restrict__ dc,
	dtype *__restrict__ g,
	dtype *__restrict__ prev_c,
	dtype *__restrict__ prev_dc,
	dtype *__restrict__ h,
	size_t N, size_t L, size_t B, int stream_idx = 0 );

__global__ void kernel_elementwise_hclstm_backward (
	dtype *__restrict__ dg,
	dtype *__restrict__ dh,
	dtype *__restrict__ c,
	dtype *__restrict__ dc,
	dtype *__restrict__ g,
	dtype *__restrict__ prev_c,
	dtype *__restrict__ prev_dc,
	dtype *__restrict__ h,
	size_t N, size_t L, size_t B );



/******************************/

void cu_elementwise_hmlstm_forward (
	dtype *__restrict__ g,
	dtype *__restrict__ g2,
	dtype *__restrict__ b,
	dtype *__restrict__ h,
	dtype *__restrict__ c,
	dtype *__restrict__ prev_c,
	size_t N, size_t L, size_t B, int stream_idx = 0 );


__global__ void kernel_elementwise_hmlstm_forward (
	dtype *__restrict__ gc,
	dtype *__restrict__ g2,
	dtype *__restrict__ b,
	dtype *__restrict__ h,
	dtype *__restrict__ c,
	dtype *__restrict__ prev_c,
	size_t N, size_t L, size_t B );

void cu_elementwise_hmlstm_backward (
	dtype *__restrict__ dg,
	dtype *__restrict__ dh,
	dtype *__restrict__ c,
	dtype *__restrict__ dc,
	dtype *__restrict__ g,
	dtype *__restrict__ prev_c,
	dtype *__restrict__ prev_dc,
	dtype *__restrict__ h,
	size_t N, size_t L, size_t B, int stream_idx = 0 );

__global__ void kernel_elementwise_hmlstm_backward (
	dtype *__restrict__ dg,
	dtype *__restrict__ dh,
	dtype *__restrict__ c,
	dtype *__restrict__ dc,
	dtype *__restrict__ g,
	dtype *__restrict__ prev_c,
	dtype *__restrict__ prev_dc,
	dtype *__restrict__ h,
	size_t N, size_t L, size_t B );

/******************************/


void cu_elementwise_hhlstm_forward (
	dtype *__restrict__ g,
	dtype *__restrict__ g2,
	dtype *__restrict__ b,
	dtype *__restrict__ h,
	dtype *__restrict__ c,
	dtype *__restrict__ prev_c,
	size_t N, size_t L, size_t B, int stream_idx = 0 );


__global__ void kernel_elementwise_hhlstm_forward (
	dtype *__restrict__ gc,
	dtype *__restrict__ g2,
	dtype *__restrict__ b,
	dtype *__restrict__ h,
	dtype *__restrict__ c,
	dtype *__restrict__ prev_c,
	size_t N, size_t L, size_t B );

void cu_elementwise_hhlstm_backward (
	dtype *__restrict__ dg,
	dtype *__restrict__ dh,
	dtype *__restrict__ c,
	dtype *__restrict__ dc,
	dtype *__restrict__ g,
	dtype *__restrict__ prev_c,
	dtype *__restrict__ prev_dc,
	dtype *__restrict__ h,
	size_t N, size_t L, size_t B, int stream_idx = 0 );

__global__ void kernel_elementwise_hhlstm_backward (
	dtype *__restrict__ dg,
	dtype *__restrict__ dh,
	dtype *__restrict__ c,
	dtype *__restrict__ dc,
	dtype *__restrict__ g,
	dtype *__restrict__ prev_c,
	dtype *__restrict__ prev_dc,
	dtype *__restrict__ h,
	size_t N, size_t L, size_t B );



/******************************/

void cu_elementwise_plstm_forward (
	dtype *__restrict__ g,
	dtype *__restrict__ g2,
	dtype *__restrict__ b,
	dtype *__restrict__ h,
	dtype *__restrict__ max_o,
	dtype *__restrict__ c,
	dtype *__restrict__ prev_c,
	size_t N, size_t L, size_t B, int stream_idx = 0 );


__global__ void kernel_elementwise_plstm_forward (
	dtype *__restrict__ gc,
	dtype *__restrict__ g2,
	dtype *__restrict__ b,
	dtype *__restrict__ h,
	dtype *__restrict__ max_o,
	dtype *__restrict__ c,
	dtype *__restrict__ prev_c,
	size_t N, size_t L, size_t B );

void cu_elementwise_plstm_backward (
	dtype *__restrict__ dg,
	dtype *__restrict__ dh,
	dtype *__restrict__ c,
	dtype *__restrict__ dc,
	dtype *__restrict__ g,
	dtype *__restrict__ prev_c,
	dtype *__restrict__ prev_dc,
	dtype *__restrict__ h,
	dtype *__restrict__ max_o,
	size_t N, size_t L, size_t B, int stream_idx = 0 );

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
	size_t N, size_t L, size_t B );


/******************************/

void cu_elementwise_alstm_forward (
	dtype *__restrict__ g,
	dtype *__restrict__ g2,
	dtype *__restrict__ b,
	dtype *__restrict__ h,
	dtype *__restrict__ max_o,
	dtype *__restrict__ c,
	dtype *__restrict__ ct,
	dtype *__restrict__ prev_c,
	dtype *__restrict__ rands,
	size_t N, size_t L, size_t B, int stream_idx = 0 );


__global__ void kernel_elementwise_alstm_forward (
	dtype *__restrict__ gc,
	dtype *__restrict__ g2,
	dtype *__restrict__ b,
	dtype *__restrict__ h,
	dtype *__restrict__ max_o,
	dtype *__restrict__ c,
	dtype *__restrict__ ct,
	dtype *__restrict__ prev_c,
	dtype *__restrict__ rands,
	size_t N, size_t L, size_t B );

void cu_elementwise_alstm_backward (
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
	size_t N, size_t L, size_t B, int stream_idx = 0 );

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
	size_t N, size_t L, size_t B );

/******************************/

void cu_elementwise_dolstm_forward (
	dtype *__restrict__ g,
	dtype *__restrict__ g2,
	dtype *__restrict__ b,
	dtype *__restrict__ h,
	dtype *__restrict__ max_o,
	dtype *__restrict__ c,
	dtype *__restrict__ ct,
	dtype *__restrict__ prev_c,
	dtype *__restrict__ rands,
	size_t N, size_t L, size_t B, int stream_idx = 0 );


__global__ void kernel_elementwise_dolstm_forward (
	dtype *__restrict__ gc,
	dtype *__restrict__ g2,
	dtype *__restrict__ b,
	dtype *__restrict__ h,
	dtype *__restrict__ max_o,
	dtype *__restrict__ c,
	dtype *__restrict__ ct,
	dtype *__restrict__ prev_c,
	dtype *__restrict__ rands,
	size_t N, size_t L, size_t B );

void cu_elementwise_dolstm_backward (
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
	size_t N, size_t L, size_t B, int stream_idx = 0 );

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
	size_t N, size_t L, size_t B );


/******************************/

void cu_elementwise_aclstm_forward (
	dtype *__restrict__ g,
	dtype *__restrict__ g2,
	dtype *__restrict__ b,
	dtype *__restrict__ h,
	dtype *__restrict__ max_o,
	dtype *__restrict__ c,
	dtype *__restrict__ prev_c,
	dtype *__restrict__ rands,
	size_t N, size_t L, size_t B, int stream_idx = 0 );


__global__ void kernel_elementwise_aclstm_forward (
	dtype *__restrict__ gc,
	dtype *__restrict__ g2,
	dtype *__restrict__ b,
	dtype *__restrict__ h,
	dtype *__restrict__ max_o,
	dtype *__restrict__ c,
	dtype *__restrict__ prev_c,
	dtype *__restrict__ rands,
	size_t N, size_t L, size_t B );

void cu_elementwise_aclstm_backward (
	dtype *__restrict__ dg,
	dtype *__restrict__ dh,
	dtype *__restrict__ c,
	dtype *__restrict__ dc,
	dtype *__restrict__ g,
	dtype *__restrict__ prev_c,
	dtype *__restrict__ prev_dc,
	dtype *__restrict__ h,
	dtype *__restrict__ max_o,
	size_t N, size_t L, size_t B, int stream_idx = 0 );

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
	size_t N, size_t L, size_t B );

/******************************/

void cu_elementwise_aslstm_forward (
	dtype *__restrict__ g,
	dtype *__restrict__ g2,
	dtype *__restrict__ b,
	dtype *__restrict__ h,
	dtype *__restrict__ max_o,
	dtype *__restrict__ c,
	dtype *__restrict__ prev_c,
	dtype *__restrict__ rands,
	size_t N, size_t L, size_t B, int stream_idx = 0 );


__global__ void kernel_elementwise_aslstm_forward (
	dtype *__restrict__ gc,
	dtype *__restrict__ g2,
	dtype *__restrict__ b,
	dtype *__restrict__ h,
	dtype *__restrict__ max_o,
	dtype *__restrict__ c,
	dtype *__restrict__ prev_c,
	dtype *__restrict__ rands,
	size_t N, size_t L, size_t B );

void cu_elementwise_aslstm_backward (
	dtype *__restrict__ dg,
	dtype *__restrict__ dh,
	dtype *__restrict__ c,
	dtype *__restrict__ dc,
	dtype *__restrict__ g,
	dtype *__restrict__ prev_c,
	dtype *__restrict__ prev_dc,
	dtype *__restrict__ h,
	dtype *__restrict__ max_o,
	size_t N, size_t L, size_t B, int stream_idx = 0 );

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
	size_t N, size_t L, size_t B );


/******************************/

void cu_elementwise_splstm_forward (
	dtype *__restrict__ g,
	dtype *__restrict__ g2,
	dtype *__restrict__ b,
	dtype *__restrict__ h,
	dtype *__restrict__ max_o,
	dtype *__restrict__ c,
	dtype *__restrict__ ct,
	dtype *__restrict__ prev_c,
	dtype *__restrict__ rands,
	size_t N, size_t L, size_t B, int stream_idx = 0 );


__global__ void kernel_elementwise_splstm_forward (
	dtype *__restrict__ gc,
	dtype *__restrict__ g2,
	dtype *__restrict__ b,
	dtype *__restrict__ h,
	dtype *__restrict__ max_o,
	dtype *__restrict__ c,
	dtype *__restrict__ ct,
	dtype *__restrict__ prev_c,
	dtype *__restrict__ rands,
	size_t N, size_t L, size_t B );

void cu_elementwise_splstm_backward (
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
	size_t N, size_t L, size_t B, int stream_idx = 0 );

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
	size_t N, size_t L, size_t B );



/******************************/


void cu_elementwise_adagrad (
	dtype learning_rate,
	dtype *__restrict__ p,
	dtype *__restrict__ d,
	dtype *__restrict__ m,
	size_t N, int stream_idx = 0 );


__global__ void kernel_elementwise_adagrad (
	dtype learning_rate,
	dtype *__restrict__ p,
	dtype *__restrict__ d,
	dtype *__restrict__ m,
	size_t N );

void cu_elementwise_adadelta (
	dtype learning_rate, dtype rho,
	dtype *__restrict__ p,
	dtype *__restrict__ d,
	dtype *__restrict__ m,
	dtype *__restrict__ u,
	size_t N, int stream_idx = 0 );


__global__ void kernel_elementwise_adadelta (
	dtype learning_rate, dtype rho,
	dtype *__restrict__ p,
	dtype *__restrict__ d,
	dtype *__restrict__ m,
	dtype *__restrict__ u,
	size_t N );

void cu_elementwise_adadelta_decay (
	dtype learning_rate, dtype rho,
	dtype *__restrict__ p,
	dtype *__restrict__ d,
	dtype *__restrict__ m,
	dtype *__restrict__ u,
	size_t N, dtype decay = 0, int stream_idx = 0 );


__global__ void kernel_elementwise_adadelta_decay (
	dtype learning_rate, dtype rho,
	dtype *__restrict__ p,
	dtype *__restrict__ d,
	dtype *__restrict__ m,
	dtype *__restrict__ u,
	size_t N, dtype decay );


void cu_rand ( dtype *__restrict__ data, size_t elements );
void cu_randn ( dtype *__restrict__ data, size_t elements, dtype mean, dtype stddev );
__global__ void kernel_elementwise_div_scalar ( dtype *__restrict__ c, dtype *__restrict__ src, dtype scalar,
		size_t n );
void cu_div_scalar ( dtype *__restrict__ data, dtype *__restrict__ src, dtype scalar, size_t elements,
					 int stream_idx = 0 );
__global__ void kernel_elementwise_mult_scalar ( dtype *__restrict__ c, dtype *__restrict__ src, dtype scalar,
		size_t n );
void cu_mult_scalar ( dtype *__restrict__ data, dtype *__restrict__ src, dtype scalar, size_t elements,
					  int stream_idx = 0 );

__global__ void kernel_elementwise_cmp ( dtype *__restrict__ c, dtype *__restrict__ a, dtype threshold, size_t n );
void cu_cmp ( dtype *__restrict__ data, dtype *__restrict__ src, dtype scalar, size_t elements, int stream_idx = 0 );

__global__ void kernel_elementwise_cmp_matrix ( dtype *__restrict__ c, dtype *__restrict__ a,
		dtype *__restrict__ threshold, size_t n );
void cu_cmp_matrix ( dtype *__restrict__ data, dtype *__restrict__ src, dtype *__restrict__ matrix, size_t elements,
					 int stream_idx = 0 );


void cu_elementwise_sparselstm_sparsity ( size_t N, size_t L, size_t B, size_t S, dtype *__restrict__ b, dtype corr );
__global__ void kernel_elementwise_sparselstm_sparsity ( size_t N, size_t L, size_t B, size_t S,
		dtype *__restrict__ b, dtype corr );

#endif
