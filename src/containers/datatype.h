/*
 *
 * Author: Kamil Rocki
 *
 *	PRECISE_MATH will enable double type and things
 *	such as gradient check
 *
 */


#ifndef __DTYPE_H_
#define __DTYPE_H_

#ifdef __PRECISE_MATH__
	
	#define dtype double
	#define cl_dtype cl_double
	
	#define cblas_gemm cblas_dgemm
	
	/* use appropriate versions of functions for floats */
	#define _log2 log2
	#define _log log
	#define _tanh tanh
	#define _pow pow
	#define _exp exp
	#define _sqrt sqrt
	#define _fabs fabs
	
#else /* PRECISE_MATH not defined */
	
	#define dtype float
	#define cl_dtype cl_float
	
	#define cblas_gemm cblas_sgemm
	
	/* use appropriate versions of functions for floats */
	#define _log2 log2f
	#define _log logf
	#define _tanh tanhf
	#define _pow powf
	#define _exp expf
	#define _sqrt sqrtf
	#define _fabs fabsf
	
#endif  /* __PRECISE_MATH__ */

#ifdef __CUDA_MATRIX__
	#include <containers/cu_matrix.h>
#endif

#ifdef __CL_MATRIX__
	#include <containers/cl_matrix.h>
#endif

#include <containers/c_matrix.h>

#ifdef __CUDA_MATRIX__
	typedef cu_matrix<dtype> MatrixType;
	typedef cu_matrix<dtype> Matrix;
	typedef cu_matrix<int> MatrixXi;
#else
	
	#ifdef __CL_MATRIX__
		typedef cl_matrix<dtype> MatrixType;
		typedef cl_matrix<dtype> Matrix;
		typedef cl_matrix<int> MatrixXi;
	#else
		typedef matrix<dtype> MatrixType;
		typedef matrix<dtype> Matrix;
		typedef matrix<int> MatrixXi;
	#endif
#endif

#endif /* __DTYPE_H_ */
