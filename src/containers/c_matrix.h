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
* Matrix<T>
*
*/

#ifndef __C_MATRIX__
#define __C_MATRIX__

#include <containers/datatype.h>
#include <random>
#include <iostream>
#include <string.h>
#include <typeinfo>
#include <functional>
#include <assert.h>

#ifdef __USE_BLAS__
	#include <cblas.h>
#endif

/* TODO: column-major or row-major */
/* 0-indexing vs 1-indexing */
/* i - which row, j - which column */

/* column-major indexing 0-based*/
#define COLMAJOR0(i, j) ((j) * rows() + (i))
/* row-major indexing 0-based*/
#define ROWMAJOR0(i, j) ((i) * cols() + (j))
/* column-major indexing 1-based*/
#define COLMAJOR1(i, j) ((j) * rows() + (i)+1)
/* row-major indexing 1-based*/
#define ROWMAJOR1(i, j) ((i) * cols() + (j)+1)

#define ORDER COLMAJOR0

#if defined(__GPU__) || defined(__CUDACC__)
	
	#include <cuda_runtime_api.h>
	#include <cuda.h>
	
#endif

template <typename T>
class matrix {

	public:
	
		T *_data_ = nullptr;
		
		size_t _rows = 0;
		size_t _cols = 0;
		size_t _size = 0;
		size_t bytes = 0;
		
		size_t bytes_allocated = 0;
		bool write = true;
		
		void alloc ( size_t rows, size_t cols ) {
		
			bytes_allocated = rows * cols * sizeof ( T );
			
			//if using cuda matrix: use page locked memory
			#ifdef __CUDA_MATRIX__
			cudaMallocHost ( ( void ** ) & ( _data_ ),  bytes_allocated );
			#else
			_data_ = ( T * ) malloc ( bytes_allocated );
			#endif
			memset ( _data_, '\0', bytes_allocated );
			
			write = true;
			
		}
		
		void dealloc() {
		
			if ( bytes_allocated > 0 ) {
			
				if ( _data_ != nullptr ) {
				
					//if using cuda matrix: use page locked memory
					#ifdef __CUDA_MATRIX__
					cudaFreeHost ( _data_ );
					#else
					free ( _data_ );
					#endif
					
					bytes_allocated = 0;
					_data_ = nullptr;
					
				}
				
			}
			
		}
		
		void block ( const matrix<T> &src, const size_t r, const size_t c, const size_t nr, const size_t nc ) {
		
			resize ( nr, nc );
			
			for ( size_t i = 0; i < nr; i++ )
				for ( size_t j = 0; j < nc; j++ )
					this->operator() ( i, j ) = src ( r + i, c + j );
					
			write = true;
			
		}
		
		
		template<typename F>
		void block_forall ( const size_t offset, const size_t cols, const F &lambda ) {
		
			for ( size_t i = 0; i < rows(); i++ )
				for ( size_t j = 0; j < cols; j++ )
				
					this->operator() ( i, j + offset ) = lambda();
					
			write = true;
		}
		
		void block_rand ( const size_t offset, const size_t cols, const dtype range_min, const dtype range_max ) {
		
			std::random_device rd;
			std::mt19937 mt ( rd() );
			std::uniform_real_distribution<T> dis ( range_min, range_max );
			
			for ( size_t i = 0; i < rows(); i++ )
				for ( size_t j = 0; j < cols; j++ )
				
					this->operator() ( i, j + offset ) = dis ( mt );
					
			write = true;
		}
		
		/* * * * * READ-WRITE ACCESS METHODS * * * * */
		
		T *data() { return _data_; }
		
		virtual ~matrix() {
		
			dealloc();
			
		}
		
		void sync_device() {
		
		}
		
		void sync_host() {
		
		}
		
		/* main constr */
		matrix ( const size_t rows, const size_t cols ) {
		
			resize ( rows, cols );
			write = true;
		}
		
		inline T &operator() ( const size_t i ) {
		
			assert ( i < _size );
			write = true;
			return data() [ i ];
		}
		
		inline T &operator() ( const size_t i, const size_t j ) {
		
			assert ( i < _rows );
			assert ( j < _cols );
			write = true;
			return data() [ORDER ( i, j )];
			
		}
		
		void setZero() {
		
			memset ( data(), '\0', bytes_allocated );
			write = true;
		}
		
		void resize ( const size_t new_rows, const size_t new_cols ) {
		
			size_t other_bytes = new_rows * new_cols * sizeof ( T );
			
			if ( other_bytes > bytes_allocated ) {
			
				/* realloc */
				dealloc();
				alloc ( new_rows, new_cols );
				
			}
			
			_rows = new_rows;
			_cols = new_cols;
			_size = new_rows * new_cols;
			bytes = other_bytes;
			
			write = true;
		}
		
		void set ( const matrix &src ) {
		
			//std::cout << "copy " << src.rows() << " x " << src.cols() << " " << src.bytes_allocated / 1024 << "kB" << std::endl;
			resize ( src.rows(), src.cols() );
			
			#ifdef __CUDA_MATRIX__
			cudaMemcpy ( _data_, src.data(), src.bytes, cudaMemcpyHostToHost );
			#else
			// matrix-matrix copy
			memcpy ( _data_, src.data(), src.bytes );
			#endif
			
			
			
			write = true;
		}
		
		//matrix ( const matrix &other ) = delete;
		//matrix &operator= ( const matrix &other ) = delete;
		
		matrix &operator= ( const matrix &other ) {
		
			write = true;
			set ( other );
			return *this;
			
		};
		
		/*TODO: move semantics */
		
		/* copy constr from matrix */
		matrix ( const matrix &other ) { write = true; operator= ( other ); }
		
		/* TODO: put them in one function, variable number of arguments */
		template<typename F>
		void forall ( const F &lambda ) {
		
			write = true;
			
			for ( size_t i = 0; i < size(); i++ )
				this->data() [i] = lambda ( );
				
		}
		
		template<typename F>
		void forall ( const F &lambda, const matrix<T> &x ) {
		
			write = true;
			
			for ( size_t i = 0; i < size(); i++ )
				this->data() [i] = lambda ( x.data() [i] );
				
		}
		
		template<typename F>
		void forall ( const F &lambda, const matrix<T> &x, const T y ) {
		
			write = true;
			
			for ( size_t i = 0; i < size(); i++ )
				this->data() [i] = lambda ( x.data() [i], y );
				
		}
		
		template<typename F>
		void forall ( const F &lambda, const matrix<T> &x, const matrix<T> &y ) {
		
			write = true;
			
			for ( size_t i = 0; i < size(); i++ )
				this->data() [i] = lambda ( x.data() [i], y.data() [i] );
				
		}
		
		template<typename F>
		void forall ( const F &lambda, const matrix<T> &x, const matrix<T> &y, const matrix<T> &z ) {
		
			write = true;
			
			for ( size_t i = 0; i < size(); i++ )
				this->data() [i] = lambda ( x.data() [i], y.data() [i], z.data() [i] );
				
		}
		
		template<typename F>
		void forall_colwise ( const F &lambda, const matrix<T> &x, const matrix<T> &v ) {
		
			write = true;
			
			// v is a vector
			for ( size_t i = 0; i < v.size(); i++ )
				for ( size_t j = 0; j < x.cols(); j++ )
					this->operator() ( i, j ) =
						lambda ( x ( i, j ), v ( i ) );
						
		}
		
		template<typename F>
		void forall_rowwise ( const F &lambda, const matrix<T> &x, const matrix<T> &v ) {
		
			write = true;
			
			// v is a vector
			for ( size_t j = 0; j < v.size(); j++ )
				for ( size_t i = 0; i < x.rows(); i++ )
					this->operator() ( i, j ) =
						lambda ( x ( i, j ), v ( j ) );
						
		}
		
		template<typename F>
		void forall ( const F &lambda, const matrix<T> &x, const matrix<T> &y, const matrix<T> &z, const T a ) {
		
			write = true;
			
			for ( size_t i = 0; i < size(); i++ )
				this->data() [i] = lambda ( x.data() [i], y.data() [i], z.data() [i], a );
				
		}
		
		/* READ-ONLY */
		template<typename F>
		T reduction ( const F &lambda, const matrix<T> &x, T val ) const {
		
			for ( size_t i = 0; i < size(); i++ )
				val = lambda ( x.data() [i], val );
				
			return val;
		}
		
		/* TODO: change to general colwise/rowwise reductions */
		void sum_colwise ( const matrix<T> &x ) {
		
			for ( size_t j = 0; j < cols(); j++ )
				for ( size_t i = 0; i < x.rows(); i++ )
					data() [j] += x ( i, j );
					
					
		}
		
		void sum_rowwise ( const matrix<T> &x ) {
		
			for ( size_t i = 0; i < rows(); i++ )
				for ( size_t j = 0; j < x.cols(); j++ )
					data() [i] += x ( i, j );
					
		}
		
		/* * * * * READ-ONLY ACCESS METHODS * * * * */
		
		/* default constr */
		matrix() { /* we don't know the dimensions, so do nothing */ }
		
		const inline size_t rows() const { return _rows; }
		const inline size_t cols() const { return _cols; }
		const inline size_t size() const { return _size; }
		
		const T *data() const { return _data_;	}
		const inline T &operator() ( const size_t i ) const { return data() [ i ]; }
		const inline T &operator() ( const size_t i, const size_t j ) const {
			return data() [ ORDER ( i, j ) ]; /* read-only access */
		}
		
		friend std::ostream &operator<< ( std::ostream &os, const matrix<T> &obj ) {
		
			// Display dimensions.
			std::cout << "RAW: " << obj.size() << std::endl;
			
			for ( size_t i = 0; i < obj.size(); i++ )
				os << obj.data() [ i ] << " " ;
				
			std::cout << std::endl;
			
			os << "[" << obj.size() << " = " << obj.rows() << " x " << obj.cols() << std::endl;
			
			// Display elements.
			if ( typeid ( T ) == typeid ( int ) ) {
				for ( size_t i = 0; i < obj.size(); i++ )
					os << ( char ) obj.data() [ i ];
					
			}
			
			else {
			
				for ( size_t i = 0; i < obj.rows(); i++ ) {
				
					for ( size_t j = 0; j < obj.cols(); j++ ) {
						os << std::setw ( 8 ) << std::setprecision ( 4 ) << obj ( i, j ) << "";
						
						if ( j % ( obj.cols() / 4 ) == ( obj.cols() / 4 - 1 ) ) os << "    ";
						
					}
					
					std::cout << std::endl;
				}
				
				
			}
			
			os << "]";
			
			return os;
		}
		
		const inline T sum() const {
		
			T s = 0;
			
			/*todo: do this better */
			for ( size_t i = 0; i < size(); i++ )
				s += this->operator() ( i ); /* uses read-only data() */
				
			return s;
			
		}
		
		dtype *operator[] ( size_t offset ) {
		
			return _data_ + offset;
			
		}
		
		template<class Archive>
		void serialize ( Archive &archive ) {
		
			archive ( _rows, _cols );
			
			for ( int i = 0; i < _rows; i++ )
				for ( int j = 0; j < _cols; j++ )
					archive ( this->operator() ( i, j ) );
					
		}
};

template<typename F, typename...X>
void elementwise ( const F &lambda, size_t elements, X ...x ) {

	for ( size_t i = 0; i < elements; i ++ )
	
		lambda ( x..., i );
		
		
		
}

void elementwise_mult ( dtype *a, dtype *b, dtype *c, size_t elements ) {

	for ( size_t i = 0; i < elements; i ++ )
	
		a[i] = b[i] * c[i];
		
		
}

template <typename T>
void row ( matrix<T> &out, const matrix<T> &m, const size_t which ) {

	for ( size_t i = 0; i < m.cols(); i++ )
		out ( 0, i ) = m ( which, i );
		
}

template <typename T>
void col ( matrix<T> &out, const matrix<T> &m, const size_t which ) {

	for ( size_t i = 0; i < m.rows(); i++ )
		out ( i, 0 ) = m ( i, which );
		
}

template <typename T>
void set_col_one_hot ( matrix<T> &m, const size_t col, const size_t bit ) {

	//not zeroing here!
	//set one
	m ( bit, col ) = 1;
	
}

template <typename T>
void set_row_one_hot ( matrix<T> &m, const size_t row, const size_t bit ) {

	//not zeroing here!
	//set one
	m ( row, bit ) = 1;
	
}

template <typename T>
void eye ( matrix<T> &m ) {

	m.setZero();
	
	for ( size_t i = 0; i < m.rows(); i++ )
		m ( i, i ) = 1;
		
}

template <typename T>
void ones ( matrix<T> &m ) {

	m.forall ( [&] () { return 1; }, m );
	
}

template <typename T>
void rand_uniform ( matrix<T> &m, const double range_min, const double range_max ) {

	std::random_device rd;
	std::mt19937 mt ( rd() );
	std::uniform_real_distribution<T> dis ( range_min, range_max );
	
	for ( int i = 0; i < m.rows(); i++ ) {
		for ( int j = 0; j < m.cols(); j++ )
			m ( i, j ) = dis ( mt );
	}
	
}

template <typename T>
void randn ( matrix<T> &m, const T mean, const T stddev ) {

	// random number generator
	std::random_device rd;
	std::mt19937 mt ( rd() );
	std::normal_distribution<T> __randn ( mean, stddev );
	
	for ( int i = 0; i < m.rows(); i++ )
		for ( int j = 0; j < m.cols(); j++ )
			m ( i, j ) = __randn ( mt );
			
}

template <typename T>
void matrix_init ( matrix<T> &m ) {

	//randn ( m, static_cast<T> ( 0 ), static_cast<T> ( 0.1 ) );
	randn ( m, static_cast<T> ( 0 ), static_cast<T> ( 1.0 ) / _sqrt ( m.rows() + m.cols() ) );
	
}

template <typename T>
void GEMM ( matrix<T> &c, const matrix<T> &a, const matrix<T> &b,
			const bool aT = false, const bool bT = false,
			const T alpha = 1.0f, const T beta = 1.0f ) {
			
	enum CBLAS_TRANSPOSE transA = aT ? CblasTrans :
								  CblasNoTrans;
	enum CBLAS_TRANSPOSE transB = bT ? CblasTrans :
								  CblasNoTrans;
								  
	size_t M = c.rows();
	size_t N = c.cols();
	size_t K = aT ? a.rows() : a.cols();
	
	/* TODO : make it flexible */
	size_t lda = aT ? K : M;
	size_t ldb = bT ? N : K;
	size_t ldc = M;
	
	cblas_gemm ( CblasColMajor, transA, transB, M, N, K, alpha,
				 a.data(), lda, b.data(), ldb, beta, c.data(), ldc );
				 
}

//f(x) = sigm(x)
inline dtype logistic ( const dtype x ) { return ( dtype ) 1 / ( ( dtype ) 1 + _exp ( -x ) ); }

//f'(x) = f(x)(1-f(x))
inline dtype logistic_prime ( const dtype x ) { return x * ( ( dtype ) 1 - x ); }

inline dtype tanh_prime ( const dtype x ) { return ( dtype ) 1 - x * x; };


//add colwise
template <typename T>
void ADDC ( matrix<T> &c, const matrix<T> &v, const T alpha = 1.0f ) {

	c.forall_colwise ( [&] ( dtype x, dtype y ) {return  x + y ; }, c, v );
	
}

template <typename T>
void ADDR ( matrix<T> &c, const matrix<T> &v, const T alpha = 1.0f ) {

	c.forall_rowwise ( [&] ( dtype x, dtype y ) {return  x + y ; }, c, v );
	
}

//divide rowwise
template <typename T>
void DIVR ( matrix<T> &c, const matrix<T> &v, const T alpha = 1.0f ) {

	c.forall_rowwise ( [&] ( dtype x, dtype y ) {return  x / y ; }, c, v );
	
}

template <typename T>
void DIVC ( matrix<T> &c, const matrix<T> &v, const T alpha = 1.0f ) {

	c.forall_colwise ( [&] ( dtype x, dtype y ) {return  x / y ; }, c, v );
	
}

template <typename T>
void ZERO ( matrix<T> &m ) {

	m.setZero();
	
}

template <typename T>
void TANH ( matrix<T> &m ) {

	m.forall ( [&] ( dtype x ) { return _tanh ( x ); }, m );
}

template <typename T>
void EXP ( matrix<T> &m ) {

	m.forall ( [&] ( dtype x ) { return _exp ( x ); }, m );
	
}

template <typename T>
void ADD ( matrix<T> &m, const T val ) {

	m.forall ( [&] ( dtype x, dtype y ) { return x + y; }, m, val );
}

template <typename T>
void SUB ( matrix<T> &m, const T val ) {

	ADD ( m, -val );
}

template <typename T>
void SUBM ( matrix<T> &c, const matrix<T> &a, const matrix<T> &b ) {

	/*a, b - read-only access */
	c.forall ( [&] ( dtype x, dtype y ) { return x - y; }, a, b );
	
}


template <typename T>
void DTANH ( matrix<T> &dx, const matrix<T> &sy, const matrix<T> &dy ) {

	//f'(x) = 1-(f(x))^2
	auto tanh_prime = [] ( const dtype x ) { return ( T ) 1 - x * x; };
	dx.forall ( [&] ( dtype x, dtype y ) { return tanh_prime ( x ) * y; }, sy, dy );
}

#define _max(x,y) ((x) > (y) ? (x) : (y))

template <typename T>
T MAX ( const matrix<T> &m ) {

	T val = -INFINITY;
	
	return m.reduction ( [&] ( dtype x, dtype y ) { return _max ( x, y ); }, m, val );
	
}

#define _min(x,y) ((x) < (y) ? (x) : (y))

template <typename T>
T MIN ( const matrix<T> &m ) {

	T val = INFINITY;
	
	return m.reduction ( [&] ( dtype x, dtype y ) { return _min ( x, y ); }, m, val );
	
}

//colwise sum
template <typename T>
void SUMC ( matrix<T> &sums, const matrix<T> &m ) {

	sums.sum_colwise ( m );
	
}

template <typename T>
void SUMR ( matrix<T> &sums, const matrix<T> &m ) {

	sums.sum_rowwise ( m );
	
}

template <typename T>
void ABS ( matrix<T> &out, const matrix<T> &in ) {

	out.forall ( [&] ( dtype x ) { return _fabs ( x ); }, out );
	
}

template <typename T>
void ADDM ( matrix<T> &c, const matrix<T> &a, const matrix<T> &b ) {

	c.forall ( [&] ( dtype x, dtype y ) { return x + y; }, a, b );
	
}

#endif /* C_MATRIX */