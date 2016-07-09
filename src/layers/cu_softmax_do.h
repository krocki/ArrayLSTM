/*
* 	@Author: kmrocki
* 	@Date:   2016-03-24 15:25:43
* 	@Last Modified by:   kmrocki
* 	@Last Modified time: 2016-03-24 18:36:57
*
* 	Implementation of softmax layer
*/

#ifndef __SOFTMAX_H__
#define __SOFTMAX_H__

#include <vector>
#include <parameters.h>
#include <timelayer.h>
#include <state.h>
#include <containers/cu_matrix.h>

template <typename T>
class Softmax : public Timelayer<T> {

	public:
	
		//temp storage for probability sums
		T sums;
		T maxima;
		T dropout_mask;
		T rands;
		
		/* main constructor */
		Softmax ( size_t _in, size_t _out, size_t _B, size_t _S ) :
			Timelayer<T> ( _in, _out, _B, _S,
			
		{	"softmax dropout"		},
		
		{
			/* define states */
			std::make_tuple ( "p", _B, _out )
			
		}, {
		
			/* define params */
			std::make_tuple ( "W", _in, _out ),
			std::make_tuple ( "b", 1, _out ),
			
			std::make_tuple ( "B_ones", _B, 1 ),
			std::make_tuple ( "M_ones", _out, 1 )
			
		} ) {
		
			/*init*/
			matrix_init ( p ( W ) );
			
			sums = T ( _B, 1 );
			maxima = T ( _B, 1 );
			
			dropout_mask = T ( _B, _in );
			rands = T ( _B, _in );
			
			p ( M_ones ).forall ( [ = ] () { return 1; } );
			p ( M_ones ).sync_device();
			p ( B_ones ).forall ( [ = ] () { return 1; } );
			p ( B_ones ).sync_device();
			
		}
		
		virtual void forward ( bool apply_dropout, size_t t = 1 ) {
		
			s ( t, x ).sync_device();
			
			dtype keep_ratio = 0.5;
			
			//v3
			if ( apply_dropout ) {
			
				cu_rand ( rands.cu_data, rands.size() );
				cu_cmp ( dropout_mask.cu_data, rands.cu_data, keep_ratio, rands.size() );
				cu_elementwise_mult ( s ( t, x ).cu_data, dropout_mask.cu_data, s ( t, x ).cu_data, s ( t, x ).size() );
				cu_mult_scalar ( s ( t, x ).cu_data, s ( t, x ).cu_data, ( dtype ) ( 1.0 / keep_ratio ), s ( t, x ).size() );
				
			}
			
			CU_GEMM ( s ( t, p ), s ( t, x ), p ( W ), false, false, 1, 0 );
			
			cu_add_row_vector ( & ( s ( t, p ) ).cu_data[0], & ( p ( b ) ).cu_data[0], this->N, s ( t, p ).rows() );
			
			// for numerical stability
			
			// s ( t, p ).sync_host();
			
			cu_row_max ( maxima.cu_data,  s ( t, p ).cu_data, this->N, s ( t, p ).rows() );
			cu_sub_col_vector ( & ( s ( t, p ) ).cu_data[0], maxima.cu_data, this->N, s ( t, p ).rows() );
			// std::cout << s(t, p) << std::endl;
			// maxima.sync_host();
			// std::cout << "MAX" << std::endl;
			// std::cout << maxima << std::endl;
			
			cu_exp ( & ( s ( t, p ) ).cu_data[0], this->N * s ( t, p ).rows() );
			
			CU_GEMM ( sums, s ( t, p ), p ( M_ones ), false, false, 1, 0 );
			
			cu_div_col_vector ( & ( s ( t, p ) ).cu_data[0], & ( sums.cu_data[0] ), this->N, s ( t, p ).rows() );
			
			s ( t, p ).sync_host();
			// B x N ./ B x 1
			
		}
		
		virtual void backward ( bool apply_dropout, size_t t ) {
		
			cu_sub ( & ( g ( t, p ).cu_data[0] ), & ( s ( t, p ).cu_data[0] ), & ( g ( t, y ).cu_data[0] ), this->N * s ( t,
					 p ).rows() );
					 
			// propagate through linear layer h->y
			cublasSetStream ( handle, streams[1] );
			CU_GEMM ( d ( W ), s ( t, x ), g ( t, p ), true, false );
			cublasSetStream ( handle, streams[2] );
			CU_GEMM ( d ( b ), p ( B_ones ), g ( t, p ), true, false );
			
			// propagate through linear layer x->h
			cublasSetStream ( handle, streams[3] );
			CU_GEMM ( g ( t, x ), g ( t, p ), p ( W ), false, true, 1, 0 );
			
			sync_stream ( 1 );
			sync_stream ( 2 );
			sync_stream ( 3 );
			
			if ( apply_dropout )
				cu_elementwise_mult ( g ( t, x ).cu_data, g ( t, x ).cu_data, s ( t, mask ).cu_data, s ( t, mask ).size() );
				
				
				
		}
		
		virtual void reset ( dtype std ) {};
		
		/* optional */
		/* copy constr */
		// Softmax( const Softmax& src) : Timelayer(src.M, src.N, src.B, src.S) { }
		
		/* assignent operator */
		// Softmax& operator=( const Softmax& src) { Timelayer::operator=(src); return *this; }
		
};


#endif