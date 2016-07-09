/*
*   @Author: kmrocki
*   @Date:   2016-02-19 15:25:43
*   @Last Modified by:   kmrocki
*   @Last Modified time: 2016-03-24 18:36:57
*
*   Implementation of aLSTM layer
*
*/

#ifndef _aLSTM_H_
#define _aLSTM_H_

#include <vector>
#include <parameters.h>
#include <timelayer.h>
#include <state.h>

#include <containers/cu_matrix.h>

/* gates:

[  0:    N-1 ] = i
[  N:   2N-1 ] = o
[ 2N:   3N-1 ] = f
[ 3N:   4N-1 ] = u

*/

#define N this->N
#define B this->B

template <typename T>
class aLSTM : public Timelayer<T> {

	public:
	
		using Timelayer<T>::s;
		T rands;
		
		size_t L = 1; // memory cells per hidden neuron
		
		/* main constructor */
		aLSTM ( size_t _M, size_t _N, size_t _B,
				size_t _S, size_t _L = 1 ) : L ( _L ),
				
			Timelayer<T> ( _M, _N, _B, _S,
			
		{	"alstm"		},
		
		{
			/* define states */
			std::make_tuple ( "h", _B, _N ),
			std::make_tuple ( "max_o", _B, _N ),
			std::make_tuple ( "c", _B, _N * _L ),
			std::make_tuple ( "ct", _B, _N * _L ),
			std::make_tuple ( "g", _B, 4 * _N * _L ),
			std::make_tuple ( "g2", _B, 4 * _N * _L )
			
		}, {
		
			/* define params */
			std::make_tuple ( "W", _M, 4 * _N * _L ),
			std::make_tuple ( "U", _N, 4 * _N * _L ),
			std::make_tuple ( "b", 1, 4 * _N * _L ),
			
			std::make_tuple ( "B_ones", _B, 1 )
			
		} ) {
		
			std::cout << "alstm levels :" << _L << std::endl;
			/*init*/
			matrix_init ( p ( W ) );
			matrix_init ( p ( U ) );
			
			rands = T ( _B, _N );
			
			// set biases of f gates to 1
			// (http://jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
			p ( b ).block_forall ( 2 * N * L, N * L, [ = ] () { return 1; } );
			//rand
			//p ( b ).block_rand ( 2 * N * L, N * L, 0, 4 );
			
			//p ( b ).block_forall ( 2 * N * L, N, [ = ] () { return 1; } );
			p ( B_ones ).forall ( [ = ] () { return 1; } );
			
			p ( B_ones ).sync_device();
			
		}
		
		~aLSTM() {
		
		
		}
		
#define c_gates 3 * N * L * s(t, g).rows()
#define i_gates 0 * N * L * s(t, g).rows()
#define f_gates 2 * N * L * s(t, g).rows()
#define o_gates 1 * N * L * s(t, g).rows()
		
		virtual void forward ( bool apply_dropout, size_t t = 1 ) {
		
			s ( t, x ).sync_device();
			s ( t - 1, h ).sync_device();
			s ( t - 1, c ).sync_device();
			
			// put in 2 separate streams
			cublasSetStream ( handle, streams[1] );
			CU_GEMM ( s ( t, g ), s ( t, x ), p ( W ), false, false, 1, 0 );
			
			cublasSetStream ( handle, streams[2] );
			CU_GEMM ( s ( t, g2 ), s ( t - 1, h ), p ( U ), false, false, 1, 0 );
			
			sync_stream ( 1 );
			sync_stream ( 2 );
			
			cu_rand ( rands.cu_data, rands.size() );
			
			//fused
			cu_elementwise_alstm_forward (	& ( s ( t, g ).cu_data[0] ),
											& ( s ( t, g2 ).cu_data[0] ),
											& ( p ( b ).cu_data[0] ),
											& ( s ( t, h ).cu_data[0] ),
											& ( s ( t, max_o ).cu_data[0] ),
											& ( s ( t, c ).cu_data[0] ),
											& ( s ( t, ct ).cu_data[0] ),
											& ( s ( t - 1, c ).cu_data[0] ),
											& ( rands.cu_data[0] ),
											N, L, s ( t, c ).rows() );
											
			s ( t, c ).sync_host();
			s ( t, h ).sync_host();
			
		}
		
		virtual void backward ( bool apply_dropout, size_t t ) {
		
			// error coming from higher layers: dh[t] = dy[t]
			
			cu_elementwise_alstm_backward (	& ( g ( t, g ).cu_data[0] ),
											& ( g ( t, y ).cu_data[0] ),
											& ( s ( t, c ).cu_data[0] ),
											& ( s ( t, ct ).cu_data[0] ),
											& ( g ( t, c ).cu_data[0] ),
											& ( s ( t, g ).cu_data[0] ),
											& ( s ( t - 1, c ).cu_data[0] ),
											& ( g ( t - 1, c ).cu_data[0] ),
											s ( t, h ).cu_data,
											& ( s ( t, max_o ).cu_data[0] ),
											N, L, g ( t, c ).rows() );
											
			//backprop through linear part (forward pass step 1)
			//these are computed in parallel
			cublasSetStream ( handle, streams[1] );
			CU_GEMM ( d ( b ), p ( B_ones ), g ( t, g ), true, false );
			
			cublasSetStream ( handle, streams[2] );
			CU_GEMM ( d ( U ), s ( t - 1, h ), g ( t, g ), true, false );
			
			cublasSetStream ( handle, streams[3] );
			CU_GEMM ( d ( W ), s ( t, x ), g ( t, g ), true, false );
			
			cublasSetStream ( handle, streams[4] );
			//backprop into inputs for lower layers
			CU_GEMM ( g ( t, x ), g ( t, g ), p ( W ), false, true );
			
			cublasSetStream ( handle, streams[5] );
			//carry - h state
			CU_GEMM ( g ( t - 1, y ), g ( t, g ), p ( U ), false , true );
			
			sync_stream ( 1 );
			sync_stream ( 2 );
			sync_stream ( 3 );
			sync_stream ( 4 );
			sync_stream ( 5 );
		}
		
		virtual void reset ( dtype std ) {
		
			randn ( s ( 0, h ), ( dtype ) 0, ( dtype ) std );
			randn ( s ( 0, c ), ( dtype ) 0, ( dtype ) std );
			// s ( 0, h ).cu_zero();
			// s ( 0, c ).cu_zero();
		}
		
		/* optional */
		/* copy constr */
		//aLSTM( const aLSTM& src) : Timelayer(src.M, src.N, src.B, src.S)  { }
		
		/* assignent operator */
		//aLSTM& operator=( const aLSTM& src) { Timelayer::operator=(src); return *this; }
		
};

#undef N
#undef B
#undef i_gates
#undef o_gates
#undef c_gates
#undef f_gates

#endif /* _aLSTM_H_ */