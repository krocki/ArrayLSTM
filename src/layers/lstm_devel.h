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
 *
 */

#ifndef _LSTM_H_
#define _LSTM_H_

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

/*#define _i_     block(    0  , 0,       N, B)
#define _o_     block(    N  , 0,       N, B)
#define _f_     2 * N  , 0,       N, B)
#define _c_     block(3 * N  , 0,       N, B)
#define _iof_   block(0      , 0,   3 * N, B)*/

#define forallblock(x, y) for ( size_t i = 0; i < (x); i++ ) for ( size_t j = 0; j < (y) ; j++)

template <typename T>
class LSTM : public Timelayer<T> {

	public:
	
		using Timelayer<T>::s;
		
		/* main constructor */
		LSTM ( size_t _M, size_t _N, size_t _B,
			   size_t _S ) :
			   
			Timelayer<T> ( _M, _N, _B, _S,
			
		{	"lstm"		},
		
		{
			/* define states */
			std::make_tuple ( "h", _B, _N ),
			std::make_tuple ( "c", _B, _N ),
			std::make_tuple ( "g", _B, 4 * _N )
			
		}, {
		
			/* define params */
			std::make_tuple ( "W", _M, 4 * _N ),
			std::make_tuple ( "U", _N, 4 * _N ),
			std::make_tuple ( "b", 1, 4 * _N )
			
		} ) {
		
			/*init*/
			matrix_init ( p ( W ) );
			matrix_init ( p ( U ) );
			
			// set biases of f gates to 1
			// (http://jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
			p ( b ).block_forall ( 2 * N, N, [ = ] () { return 1; } );
			
		}
		
#define c_gates 3 * N * s(t, g).rows()
#define i_gates 0 * N * s(t, g).rows()
#define f_gates 2 * N * s(t, g).rows()
#define o_gates 1 * N * s(t, g).rows()
		
		virtual void forward ( size_t t = 1 ) {
		
			// 1. gates - linear part
			ZERO ( s ( t, g ) );
			GEMM ( s ( t, g ), s ( t, x ), p ( W ) );
			GEMM ( s ( t, g ), s ( t - 1, h ), p ( U ) );
			
			ADDR ( s ( t, g ), p ( b ) );
			
			//fused 2-5
			elementwise ( [] ( dtype * cc, dtype * o, dtype * i, dtype * c, dtype * f, dtype * prev_c, dtype * h, size_t j ) {
			
				// 2. gates - nonlinearities
				i[j] = logistic ( i[j] );
				f[j] = logistic ( f[j] );
				o[j] = logistic ( o[j] );
				c[j] = _tanh ( c[j] );
				
				//3.candidate context
				cc[j] =  i[j] * c[j] + f[j] * prev_c[j];
				
				// 4. nonlinearity : c[t] = tanh(c[t])
				cc[j] = _tanh ( cc[j] );
				
				//5. update hidden state : h[t] = o[t] * c[t]
				h[j] = o[j] * cc[j];
				
			}, N * B,
			
			s ( t, c ) [0], s ( t, g ) [o_gates], s ( t, g ) [i_gates], s ( t, g ) [c_gates], s ( t, g ) [f_gates],
			s ( t - 1, c ) [0], s ( t, h ) [0] );
			
		}
		
		virtual void backward ( size_t t ) {
		
			// error coming from higher layers: dh[t] = dy[t]
			//g ( t, h ).set ( g ( t, y ) );
			
			//fused 5-4
			elementwise ( [] ( dtype * gc, dtype * gh, dtype * o, dtype * cc, size_t i ) {
			
				// backpropagate through (forward pass step 5) : dc[t] = dh[t] * o[t]
				gc[i]  = gc[i] + gh[i] * o[i];
				
				// propagate through tanh nonlinearity (forward pass step 4)
				gc[i] = gc[i] * tanh_prime ( cc[i] );
				
			}, N * B, g ( t, c ) [0], g ( t, y ) [0], s ( t, g ) [o_gates], s ( t, c ) [0] );
			
			// backpropagate through gates (forward pass step 3)
			elementwise_mult ( g ( t, g ) [N * B], g ( t, y ) [0], s ( t, c ) [0], N * B );
			elementwise_mult ( g ( t, g ) [0], g ( t, c ) [0], s ( t, g ) [3 * N * B], N * B );
			elementwise_mult ( g ( t, g ) [2 * N * B], g ( t, c ) [0], s ( t - 1, c ) [0], N * B );
			elementwise_mult ( g ( t, g ) [3 * N * B], g ( t, c ) [0], s ( t, g ) [0], N * B );
			
			// propagate do, di, df, du though nonlinear parts (forward pass step 2)
			elementwise ( [] ( dtype * z, dtype * x, dtype * y, size_t i ) {
			
				z[i] = y[i] * logistic_prime ( x[i] );
				
			}, 3 * N * B, g ( t, g ) [0], s ( t, g ) [0], g ( t, g ) [0] );
			
			elementwise ( [] ( dtype * z, dtype * x, dtype * y, size_t i ) {
			
				z[i] = y[i] * tanh_prime ( x[i] );
				
			}, N * B, g ( t, g ) [3 * N * B], s ( t, g ) [3 * N * B], g ( t, g ) [3 * N * B] );
			
			//backprop through linear part (forward pass step 1)
			SUMC ( d ( b ), g ( t, g ) );
			GEMM ( d ( U ), s ( t - 1, h ), g ( t, g ), true, false );
			GEMM ( d ( W ), s ( t, x ), g ( t, g ), true, false );
			
			//backprop into inputs for lower layers
			GEMM ( g ( t, x ), g ( t, g ), p ( W ), false, true );
			
			//carry - h state
			GEMM ( g ( t - 1, y ), g ( t, g ), p ( U ), false , true );
			
			//carry - c state
			elementwise_mult ( g ( t - 1, c ) [0], g ( t, c ) [0], s ( t, g ) [2 * N * B], N * B );
			
		}
		
		virtual void reset ( dtype std ) {
		
			randn ( s ( 0, h ), ( dtype ) 0, ( dtype ) std );
			randn ( s ( 0, c ), ( dtype ) 0, ( dtype ) std );
			
		}
		
		/* optional */
		/* copy constr */
		//LSTM( const LSTM& src) : Timelayer(src.M, src.N, src.B, src.S)  { }
		
		/* assignent operator */
		//LSTM& operator=( const LSTM& src) { Timelayer::operator=(src); return *this; }
		
};

#undef N
#undef B

#endif /* _LSTM_H_ */