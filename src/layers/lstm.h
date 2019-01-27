/*
 *
 * Author: Kamil Rocki
 *
 *
 */

#ifndef _LSTM_H_
#define _LSTM_H_

#include <vector>
#include <parameters.h>
#include <timelayer.h>
#include <state.h>

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
	
	
	
		/* main constructor */
		LSTM ( size_t _M, size_t _N, size_t _B,
			   size_t _S ) :
			   
			Timelayer<T> ( _M, _N, _B, _S,
			
		{	"lstm"		},
		
		{
			/* define states */
			std::make_tuple ( "h", _B, _N ),
			std::make_tuple ( "c", _B, _N ),
			std::make_tuple ( "ct", _B, _N ),
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
			p ( b ).block_forall ( 2, 1, [ = ] () { return 1; } );
			
		}
		
		virtual void forward ( size_t t = 1 ) {
		
			// 1. gates - linear part
			ZERO ( s ( t, g ) );
			GEMM ( s ( t, g ), s ( t, x ), p ( W ) );
			GEMM ( s ( t, g ), s ( t - 1, h ), p ( U ) );
			
			ADDR ( s ( t, g ), p ( b ) );
			
			//T &s_t_g = s ( t, g );
			
			//2. gates - nonlinearities
			forallblock ( B, N ) {
			
				// s ( t, g )._iof_.noalias() = s ( t, g )._iof_.unaryExpr ( (
				// 								 dtype ( * ) ( const dtype ) ) logistic );
				
				// s ( t, g )._c_.noalias() = s ( t, g )._c_.unaryExpr (
				// 							   std::ptr_fun ( ::tanh ) );
				
				s ( t, g ) ( i, j ) = logistic ( s ( t, g ) ( i, j ) );
				s ( t, g ) ( i , j + N ) = logistic ( s ( t, g ) ( i , j + N ) );
				s ( t, g ) ( i, j + 2 * N ) = logistic ( s ( t, g ) ( i , j + 2 * N ) );
				
				s ( t, g ) ( i , j + 3 * N ) = tanh ( s ( t, g ) ( i, j + 3 * N ) );
				
				
			}
			
			// //3.candidate context
			// s ( t, c ).array() = s ( t, g )._i_.array() *
			// 					 s ( t, g )._c_.array() +
			// 					 s ( t, g )._f_.array() *
			// 					 s ( t - 1, c ).array();
			
			forallblock ( B, N ) {
			
				s ( t, c ) ( i, j ) = s ( t, g ) ( i, j ) * s ( t, g ) ( i , j + 3 * N ) +
									  s ( t, g ) ( i , j + 2 * N ) * s ( t - 1, c ) ( i, j );
									  
									  
			}
			
			
			// //4. nonlinearity : c[t] = tanh(c[t])
			// s ( t, c ).noalias() = s ( t, c ).unaryExpr ( std::ptr_fun (
			// 						   ::tanh ) );
			
			forallblock ( B, N ) {
			
				s ( t, c ) ( i, j ) = tanh ( s ( t, c ) ( i, j ) );
				
			}
			
			
			forallblock ( B, N ) {
			
				s ( t, h ) ( i, j ) = s ( t, g ) ( i , j + N ) * s ( t, c ) ( i, j );
				
			}
			
			// //5. update hidden state : h[t] = o[t] * c[t]
			// s ( t, h ).array() = s ( t, g )._o_.array() *
			// 					 s ( t, c ).array();
			
		}
		
		virtual void backward ( size_t t ) {
		
			// error coming from higher layers: dh[t] = dy[t]
			// g ( t, h ) = g ( t, y );
			
			g ( t, h ).set ( g ( t, y ) );
			
			// // backpropagate through (forward pass step 5) : dc[t] = dh[t] * o[t]
			// g ( t, c ).array() += g ( t, h ).array() *
			// 					  s ( t, g )._o_.array();
			
			forallblock ( B, N ) {
			
				g ( t, c ) ( i, j ) += g ( t, h ) ( i, j ) * s ( t, g ) ( i , j + N );
				
			}
			
			// // propagate through tanh nonlinearity (forward pass step 4)
			// g ( t, c ).array() = g ( t, c ).array() *
			// 					 s ( t, c ).unaryExpr ( std::ptr_fun ( tanh_prime ) ).array();
			
			forallblock ( B, N ) {
			
				g ( t, c ) ( i, j ) = g ( t, c ) ( i, j ) * tanh_prime ( s ( t, c ) ( i, j ) );
				
			}
			
			// // backpropagate through gates (forward pass step 3)
			// g ( t, g )._o_.array() = g ( t, h ).array() *
			// 						 s ( t, c ).array();
			
			// g ( t, g )._i_.array() = g ( t, c ).array() *
			// 						 s ( t, g )._c_.array();
			// g ( t, g )._f_.array() = g ( t, c ).array() * s ( t - 1, c ).array();
			// g ( t, g )._c_.array() = g ( t, c ).array() *
			// 						 s ( t, g )._i_.array();
			
			forallblock ( B, N ) {
			
				g ( t, g ) ( i , j + N ) = g ( t, h ) ( i, j ) * s ( t, c ) ( i, j );
				g ( t, g ) ( i, j ) = g ( t, c ) ( i, j ) * s ( t, g ) ( i, j + 3 * N );
				g ( t, g ) ( i , j + 2 * N ) = g ( t, c ) ( i, j ) * s ( t - 1, c ) ( i, j );
				g ( t, g ) ( i , j + 3 * N ) = g ( t, c ) ( i, j ) * s ( t, g ) ( i, j );
				
			}
			
			// // propagate do, di, df though sigmoids (forward pass step 2)
			// g ( t, g )._iof_.array() =
			// 	g ( t, g )._iof_.array() * s ( t, g )._iof_.unaryExpr (
			// 		std::ptr_fun ( logistic_prime ) ).array();
			
			forallblock ( B, N ) {
			
				g ( t, g ) ( i, j ) = g ( t, g ) ( i, j ) * logistic_prime ( s ( t, g ) ( i, j ) );
				g ( t, g ) ( i , j + N ) = g ( t, g ) ( i , j + N ) * logistic_prime ( s ( t, g ) ( i , j + N ) );
				g ( t, g ) ( i , j + 2 * N ) = g ( t, g ) ( i , j + 2 * N ) * logistic_prime ( s ( t, g ) ( i , j + 2 * N ) );
				g ( t, g ) ( i , j + 3 * N ) = g ( t, g ) ( i , j + 3 * N ) * tanh_prime ( s ( t, g ) ( i , j + 3 * N ) );
				
			}
			
			// // propagate u through tanh (forward pass step 2)
			// g ( t, g )._c_.array() =
			// 	g ( t, g )._c_.array() * s ( t, g )._c_.unaryExpr (
			// 		std::ptr_fun ( tanh_prime ) ).array();
			
			// //backprop through linear part (forward pass step 1)
			// d ( b ) += g ( t, g ).rowwise().sum();
			SUMC ( d ( b ), g ( t, g ) );
			GEMM ( d ( U ), s ( t - 1, h ), g ( t, g ), true, false );
			GEMM ( d ( W ), s ( t, x ), g ( t, g ), true, false );
			
			//backprop into inputs for lower layers
			GEMM ( g ( t, x ), g ( t, g ), p ( W ), false, true );
			
			//carry - h state
			GEMM ( g ( t - 1, y ), g ( t, g ), p ( U ), false , true );
			// //carry - c state
			// g ( t - 1, c ).array() = g ( t, c ).array() *
			// 						 s ( t, g )._f_.array();
			forallblock ( B, N ) {
			
				g ( t - 1, c ) ( i, j ) = g ( t, c ) ( i, j ) * s ( t, g ) ( i , j + 2 * N );
				
			}
			
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
