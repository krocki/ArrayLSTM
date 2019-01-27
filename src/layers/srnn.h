/*
 *
 * Author: Kamil Rocki
 *
 * CPU Implementation of simple RNN
 * Technical Report, section 2
 */

#ifndef __RNN_H__
#define __RNN_H__

#include <vector>
#include <parameters.h>
#include <timelayer.h>
#include <state.h>

#define N this->N
#define B this->B

/* define algotithms */
template <typename T>
class sRNN : public Timelayer<T> {

	public:
	
	
		/* main constructor */
		sRNN ( size_t _M, size_t _N, size_t _B,
			   size_t _S ) :
			   
			Timelayer<T> ( _M, _N, _B, _S,
			
		{	"simple RNN"	},
		
		{
			/* define states */
			std::make_tuple ( "h", _B, _N )
			
		}, {
		
			/* define params */
			std::make_tuple ( "W", _M, _N ), // x -> h
			std::make_tuple ( "U", _N, _N ), // h -> h
			std::make_tuple ( "b", 1, _N ) 	// biases
			
		} ) {
		
			/*init*/
			matrix_init ( p ( W ) );
			matrix_init ( p ( U ) );
			
		}
		
		/*  define forward function */
		virtual void forward ( size_t t = 1 ) {
		
			/* linear activation */
			ZERO ( s ( t, h ) );
			
			/* this can be computed in parallel for all time steps */
			GEMM ( s ( t, h ), s ( t, 	x ), p ( W ) );
			GEMM ( s ( t, h ), s ( t - 1, h ), p ( U ) );
			
			/* bias */
			ADDR ( s ( t, h ), p ( b ) );
			
			/* non-linearity */
			TANH ( s ( t, h ) );
			
		}
		
		virtual void backward ( size_t t ) {
		
			//backpropagate through tanh
			DTANH ( g ( t, h ), s ( t, h ), g ( t, y ) );
			
			//backprop through linear part
			SUMC ( d ( b ), g ( t, h ) );
			
			GEMM ( d ( U ), s ( t - 1, h ), g ( t, h ), true, false );
			
			//this can be computed in parallel for all time steps
			GEMM ( d ( W ), s ( t, x ), g ( t, h ), true, false );
			
			//backprop into inputs
			GEMM ( g ( t, x ), g ( t, h ), p ( W ), false, true );
			
			//carry
			GEMM ( g ( t - 1, y ), g ( t, h ), p ( U ), false, true );
			
		}
		
		virtual void reset ( dtype std ) {
		
			randn ( s ( 0, h ), ( dtype ) 0, ( dtype ) std );
			
		}
		
};

#undef N
#undef B

#endif /* __RNN_H__ */
