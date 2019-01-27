/*
 *
 * Author: Kamil Rocki
 * Softmax Layer (cpu version)
 */

#ifndef __SOFTMAX_H__
#define __SOFTMAX_H__

#include <vector>
#include <parameters.h>
#include <timelayer.h>
#include <state.h>

template <typename T>
class Softmax : public Timelayer<T> {

	public:
	
		//temp storage for probability sums
		T sums;
		
		/* main constructor */
		Softmax ( size_t _in, size_t _out, size_t _B, size_t _S ) :
			Timelayer<T> ( _in, _out, _B, _S,
			
		{	"softmax"		},
		
		{
			/* define states */
			std::make_tuple ( "p", _B, _out )
			
		}, {
		
			/* define params */
			std::make_tuple ( "W", _in, _out ),
			std::make_tuple ( "b", 1, _out )
			
		} ) {
		
			/*init*/
			matrix_init ( p ( W ) );
			sums = T ( _B, 1 );
			
		}
		
		virtual void forward ( size_t t = 1 ) {
		
			// outputs
			ZERO ( s ( t, p ) );
			GEMM ( s ( t, p ), s ( t, x ), p ( W ) );
			ADDR ( s ( t, p ), p ( b ) );
			
			// for numerical stability
			SUB ( s ( t, p ), MAX ( s ( t, p ) ) );
			
			// softmax - normalize p
			EXP ( s ( t, p ) );
			
			ZERO ( sums );
			SUMR ( sums, s ( t, p ) );
			
			DIVC ( s ( t, p ), sums );
			
		}
		
		virtual void backward ( size_t t ) {
		
			// prediction error
			SUBM ( g ( t, p ), s ( t, p ), g ( t, y ) );
			
			// propagate through linear layer h->y
			GEMM ( d ( W ), s ( t, x ), g ( t, p ), true, false );
			
			SUMC ( d ( b ), g ( t, p ) );
			
			// propagate through linear layer x->h
			GEMM ( g ( t, x ), g ( t, p ), p ( W ), false, true );
			
		}
		
		virtual void reset ( dtype std ) {};
		
		/* optional */
		/* copy constr */
		// Softmax( const Softmax& src) : Timelayer(src.M, src.N, src.B, src.S) { }
		
		/* assignent operator */
		// Softmax& operator=( const Softmax& src) { Timelayer::operator=(src); return *this; }
		
};

#endif
