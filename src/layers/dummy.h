/*
 *
 * Author: Kamil Rocki 
 *
 * 	a dummy layer template
 *	copy this into a new file
 *	and implement missing parts
 *
 */

#ifndef __DUMMY_H__
#define __DUMMY_H__

#include <vector>
#include <matrix.h>
#include <parameters.h>
#include <timelayer.h>
#include <state.h>

/* define algotithms */
class Dummy : public Timelayer {

	public:
	
		/* main constructor */
		Dummy ( size_t _M, size_t _N, size_t _B,
				size_t _S ) :
			Timelayer ( _M, _N, _B, _S,
			
		{	"dummy"		},
		
		{
			/* define states */
			/*std::make_tuple ( "h", _N, _B )*/
			
		}, {
		
			/* define params */
			/*std::make_tuple ( "W", 2 * _N, _M )*/
			
		} ) {
		
			/*init*/
			/*matrix_init ( p["W"] );*/
			
		}
		
		virtual void forward ( size_t t = 1 ) {
		
			/* do something */
			
		}
		
		virtual void backward ( size_t t ) {
		
			/* do something */
			
		}
		
		/* optional */
		/* copy constr */
		// Dummy( const Dummy& src) : Timelayer(src.M, src.N, src.B, src.S)  { }
		
		/* assignent operator */
		// Dummy& operator=(const Dummy& src) { Timelayer::operator=(src); return *this; }
		
};

#endif /* __DUMMY_H__ */
