/*
 *
 * Author: Kamil Rocki
 */

#ifndef __STATE_H__
#define __STATE_H__

#include <map>
#include <containers/matrixarray.h>

/* Bare class State is just a MatrixArray + it defines some virtual methods which need to be implemented */
template <typename T>
class State : public MatrixArray<T> {

	public:
	
		/* default constr */
		State<T>() : MatrixArray<T>() {};
		
		/* main constr */
		State<T> ( size_t M, size_t N, size_t B, std::string name,
				   std::initializer_list<std::tuple<std::string, size_t, size_t>>
				   args, std::string id ) :
			MatrixArray<T> ( name, args, id ) {
			
			/* add {x, y} as default states, assuming that any state has some inputs and outputs */
			MatrixArray<T>::add (
			
			{
			
				std::make_tuple ( "x", B, M ), 	// in
				std::make_tuple ( "y", B, N ) 	// out
				
			} );
			
		};
		
		/* copy constr */
		State<T> ( const State<T> &other ) : MatrixArray<T> (
				other ) { }
				
		/* assignment op */
		State<T> &operator= ( const State<T> &other ) {
		
			MatrixArray<T>::operator= ( other );
			return *this;
			
		}
};

#endif /*__STATE_H__*/
