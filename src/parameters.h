/*
 *
 * Author: Kamil Rocki
 *	Similar abstract class State
 */

#ifndef __PARAMETERS_H__
#define __PARAMETERS_H__

#include <map>
#include <containers/matrixarray.h>

template <typename T>
class Parameters : public MatrixArray<T> {

	public:
	
		/* default constr */
		Parameters<T>() : MatrixArray<T>() {};
		
		/* main constr */
		Parameters<T> ( std::string name,
						std::initializer_list<std::tuple<std::string, size_t, size_t>>
						args, std::string id ) :
			MatrixArray<T> ( name, args, id ) { };
			
		/* copy constr */
		Parameters<T> ( const Parameters<T> &other ) : MatrixArray<T> (
				other ) { N = other.N; }
				
		/* assignment op */
		Parameters<T> &operator= ( const Parameters<T> &other ) {
		
			N = other.N;
			MatrixArray<T>::operator= ( other );
			return *this;
			
		}
		
		template <typename otherType>
		Parameters<T> &operator= ( const Parameters<otherType> &other ) {
		
			N = other.N;
			MatrixArray<T>::operator= ( other );
			return *this;
			
		}
		
		size_t N;
		
};

#endif /*__PARAMETERS_H__*/
