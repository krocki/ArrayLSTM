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
 * 	@Date:   2016-04-05
 * 	@Last Modified by:   kmrocki
 * 	@Last Modified time: 2016-04-05 18:36:57
 *
 * 	Abstract parameter class, provides interface for algorithms
 *	implementing learning (optimization.h)
 *
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
