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
* 	Dynamic Matrix
*	It's just what it looks like - std::vector<Matrix>;
*	elements are are accessible by name with operator []
*	name needs to be given together with name in the
*	main constructor
*	example:
*
*	m = MatrixArray(

		{	"something"	}, {

			std::make_tuple("W", M, N),
			std::make_tuple("U", X, Y),

		}
	)

	will make a 2 sub-matrices "W" and "U" of dimensions M x N and X x Y
	respectively;

	later, m['W'] will return the first matrix and m['U'] the second one
	other things are just implementations of operators and IO
*/

#ifndef __MATRIXARRAY_H__
#define __MATRIXARRAY_H__

template <typename T>
class MatrixArray {

	public:
	
		std::vector<T> matrices;
		std::string name;
		std::map<std::string, size_t> namemap;
		
		MatrixArray<T>() = default;
		
		/* the main constructor */
		MatrixArray<T> ( std::string _name,
						 std::initializer_list<std::tuple<std::string, size_t, size_t>>
						 args, std::string id ) : name ( _name + " " + id ) {
						 
			add ( args );
			
		}
		
		void add (
			std::initializer_list<std::tuple<std::string, size_t, size_t>>
			args ) {
			
			for ( auto i : args ) {
			
				namemap[std::get<0> ( i )] = matrices.size();
				matrices.push_back ( T ( std::get<1> ( i ),
										 std::get<2> ( i ) ) );
										 
				matrices.back().setZero();
				
			}
			
		}
		
		MatrixArray<T> ( const MatrixArray<T> &other ) {
		
			namemap = other.namemap;
			name = other.name;
			matrices = other.matrices;
			
		}
		
		MatrixArray<T> &operator= ( const MatrixArray<T> &other ) {
		
			namemap = other.namemap;
			name = other.name;
			matrices = other.matrices;
			
			return *this;
			
		}
		
		template <typename otherType>
		MatrixArray<T> &operator= ( const MatrixArray<otherType> &other ) {
		
			namemap = other.namemap;
			name = other.name;
			
			for ( size_t i = 0; i < matrices.size(); i++ )
				matrices[i] = other.matrices[i];
				
			return *this;
			
		}
		
		T &operator[] ( char key ) {
		
			return ( *this ) [std::string ( 1, key )];
			
		}
		
		T &operator[] ( std::string key ) {
		
			/*			if ( namemap.find ( key ) == namemap.end() )
							std::cout << "Warning !!! " << name <<
									  "::[] - key not found:" << key << std::endl;
									 */
			
			return matrices[namemap[key]];
			
		}
		
		void zero() {
		
			for ( size_t i = 0; i < matrices.size(); i++ )
				matrices[i].setZero();
				
		}
		
		void cu_zero() {
		
			for ( size_t i = 0; i < matrices.size(); i++ )
				matrices[i].cu_zero();
				
		}
		
		void sync_host() {
		
			for ( size_t i = 0; i < matrices.size(); i++ )
				matrices[i].sync_host();
		}
		
		template<class Archive>
		void serialize ( Archive &archive ) {
		
			archive ( name );
			archive ( namemap );
			archive ( matrices );
			
		}
		
};

#endif /*__MATRIXARRAY_H__*/
