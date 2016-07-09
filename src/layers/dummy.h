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
 *
 * 	@Author: kmrocki
 * 	@Date:   2016-02-19 15:25:43
 * 	@Last Modified by:   kmrocki
 * 	@Last Modified time: 2016-03-24 18:36:57
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