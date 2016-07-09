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
 * 	@Date:   2016-04-26
 * 	@Last Modified by:   kmrocki
 * 	@Last Modified time: 2016-04-05 18:36:57
 *
 *	Algorithms implementing 'learning'
 *	They are described using class Parameters
 *	where:
 *	p - current weights (parameters)
 *	d - gradients
 *	m - memory/history of past gradients
 *

	TODO: other update types

	+ plain sgd
	+ adam
	+ adadelta
	+ rmsprop

	+ make class 'Trainer' or somthing like that
	+ all algorithms should implement some abstract function 'adapt(p,d,m,alpha)'

	+ add gradient clipping

	http://lasagne.readthedocs.org/en/latest/modules/updates.html

 */

#ifndef __OPTIMIZATION_H__
#define __OPTIMIZATION_H__

#include <parameters.h>
#include <assert.h>
#include <containers/cu_matrix.h>

/* this is pseudo adadelta */
template<typename T>
void adadelta ( Parameters<T> &weights, Parameters<T> &gradients, Parameters<T> &memory, Parameters<T> &updates,
				const dtype learning_rate, const dtype rho ) {
				
	assert ( weights.matrices.size() ==
			 gradients.matrices.size() &&
			 weights.matrices.size() == memory.matrices.size() );
			 
	auto sqrt_eps = [] ( const dtype x, const dtype eps ) { return _sqrt ( x + eps ); };
	
	for ( size_t i = 0; i < weights.matrices.size(); i++ ) {
	
		#ifdef __CUDA_MATRIX__
	
		cu_elementwise_adadelta (	learning_rate, rho,
									weights.matrices[i].cu_data,
									gradients.matrices[i].cu_data,
									memory.matrices[i].cu_data,
									updates.matrices[i].cu_data,
									weights.matrices[i].size() );
									
									
									
		#else
									
		#endif
									
									
	}
	
}

template<typename T>
void adadelta_decay ( Parameters<T> &weights, Parameters<T> &gradients, Parameters<T> &memory, Parameters<T> &updates,
					  const dtype learning_rate, const dtype rho, const dtype decay ) {
					  
	assert ( weights.matrices.size() ==
			 gradients.matrices.size() &&
			 weights.matrices.size() == memory.matrices.size() );
			 
	auto sqrt_eps = [] ( const dtype x, const dtype eps ) { return _sqrt ( x + eps ); };
	
	for ( size_t i = 0; i < weights.matrices.size(); i++ ) {
	
		#ifdef __CUDA_MATRIX__
	
		cu_elementwise_adadelta_decay (	learning_rate, rho,
										weights.matrices[i].cu_data,
										gradients.matrices[i].cu_data,
										memory.matrices[i].cu_data,
										updates.matrices[i].cu_data,
										weights.matrices[i].size(), decay );
										
										
										
		#else
										
		#endif
										
										
	}
	
}
template<typename T>
void adagrad ( Parameters<T> &weights, Parameters<T> &gradients, Parameters<T> &memory,
			   const dtype learning_rate ) {
			   
	assert ( weights.matrices.size() ==
			 gradients.matrices.size() &&
			 weights.matrices.size() == memory.matrices.size() );
			 
	auto sqrt_eps = [] ( const dtype x, const dtype eps ) { return _sqrt ( x + eps ); };
	
	for ( size_t i = 0; i < weights.matrices.size(); i++ ) {
	
		//clip
		// gradients.matrices[i].forall ( [&] ( dtype x ) {
		
		// 	return _max ( _min ( x, 1 ), -1 );
		
		// }, gradients.matrices[i] );
		
		#ifdef __CUDA_MATRIX__
		
		cu_elementwise_adagrad (	learning_rate,
									weights.matrices[i].cu_data,
									gradients.matrices[i].cu_data,
									memory.matrices[i].cu_data,
									weights.matrices[i].size() );
									
									
		/* some old CPU code below, afraid to remove */
		
		//weights.matrices[i].sync_host();
		//memory.matrices[i].sync_host();
		#else
		// elementwise ( [&] ( dtype * p, dtype * d, dtype * m, size_t i ) {
		
		// 	m[i] += d[i] * d[i];
		// 	p[i] -= learning_rate * d[i] / sqrt_eps ( m[i], 1e-10 );
		
		// },
		
		// weights.matrices[i].size(),
		// weights.matrices[i] [0],
		// gradients.matrices[i] [0],
		// memory.matrices[i] [0] );
		#endif
		// //keep history of magnitude
		// memory.matrices[i].forall ( [&] ( dtype x, dtype y ) {
		
		// 	// m += g .* g
		// 	//if ( !std::isnan ( y ) )
		// 	return ( x += y * y );
		// 	//else
		// 	//	return x;
		
		// }, memory.matrices[i], gradients.matrices[i] );
		
		// //update
		// weights.matrices[i].forall ( [&] ( dtype x, dtype y, dtype z, dtype a ) {
		
		// 	// p -= alpha * g./sqrt(m + eps)
		// 	//if ( !std::isnan ( z ) )
		
		// 	return ( x -=  a * y / sqrt_eps ( z, 1e-10 ) );
		// 	//else
		// 	//	return x;
		
		// }, weights.matrices[i], gradients.matrices[i], memory.matrices[i], learning_rate );
		
	}
	
}

#endif /* __OPTIMIZATION_H__ */
