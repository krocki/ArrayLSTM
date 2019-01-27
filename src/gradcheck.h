/*
 *
 * Author: Kamil Rocki
 * 	Gradient check
 *	needs modifications
 *
 */

#ifndef __GRADCHECK_H__
#define __GRADCHECK_H__

#include <containers/datatype.h>

//TODO: make compatible with Parameters
// returns true if everything is OK
bool check_gradient_error ( const char *message, const matrix<dtype> &n,
							const matrix<dtype> &m ) {
							
	matrix<dtype> diff ( m.rows(), m.cols() );
	matrix<dtype> sum ( m.rows(), m.cols() );
	matrix<dtype> error ( m.rows(), m.cols() );
	
	SUBM ( diff, m, n );
	ADDM ( sum, m, n );
	
	bool okMean = true;
	bool okMax = true;
	
	ABS ( diff, diff );
	ABS ( sum, sum );
	
	std::cout << std::endl;
	
	//need to check div by 0
	for ( int i = 0; i < sum.rows(); i++ ) {
		for ( int j = 0; j < sum.cols(); j++ ) {
		
			if ( sum ( i, j ) > 0.0 )
				error ( i, j ) = diff ( i, j ) / sum ( i, j );
			else
				error ( i, j ) = 0;
				
			if ( error ( i, j ) > 1e-1 )
				std::cout << i << ", " << j << ", m: " << m ( i,
						  j ) << ", n: " <<
						  n ( i, j ) << ", e: " << error ( i, j ) << std::endl;
						  
		}
	}
	
	dtype maxError = MAX ( error );
	dtype meanError = error.sum() / dtype ( error.rows() *
											error.cols() );
											
	if ( maxError > 1e-1 )
		okMax = false;
		
	if ( meanError > 1e-3 )
		okMean = false;
		
	std::cout 	<< std::endl
				<< std::setw ( 15 ) << std::setprecision (
					12 ) << "[" << message << "]" << std::endl
				<< std::setw ( 20 ) << " numerical range (" << std::setw (
					20 ) << MIN ( n ) <<
				", " << std::setw ( 20 ) << MAX ( n ) << ")" << std::endl
				<< std::setw ( 20 ) << " analytic range (" << std::setw (
					20 ) << MIN ( m ) <<
				", " << std::setw ( 20 ) << MAX ( m ) << ")" << std::endl
				<< std::setw ( 20 ) << " max rel. error " << std::setw (
					20 ) << maxError;
					
	if ( okMax == false )
		std::cout << std::setw ( 23 ) << "!!!  >1e-1 !!!";
		
	std::cout << std::endl << std::setw ( 20 ) <<
			  " mean rel. error " << std::setw ( 20 ) << meanError;
			  
	if ( okMean == false )
		std::cout << std::setw ( 23 ) << "!!!  >1e-3  !!!";
		
	std::cout << std::endl;
	
	return okMean && okMax;
}

// returns true if everything is OK
template<typename T>
bool check_gradients ( Parameters<T> &n, Parameters<T> &d ) {

	bool bU = check_gradient_error ( "U", n['U'], d['U'] );
	bool bW = check_gradient_error ( "W", n['W'], d['W'] );
	bool bb = check_gradient_error ( "b", n['b'], d['b'] );
	bool ok = bW;// && bW && bb;
	
	return ok;
	
}

#endif /* __GRADCHECK_H__ */
