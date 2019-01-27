/*
 *
 * Author: Kamil Rocki
 * 	Implementation of deep RNN
 *
 *  TODO: clean up, serialization code
 */

#ifndef __DEEPLSTM_H_
#define __DEEPLSTM_H_

#include <timelayer.h>
#include <containers/datatype.h>

/* different layer types */

#ifdef __USE_CUDA__
	
	#include <layers/lstm_cuda.h>
	//#include <layers/hlstm.h>
	#include <layers/cu_softmax.h>
	//#include <layers/splstm.h>
	//#include <layers/clstm.h>
	//#include <layers/hmlstm.h>
	//#include <layers/alstm.h>
	//#include <layers/dolstm.h>
	//#include <layers/attLSTM.h>
	//#include <layers/hardattLSTM.h>
	
#else
	
	/*cpu code*/
	#include <layers/lstm_devel.h>
	//#include <layers/srnn.h>
	#include <layers/softmax.h>
	
#endif

/*TODO : remove. unsafe */
#undef p
#undef d
#undef s
#undef g

#include <optimization.h>
#include <gradcheck.h>

#include <algorithm>

template <typename MatrixType>
class DeepLSTM {

	public:
	
		Timelayer<MatrixType> *outputlayer;
		
		std::vector<Timelayer<MatrixType> *> layers;
		
		DeepLSTM ( size_t _M, size_t _N, size_t _B, size_t _S, size_t _D,
				   std::initializer_list<Timelayer<MatrixType> *> args = {} ) :
			M ( _M ), N ( _N ), B ( _B ), S ( _S ), D ( _D ) {
			
			for ( auto i : args ) {
			
				// TODO:
				
			}
			
			//TODO: remove layer declaration from here
			
			/*			layers.push_back ( new aLSTM<MatrixType> ( _M, _N, _B, _S, 2 ) );
			
						for ( size_t d = 1; d < D; d++ )
							layers.push_back ( new aLSTM<MatrixType> ( _N, _N, _B, _S, 2 ) );
			*/
			
			//D LSTM layers
			layers.push_back ( new LSTM<MatrixType> ( _M, _N, _B, _S ) );
			
			for ( size_t d = 1; d < D; d++ )
				layers.push_back ( new LSTM<MatrixType> ( _N, _N, _B, _S ) );
				
			//+ 1 softmax layer
			layers.push_back ( new Softmax<MatrixType> ( _N, _M, _B, _S ) );
			
			//pointer to the output layer
			outputlayer = layers[D];
			
			//temp storage
			surprisals.resize ( B, M );
			
		}
		
		~DeepLSTM() {
		
			for ( size_t i = 0; i < layers.size(); i++ )
				delete ( layers[i] );
				
			layers.clear();
			
		}
		
		template<class Archive>
		void serialize ( Archive &archive ) {
		
			archive ( M, N, B, S, D );
			
			for ( size_t d = 0; d <= D; d++ )
				archive ( *layers[d] );
				
				
			// archive(dx);
			// archive(surprisals);
			
		}
		
		void sync_grads_device() {
		
			for ( size_t d = 0; d <= D; d++ ) {
			
				for ( size_t w = 0; w < layers[d]->p.matrices.size(); w++ )
				
					layers[d]->d.matrices[w].sync_device();
					
			}
			
		}
		
		void sync_grads_host() {
		
			for ( size_t d = 0; d <= D; d++ ) {
			
				for ( size_t w = 0; w < layers[d]->p.matrices.size(); w++ )
				
					layers[d]->d.matrices[w].sync_host();
					
			}
			
		}
		
		void sync_all_host() {
		
			for ( size_t d = 0; d <= D; d++ )
				layers[d]->sync_all_host();
				
			for ( size_t t = 0; t < S; t++ )
				dx[t].sync_host();
				
			surprisals.sync_host();
			
		}
		
		void sync_params() {
		
			for ( size_t d = 0; d <= D; d++ ) {
			
				for ( size_t w = 0; w < layers[d]->p.matrices.size(); w++ )
				
					layers[d]->p.matrices[w].sync_device();
					
			}
			
		}
		
		void sync_params_host() {
		
			for ( size_t d = 0; d <= D; d++ ) {
			
				for ( size_t w = 0; w < layers[d]->p.matrices.size(); w++ )
				
					layers[d]->p.matrices[w].sync_host();
					
			}
			
		}
		
		void sync_memory() {
		
			for ( size_t d = 0; d <= D; d++ ) {
			
				for ( size_t w = 0; w < layers[d]->m.matrices.size(); w++ )
				
					layers[d]->m.matrices[w].sync_device();
					
			}
			
		}
		
		void forward ( bool apply_dropout, std::vector<MatrixType> &x ) {
		
			layers[0]->forward ( apply_dropout, x );
			
			for ( size_t d = 1; d <= D; d++ )
			
				layers[d]->forward ( apply_dropout, layers[d - 1]->s, 'h' );
				
		}
		
		void forward ( bool apply_dropout, MatrixType &x, size_t t = 1 ) {
		
			layers[0]->s[t]['x'] = x ;
			layers[0]->forward ( apply_dropout, t );
			
			for ( size_t d = 1; d <= D; d++ ) {
			
				layers[d]->s[t]['x'] = layers[d - 1]->s[t]['h'];
				layers[d]->forward ( apply_dropout, t );
				
				
			}
			
		}
		
		void backward ( bool apply_dropout, std::vector<MatrixType> &target ) {
		
			dx = target;
			
			for ( size_t d = D; d > 0; d-- ) {
			
				layers[d]->backward ( apply_dropout, dx );
				
				for ( size_t t = 0; t < S; t++ )
					dx[t] = layers[d]->g[t]['x'];
					
			}
			
			layers[0]->backward ( apply_dropout, dx );
			
		}
		
		dtype cu_loss ( std::vector<MatrixType> &target, size_t symbols, bool bits = false ) {
		
			dtype loss = 0.0;
			
			for ( size_t t = S - symbols; t < S;
					t++ ) { // compute activations for sequence
					
				// cross-entropy loss, sum logs of probabilities of target outputs
				loss += cu_cross_entropy_loss ( outputlayer->s[t]['p'], target[t] );
				
			}
			
			return loss;
			
		}
		
		dtype loss ( std::vector<MatrixType> &target, size_t symbols, bool bits = false ) {
		
			dtype loss = 0.0;
			
			for ( size_t t = S - symbols; t < S;
					t++ ) { // compute activations for sequence
					
				if ( bits ) {
				
					for ( size_t k = 0; k < surprisals.size(); k++ )
						surprisals ( k ) = -_log2 ( outputlayer->s[t]['p'] ( k ) ) * target[t] ( k );
						
				}
				
				else {
				
					for ( size_t k = 0; k < surprisals.size(); k++ )
						surprisals ( k ) = -_log ( outputlayer->s[t]['p'] ( k ) ) * target[t] ( k );
				}
				
				// cross-entropy loss, sum logs of probabilities of target outputs
				loss += surprisals.sum();
				
			}
			
			return loss;
			
		}
		
		
		void adapt ( dtype learning_rate, dtype rho = 0.95 ) {
		
			/* adjust params in all layers */
			for ( size_t d = 0; d <= D; d++ )
				adadelta ( layers[d]->p, layers[d]->d, layers[d]->m, layers[d]->u,
						   learning_rate, rho );
						   
			// adagrad ( layers[d]->p, layers[d]->d, layers[d]->m,
			// 		  learning_rate );
			
		}
		
		std::vector<char> sample ( size_t characters_to_generate, MatrixType &codes, std::string seed = " ",
								   dtype reset_std = 0.0 ) {
								   
			std::vector<char> generated_text;
			
			//make a copy
			DeepLSTM<MatrixType> testnet ( M, N, 1, S, D );
			sync_params_host();
			testnet.loadParams<MatrixType> ( *this );
			testnet.sync_params();
			
			testnet.resetContext ( reset_std );
			MatrixType probs ( 1, M );
			MatrixType cdf ( 1, M );
			cdf.setZero();
			
			std::random_device rd;
			std::mt19937 gen ( rd() );
			std::uniform_real_distribution<> dis ( 0, 1 );
			
			size_t index;
			
			MatrixType x ( 1, M );
			
			for ( size_t ii = 0; ii < characters_to_generate - 1;
					ii++ ) {
					
				size_t ev_x;
				
				if ( ii < seed.length() )
				
					ev_x = ( size_t ) seed[ii];
					
				else
				
					ev_x = index;
					
					
				generated_text.push_back ( ( char ) ev_x );
				
				row ( x, codes, ev_x );
				
				testnet.forward ( false, x );
				testnet.carryContext ( 1 );
				probs = testnet.outputlayer->s[1]['p'];
				
				dtype sum = probs.sum();
				
				for ( size_t k = 0; k < probs.size(); k++ )
					probs ( k ) = probs ( k ) / sum;
					
				//cumsum, TODO: something nicer
				cdf ( 0 ) = probs ( 0 );
				
				for ( size_t ii = 1; ii < probs.size(); ii++ )
					cdf ( ii ) = cdf ( ii - 1 ) + probs ( ii );
					
				dtype r = dis ( gen );
				
				// find the lowest number in cdf that's larger or equal to r
				
				for ( size_t ii = 0; ii < cdf.size(); ii++ ) {
				
					if ( r < cdf ( ii ) ) {
					
						index = ii;
						break;
					}
					
				}
				
			}
			
			return generated_text;
			
		}
		
		dtype test ( MatrixXi &test, size_t seq_length, MatrixType &codes, dtype reset_std = 0.0, bool bits = false ) {
		
			dtype error = 0;
			size_t trials = 100;
			size_t length = seq_length;
			
			size_t test_length = test.rows();
			
			//make a copy
			DeepLSTM<MatrixType> testnet ( M, N, 1, 2, D );
			sync_params_host();
			testnet.loadParams<MatrixType> ( *this );
			testnet.sync_params();
			
			for ( size_t k = 0; k < trials; k++ ) {
			
				testnet.resetContext ( reset_std );
				
				MatrixType probs ( 1, M );
				MatrixType x ( 1, M );
				
				size_t pos = rand() % ( test.size() - length - 2 );
				
				dtype partial_error = 0;
				
				for ( size_t ii = pos; ii < pos + length; ii++ ) {
				
					size_t ev_x = ( ( int * ) test.data() ) [ii];
					size_t ev_t = ( ( int * ) test.data() ) [ii + 1];
					
					row ( x, codes, ev_x );
					
					testnet.forward ( false, x );
					testnet.carryContext ( 1 );
					probs = testnet.outputlayer->s[1]['p'];
					
					if ( bits )
						partial_error += -log2 ( probs ( ev_t ) );
					else
						partial_error += -log ( probs ( ev_t ) );
						
					std::cout << std::setprecision ( 2 ) << std::setw (
								  15 ) << "Testing... " <<
							  100.0f * ( float ) ( k ) / ( float ) ( trials ) <<
							  "%\r" <<  std::flush;
				}
				
				partial_error /= length;
				
				error += partial_error;
			}
			
			return error / trials;
			
		}
		
		/*
			- sample size
			- min
			- mean
			- max
			- std err
		
		*/
		
		std::tuple<size_t, dtype, dtype, dtype, dtype>
		test_batch ( MatrixXi &test, size_t seq_length, MatrixType &codes, dtype reset_std = 0.0,
					 bool bits = false ) {
					 
			dtype error = 0;
			size_t trials = 5;
			size_t length = seq_length;
			
			size_t test_length = test.rows();
			
			size_t __B = B;
			
			std::vector<dtype> datapoints;
			
			//make a copy
			DeepLSTM<MatrixType> testnet ( M, N, __B, 2, D );
			sync_params_host();
			testnet.loadParams<MatrixType> ( *this );
			testnet.sync_params();
			
			size_t ev_x[__B];
			size_t ev_t[__B];
			size_t pos[__B];
			
			for ( size_t k = 0; k < trials; k++ ) {
			
				testnet.resetContext ( reset_std );
				
				MatrixType probs ( __B, M );
				MatrixType x ( __B, M );
				
				for ( size_t b = 0; b < __B; b++ )
					pos[b] = rand() % ( test.size() - length - 2 );
					
				dtype partial_error = 0;
				
				for ( size_t ii = 0; ii < length; ii++ ) {
				
					x.setZero();
					
					for ( size_t b = 0; b < __B; b++ ) {
					
						ev_x[b] = ( ( int * ) test.data() ) [pos[b] + ii];
						ev_t[b] = ( ( int * ) test.data() ) [pos[b] + ii + 1];
						
						set_row_one_hot ( x, b, ev_x[b] );
						
					}
					
					testnet.forward ( false, x );
					testnet.carryContext ( 1 );
					
					probs = testnet.outputlayer->s[1]['p'];
					
					for ( size_t b = 0; b < __B; b++ ) {
					
						float b_error;
						
						if ( bits )
							b_error = -log2 ( probs ( b, ev_t[b] ) );
						else
							b_error = -log ( probs ( b, ev_t[b] ) );
							
						if ( !std::isnan ( b_error ) && !std::isinf ( b_error ) )
							partial_error += b_error;
						else
						
							partial_error += 8;
							
					}
					
					std::cout << std::setprecision ( 3 ) << std::setw (
								  15 ) << "Testing... " <<
							  100.0f * ( float ) ( k ) / ( float ) ( trials ) <<
							  "%\r" <<  std::flush;
				}
				
				partial_error /= ( length * B );
				
				error += partial_error;
				
				datapoints.push_back ( partial_error );
			}
			
			double sum = std::accumulate ( datapoints.begin(), datapoints.end(), 0.0 );
			double mean = sum / datapoints.size();
			
			std::vector<double> diff ( datapoints.size() );
			std::transform ( datapoints.begin(), datapoints.end(), diff.begin(), [mean] ( double x ) { return x - mean; } );
			double sq_sum = std::inner_product ( diff.begin(), diff.end(), diff.begin(), 0.0 );
			//std::cout << sq_sum << std::endl;
			double stdev = std::sqrt ( sq_sum / ( datapoints.size() - 1 ) );
			//std::cout << stdev << std::endl;
			std::vector<dtype>::iterator minval = std::min_element ( std::begin ( datapoints ), std::end ( datapoints ) );
			std::vector<dtype>::iterator maxval = std::max_element ( std::begin ( datapoints ), std::end ( datapoints ) );
			
			//std::cout << "SAMPLES = " << datapoints.size() << ", MIN = " << *minval << ", MAX = " << *maxval << ", MEAN = " << mean
			//		  << " +/- " << stdev / sqrt ( datapoints.size() ) << std::endl;
			
			return  std::make_tuple ( datapoints.size(), *minval, mean, *maxval, stdev / sqrt ( datapoints.size() ) );
			//return error / trials;
			
		}
		
		void resetContext ( dtype std, size_t T = 0 ) {
		
			for ( size_t d = 0; d < D; d++ )
				layers[d]->reset ( std );
				
		}
		
		void carryContext ( size_t T ) {
		
			for ( size_t d = 0; d < D; d++ )
				layers[d]->s[0] = layers[d]->s[T];
				
		}
		
		/* TODO: change to copy constr */
		template<typename otherType>
		void loadParams ( const DeepLSTM<otherType> &src ) {
		
			for ( size_t d = 0; d < D; d++ )
				layers[d]->p = src.layers[d]->p;
				
			outputlayer->p = src.outputlayer->p;
			
		}
		
		/* all these things should be moved somewhere */
		/* possibly to gradheck.h */
		
		void compute_all_numerical_grads ( std::vector<MatrixType> &x, std::vector<MatrixType> &target ) {
		
			layers[0]->n = layers[0]->d;
			
			sync_params_host();
			
			std::cout << std::endl << std::endl << std::setw (
						  9 ) << "b" << std::endl;
			numerical_grads ( layers[0]->n['b'], layers[0]->p['b'],
							  layers[0]->p, x, target );
			std::cout << std::endl << std::endl << std::setw (
						  9 ) << "U" << std::endl;
			numerical_grads ( layers[0]->n['U'], layers[0]->p['U'],
							  layers[0]->p, x, target );
			std::cout << std::endl << std::endl << std::setw (
						  9 ) << "W" << std::endl;
			numerical_grads ( layers[0]->n['W'], layers[0]->p['W'],
							  layers[0]->p, x, target );
							  
		}
		
		void numerical_grads ( MatrixType &n, MatrixType &_p, Parameters<MatrixType> &P, std::vector<MatrixType> &x,
							   std::vector<MatrixType> &target ) {
							   
			dtype delta = 1e-5;
			size_t grads_checked = 0;
			size_t grads_to_check = 25;
			//check only a fraction
			dtype percentage_to_check = ( dtype ) grads_to_check /
										( dtype ) ( _p.cols() * _p.rows() );
			std::random_device rd;
			std::mt19937 gen ( rd() );
			std::uniform_real_distribution<> dis ( 0, 1 );
			
			for ( size_t i = 0; i < n.rows(); i++ ) {
				for ( size_t j = 0; j < n.cols(); j++ ) {
				
					dtype r = dis ( gen );
					
					if ( r >= percentage_to_check )
						continue;
						
					else {
					
						dtype minus_loss, plus_loss;
						dtype original_value = _p ( i, j );
						
						_p ( i, j ) = original_value - delta;
						sync_params();
						forward ( false, x );
						minus_loss = loss ( target, S - 1 );
						
						_p ( i, j ) = original_value + delta;
						sync_params();
						forward ( false, x );
						plus_loss = loss ( target, S - 1 );
						
						dtype grad = ( plus_loss - minus_loss ) / ( delta * 2 );
						
						n ( i, j ) = grad;
						_p ( i, j ) = original_value;
						
						sync_params();
						grads_checked++;
						
						std::cout << std::setw ( 9 ) << i << std::setw (
									  9 ) << j << std::setw ( 9 ) << grads_checked << "\r" <<
								  std::flush;
					}
				}
			}
			
		}
		
		void gradcheck() {
		
			check_gradients ( layers[0]->n, layers[0]->d );
			
		}
		
		const size_t M, N, B, S, D;
		
		//temp var using backward pass
		std::vector<MatrixType> dx;
		
		//temporary storage
		MatrixType surprisals;
};


#endif
