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
 * run like this
 *
 * ./deeplstm N B S GPU
 *
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <timer.h>

// for now, define matrix type here

#include <containers/datatype.h>

#include <deeplstm.h>
#include <utils.h>

#include <containers/io.h>
#include <serialization.h>

#include <cuda.h>

int main ( int argc, char *argv[] ) {

	//openblas_set_num_threads ( 1 );
	
	printf ( "argc = %d\n", argc );
	
	for ( int i = 0; i < argc; ++i )
		printf ( "argv[ %d ] = %s\n", i, argv[ i ] );
		
	assert ( argc >= 5 );
	
	// hidden size
	const size_t    N               = atoi ( argv[1] );
	// vocab size (# of distinct observable events)
	const size_t    M               = 256;
	// sequence length for learning
	const size_t    S               = atoi ( argv[2] );
	// batch size
	const size_t    B               = atoi ( argv[3] );
	// depth
	const size_t    D               = 1;
	
	const int 	gpu_number 	= atoi ( argv[4] );
	
	bool dropout = true;
	
	cudaSetDevice ( gpu_number );
	
	//MatrixXi 	_data           = rawread ( "data/large/enwiki-20160601-pages-articles.xml" );
	
	/* TODO: move to command line */
	MatrixXi 	_data           = rawread ( "data/enwik8.txt" );
	std::string out_filename    	= "enwik8_lstm_0703_N" + std::to_string ( N ) +
									  "_S" + std::to_string ( S ) +
									  "_B" + std::to_string ( B ) ;
									  
	const size_t    epoch_length    = 10000;
	const size_t    test_every      = 10;
	const dtype     learning_rate   = 1e-3 * 1;
	const size_t    epochs          = 1000000;
	const dtype     loss_dampening  = 0.999;
	const dtype     reset_std       = 0.0;
	
	// if true - loss in bits (lg2)
	// if false - loss in nats (ln)
	
	const bool      loss_in_bits    = true;
	const float     train_percent   = 90.0f;
	const float     valid_percent   = 5.0f;
	
	size_t serialize_every = 8; // * test_every
	size_t serialize_counter = 0;
	
	double test_loss_dampening = 0.0;
	
	#ifdef __USE_CLBLAS__
	init_clblas ();
	#endif
	
	#ifdef __USE_CUDA__
	init_cublas ( gpu_number );
	init_curand ( );
	#endif
	
	std::cout << "N = " << N << ", S = " << S << ", B = " <<
			  B << ", D = " << D << ", e = " <<
			  epoch_length << ", alpha: " << learning_rate << ", GPU: " << gpu_number <<
			  ", fname: " << out_filename << std::endl;
			  
	//select 'train_percent' % of data as training data
	
	size_t percent_size = _data.size() / 100;
	
	// data used for training
	MatrixXi data, test, valid;
	
	data.block ( _data, 0, 0, train_percent * percent_size, 1 );
	
	// data used for validation
	valid.block ( _data, ( train_percent ) * percent_size, 0,
				  valid_percent * percent_size, 1 );
				  
	// data used for testing
	test.block ( _data, ( train_percent + valid_percent ) * percent_size, 0,
				 _data.size() - ( train_percent + valid_percent ) * percent_size, 1 );
				 
	//std::cout << test << std::endl;
	
	// DEBUG code
	std::cout   << "Train set size: "     << data.size()                << ", "
				<< "Test set size: "  	<< test.size()                << ", "
				<< "Total: "          << data.size() + test.size()  << std::endl;
				
	assert ( data.size() > 0 );
	
	//define the net
	/* TODO: get rid of ( M, N, B, S, D and leave only {} */
	DeepLSTM<MatrixType> deeplstm ( M, N, B, S, D, {
	
		new LSTM<MatrixType>	( 100, 100, 1, 5 ),
		new Softmax<MatrixType> ( 100, 100, 1, 5 )
		
	} );
	
	// this is an identity matrix that
	// is used to encode inputs, 1 of K encoding
	// (MATLAB's eye())
	MatrixType codes ( M, M );
	eye ( codes );
	
	// temp matrix for storing input
	std::vector<MatrixType> x ( S );
	// targets - desired outputs
	std::vector<MatrixType> target ( S );
	
	//zero them
	for ( size_t t = 0; t < S; t++ ) {
	
		target[t].resize ( B, M );
		x[t].resize ( B, M );
		
	}
	
	Timer epoch_timer, flops_timer, test_timer, main_timer;
	
	size_t length = data.rows();
	size_t positions[B];
	dtype loss, epoch_loss;
	Matrix results;
	size_t results_size = 0;
	
	// some approximation on the number of FlOPs
	dtype flops_per_iteration = count_flops ( M, N, S, B ) * D;
	
	dtype flops_per_epoch = flops_per_iteration *
							( length - S );
	dtype gflops_per_sec = 0;
	
	test_timer.start();
	main_timer.start();
	
	unsigned long iterations = 0L;
	dtype smooth_loss = -1;
	dtype train_error = -1;
	dtype test_error = -1;
	
	deeplstm.sync_memory();
	deeplstm.sync_params();
	deeplstm.sync_grads_device();
	
	for ( size_t e = 0; e < epochs; e++ ) {
	
		//initial positions
		
		for ( size_t b = 0; b < B; b++ )
		
			positions[b] = rand() % ( length - epoch_length - 1 - S );
			
			
		deeplstm.resetContext ( reset_std );
		
		epoch_timer.start();
		flops_timer.start();
		
		for ( size_t i = 0; i < epoch_length; i += S ) {
		
			loss = 0;
			
			dtype test_time = test_timer.end();
			dtype time_since_start = main_timer.end();
			
			if ( ( i % 1 ) == 0 ) {
			
				dtype flops_time = flops_timer.end();
				
				size_t eta_sec = ( flops_time * ( ( float ) epoch_length -
												  ( float ) i ) ) / 100.0f;
												  
				float epochs_left = epochs - ( e + ( float ) ( i + 1 ) /
											   ( float ) epoch_length );
				float time_per_epoch = time_since_start / ( e + ( float ) (
										   i + 1 ) / ( float ) epoch_length );
										   
				float total_eta = epochs_left * time_per_epoch;
				tm_t eta = seconds2time ( ( size_t ) total_eta );
				
				size_t test_eta = test_every - ( size_t ) test_time;
				
				gflops_per_sec = ( 1.0f * flops_per_iteration / powf ( 2,
								   30 ) ) / flops_time;
								   
				PRINT_INFO();
				
				flops_timer.start();
			}
			
			for ( size_t t = 0; t < S; t++ ) {
			
				target[t].setZero();
				x[t].setZero();
				
			}
			
			for ( size_t b = 0; b < B; b++ ) {
			
				size_t event = ( ( int * )
								 data.data() ) [positions[b]];      // current observation, uchar (0-255)
								 
				for ( size_t t = 0; t < S; t++ ) {
				
					size_t ev_x = ( ( int * ) data.data() ) [positions[b] + t];
					size_t ev_t = ( ( int * ) data.data() ) [positions[b] + t +
								  1];
								  
					set_row_one_hot ( target[t], b, ev_t );
					set_row_one_hot ( x[t], b, ev_x );
					
				}
				
				positions[b] += S;
				
			}
			
			// std::cout << bytes_allocated_total << std::endl;
			iterations++;
			
			deeplstm.forward ( dropout, x );
			
			loss = deeplstm.loss ( target, 1, loss_in_bits );
			//loss = deeplstm.cu_loss (target, 1, loss_in_bits );
			
			if ( !std::isnan ( loss ) && !std::isinf ( loss ) )
				smooth_loss = smooth_loss < 0 ? loss / B : smooth_loss *
							  loss_dampening + ( 1 - loss_dampening ) * loss /
							  ( B ); // loss/char
							  
							  
			deeplstm.backward ( dropout, target );
			
			if ( test_time > test_every ) {
			
				serialize_counter++;
				
				#ifdef __PRECISE_MATH__
				
				std::cout << std::endl << "*** Checking gradients... " <<
						  std::endl << std::endl;
				//deeplstm.sync_params_host();
				deeplstm.sync_grads_host();
				deeplstm.compute_all_numerical_grads ( x, target );
				deeplstm.gradcheck();
				
				#endif
				
				//dtype train_error = smooth_loss;
				/*
					- sample size
					- min
					- mean
					- max
					- std err
				
				*/
				std::tuple<size_t, dtype, dtype, dtype, dtype> test_error_tuple, test2_error_tuple, train_error_tuple;
				
				train_error_tuple = deeplstm.test_batch ( data, epoch_length / 10, codes, reset_std,
									loss_in_bits );
				test_error_tuple = deeplstm.test_batch ( valid, epoch_length / 10, codes, reset_std,
								   loss_in_bits );
								   
				test2_error_tuple = deeplstm.test_batch ( test, epoch_length / 10, codes, reset_std,
									loss_in_bits );
									
				dtype avg_test_error = std::get<2> ( test_error_tuple );
				dtype avg_train_error = std::get<2> ( train_error_tuple );
				
				dtype avg_test2_error = std::get<2> ( test2_error_tuple );
				
				//std::cout << "Test: " <<  avg_test2_error << std::endl;
				
				if ( !std::isnan ( avg_test_error ) && !std::isinf ( avg_test_error ) )
					test_error = test_error < 0 ? avg_test_error : test_error *
								 test_loss_dampening + ( 1 - test_loss_dampening ) * avg_test_error;
								 
				if ( !std::isnan ( avg_train_error )  && !std::isinf ( avg_train_error ) )
					train_error = train_error < 0 ? avg_train_error : train_error *
								  test_loss_dampening + ( 1 - test_loss_dampening ) * avg_train_error;
								  
								  
				size_t test_length = test.rows();
				
				// std::cout << std::setprecision ( 3 ) << "Smooth loss: " << smooth_loss << ", Train error: " <<
				// 		  train_error << ", Test error: " << test_error << std::endl;
				
				// TODO: clean up
				results_size++;
				
				//TODO: change to std::string
				std::vector<char> generated_text = deeplstm.sample ( 5000,
												   codes, " ", reset_std );
												   
				std::ofstream FILE ( "samples/" + out_filename +
									 "_sample" "_" + to_string_with_precision ( test_error * 1000,
											 0 ) + ".txt", std::ios::out | std::ofstream::binary );
				std::copy ( generated_text.begin(), generated_text.end(),
							std::ostreambuf_iterator<char> ( FILE ) );
							
				FILE.close();
				
				/* TODO: move this somewhere */
				
				std::string results =
					to_string_with_precision ( results_size, 0 ) + " " +
					to_string_with_precision ( ( e + ( float ) ( i + 1 ) / ( float ) epoch_length ), 1 ) + " " +
					to_string_with_precision ( iterations, 0 ) + " " +
					to_string_with_precision ( test_time, 0 ) + " " +
					to_string_with_precision ( train_error, 3 ) + " " +
					to_string_with_precision ( test_error, 3 ) + " " +
					to_string_with_precision ( smooth_loss, 3 ) + " " +
					to_string_with_precision ( gflops_per_sec, 1 ) + " " +
					to_string_with_precision ( std::get<1> ( train_error_tuple ), 3 ) + " " +
					to_string_with_precision ( std::get<3> ( train_error_tuple ), 3 ) + " " +
					to_string_with_precision ( std::get<4> ( train_error_tuple ), 3 ) + " " +
					to_string_with_precision ( std::get<1> ( test_error_tuple ), 3 ) + " " +
					to_string_with_precision ( std::get<3> ( test_error_tuple ), 3 ) + " " +
					to_string_with_precision ( std::get<4> ( test_error_tuple ), 3 ) + " " +
					to_string_with_precision ( std::get<1> ( test2_error_tuple ), 3 ) + " " +
					to_string_with_precision ( std::get<2> ( test2_error_tuple ), 3 ) + " " +
					to_string_with_precision ( std::get<3> ( test2_error_tuple ), 3 ) + " " +
					to_string_with_precision ( std::get<4> ( test2_error_tuple ), 3 );
					
				std::ofstream out ( "results6/" + out_filename + ".txt", std::ofstream::out | std::ofstream::app );
				
				out << results << std::endl;
				std::cout << results << std::endl;
				out.close();
				
				if ( serialize_counter == serialize_every ) {
				
					std::cout << "Syncing CPU-GPU... " << std::flush;
					deeplstm.sync_all_host();
					std::cout << "Done." << std::endl;
					std::cout << "Serializing... " << std::flush;
					std::ofstream serialization_out ( "snapshots/" + out_filename + "_" + to_string_with_precision ( test_error * 1000,
													  0 ) + ".json" );
					//cereal::PortableBinaryOutputArchive archive_o(out);
					cereal::JSONOutputArchive archive_o ( serialization_out );
					archive_o ( deeplstm );
					
					std::cout << "Done." << std::endl;
					serialize_counter = 0;
				}
				
				test_timer.start();
				
			}
			
			//weight update
			deeplstm.adapt ( learning_rate );
			
			//state(0) = state(last)
			deeplstm.carryContext ( S - 1 );
		}
		
	}
	
	#ifdef __USE_CLBLAS__
	teardown_clblas ();
	#endif
	
	#ifdef __USE_CUDA__
	teardown_cublas ();
	#endif
	
	return 0;
	
}
