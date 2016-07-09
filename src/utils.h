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
 * @Date:   2016-03-24 15:25:43
 * @Last Modified by:   kmrocki
 * @Last Modified time: 2016-03-24 18:36:57
 *
 * TODO: clean up
 */

#ifndef __UTILS_H__
#define __UTILS_H__

#include <time.h>
typedef struct tm tm_t;

#include <sstream>

#define safe_delete(x) if ((x) != nullptr) delete((x))

dtype count_flops ( size_t M, size_t N, size_t S,
					size_t B ) {
					
					
	return ( S - 1 ) * (
			   //forward
			   ( N * M * B * 2 ) + ( N * 4 * N * B ) + ( N * 4 * B * 2 ) +
			   ( 5 * N * 4 * B ) + //nolinearities
			   ( 6 * N * B ) + //c(t) + h(t)
			   ( M * N * B * 2 ) + // y[t].array()
			   ( 8 * N * B ) + //probs[t]
			   //backward
			   ( N * B ) +
			   ( M * B * N * 3 ) +
			   ( N * B * 6 ) +
			   ( N * M * B * 4 )
			   + // dh = Why.transpose() * dy[t] + dhnext;
			   ( N * B * 8 ) +
			   ( N * 4 * B * M * 3 ) + // dU += dg * h[t - 1].transpose();
			   ( N * 4 * B * N * 3 )
			   + // dW += dg * deeplstm.lstm[0].x[t].transpose();
			   ( N * 4 * B ) +
			   ( N * 4 * N * B * 2 ) + //dhnext = U.transpose() * dg;
			   ( N * B ) //dcnext.array() = dc.array() * g[t].block<N, B>(2 * N, 0).array();
		   ) +
		   8 * ( M * N + M + N * 4 * N + N * 4 * M + N * 4 ); //adapt
};

tm_t seconds2time ( size_t seconds ) {

	tm_t t;
	t.tm_hour = seconds / 3600;
	t.tm_min = ( seconds % 3600 ) / 60;
	t.tm_sec = seconds % 60;
	return t;
}

template <typename T>
std::string to_string_with_precision ( const T a_value, const int n = 6 ) {
	std::ostringstream out;
	out << std::fixed << std::setprecision ( n ) << a_value;
	return out.str();
}

#define PRINT_INFO() 	std::cout << std::setw(12) << "[Epoch " << e + 1 << "/" << epochs << "]" << std::fixed << \
								  std::setw(6) << std::setprecision(0) << \
								  100.0f * (float)(i + 1) / (float)epoch_length << "%   (" << \
								  std::setw(2) << eta.tm_hour << " h " << std::setfill('0') << \
								  std::setw(2) << eta.tm_min << " m " << std::setfill('0') << \
								  std::setw(2) << eta.tm_sec << " s, " << test_eta << " s)" << std::setfill(' ') << \
								  std::setw(8) << std::setprecision(6) << "ce = " << smooth_loss << \
								  std::setw(7) << std::setprecision(1) << gflops_per_sec << \
								  " GFlOP/s    " << "\r" << std::flush;
// << (double)total_bytes_copied / (double)(1024 * 1024 * 1024)
// << " GB \r" << std::flush;
#endif