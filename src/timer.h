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
 * 	@Date:   2015-04-29
 * 	@Last Modified by:   kmrocki
 * 	@Last Modified time: 2015-04-29 15:23:21
 */

#ifndef __TIMER_H__
#define __TIMER_H__

#include <sys/time.h>

class Timer {

	public:
	
		Timer() = default;
		~Timer() = default;
		
		void start ( void ) {
		
			gettimeofday ( &s, NULL );
		}
		
		double end ( void ) {
		
			struct timeval  diff_tv;
			gettimeofday ( &e, NULL );
			
			diff_tv.tv_usec = e.tv_usec - s.tv_usec;
			diff_tv.tv_sec = e.tv_sec - s.tv_sec;
			
			if ( s.tv_usec > e.tv_usec ) {
				diff_tv.tv_usec += 1000000;
				diff_tv.tv_sec--;
			}
			
			return ( double ) diff_tv.tv_sec + ( ( double )
												 diff_tv.tv_usec / 1000000.0 );
												 
		}
		
		/* TODO
			void pause() {}
		*/
		
		
	protected:
	
		struct timeval s;
		struct timeval e;
};

#endif /* __TIMER_H__ */