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
 * 	@Author: kmrocki
 * 	@Date:   2016-04-05
 * 	@Last Modified by:   kmrocki
 * 	@Last Modified time: 2016-04-05 18:36:57
 *
 *	stuff for serialization
 */

#ifndef __SERIALIZATION_H__
#define __SERIALIZATION_H__

#ifdef __USE_CEREAL__
	
	#include <cereal/cereal.hpp>
	#include <cereal/archives/binary.hpp>
	#include <cereal/archives/portable_binary.hpp>
	#include <cereal/archives/xml.hpp>
	#include <cereal/archives/json.hpp>
	#include <cereal/types/polymorphic.hpp>
	#include <cereal/types/map.hpp>
	#include <cereal/types/vector.hpp>
	
#endif

#endif /* __SERIALIZATION_H__ */