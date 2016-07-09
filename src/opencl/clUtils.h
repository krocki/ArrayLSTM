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
 * OpenCL utils
 *
 */

#define STR_EXPAND(tok) #tok
#define STR(tok) STR_EXPAND(tok)
#define CL_SAFE_CALL(x)   do { \
		cl_int ret = (x); checkError(ret, STR(x)); \
	} while (0)

static const char *errorstring[] = {
	"cl_success",
	"cl_device_not_found",
	"cl_device_not_available",
	"cl_compiler_not_available",
	"cl_mem_object_allocation_failure",
	"cl_out_of_resources",
	"cl_out_of_host_memory",
	"cl_profiling_info_not_available",
	"cl_mem_copy_overlap",
	"cl_image_format_mismatch",
	"cl_image_format_not_supported",
	"cl_build_program_failure",
	"cl_map_failure",
	"",
	"",
	"",
	"",
	"",
	"",
	"",
	"",
	"",
	"",
	"",
	"",
	"",
	"",
	"",
	"",
	"",
	"cl_invalid_value",
	"cl_invalid_device_type",
	"cl_invalid_platform",
	"cl_invalid_device",
	"cl_invalid_context",
	"cl_invalid_queue_properties",
	"cl_invalid_command_queue",
	"cl_invalid_host_ptr",
	"cl_invalid_mem_object",
	"cl_invalid_image_format_descriptor",
	"cl_invalid_image_size",
	"cl_invalid_sampler",
	"cl_invalid_binary",
	"cl_invalid_build_options",
	"cl_invalid_program",
	"cl_invalid_program_executable",
	"cl_invalid_kernel_name",
	"cl_invalid_kernel_definition",
	"cl_invalid_kernel",
	"cl_invalid_arg_index",
	"cl_invalid_arg_value",
	"cl_invalid_arg_size",
	"cl_invalid_kernel_args",
	"cl_invalid_work_dimension",
	"cl_invalid_work_group_size",
	"cl_invalid_work_item_size",
	"cl_invalid_global_offset",
	"cl_invalid_event_wait_list",
	"cl_invalid_event",
	"cl_invalid_operation",
	"cl_invalid_gl_object",
	"cl_invalid_buffer_size",
	"cl_invalid_mip_level",
	"cl_invalid_global_work_size",
};

__inline static const char *oclErrorString ( cl_int error ) {

	const int  errorcount = sizeof ( errorstring ) / sizeof ( errorstring[0] );
	const int  index = -error;
	return ( index >= 0 && index < errorcount ) ? errorstring[index] : "";
}

void checkError ( const cl_int ciErrNum, const char *const operation ) {
	if ( ciErrNum != CL_SUCCESS )
		printf ( "ERROR: %s failed, %s\n", operation, oclErrorString ( ciErrNum ) );
}