/*
 *
 * simple Matrix IO
 *
 * Author: Kamil Rocki
 *
 */

#ifndef __IO_H__
#define __IO_H__

#include <fstream>
#include <sstream>
#include <containers/datatype.h>

MatrixXi rawread ( const char *filename ) {

	MatrixXi m ( 0, 0 );
	
	if ( FILE *fp = fopen ( filename, "rb" ) ) {
	
		std::vector<unsigned char> v;
		char buf[1024];
		
		while ( size_t len = fread ( buf, 1, sizeof ( buf ), fp ) )
			v.insert ( v.end(), buf, buf + len );
			
		fclose ( fp );
		
		if ( v.size() > 0 ) {
		
			std::cout << "Read " << v.size() << " bytes (" << filename
					  << ")" << std::endl;
					  
			m.resize ( v.size(), 1 );
			
			// TODO: probably there is a better way to map std::vector to Eigen::MatrixXi
			for ( int i = 0; i < v.size(); i++ )
			
				m ( i ) = ( int ) v[i];
				
				
		} else
		
			std::cout << "Empty file! (" << filename << ")" <<
					  std::endl;
					  
					  
	} else
	
		std::cout << "fopen error: (" << filename << ")" <<
				  std::endl;
				  
	return m;
};

void save_matrix_to_file ( Matrix &m,
						   std::string filename ) {
						   
	// std::cout << "Saving a matrix to " << filename << "... " <<
	// 		  std::endl;
	// std::ofstream file ( filename.c_str() );
	
	// if ( file.is_open() ) {
	
	// 	file << m;
	// 	file.close();
	
	// } else
	
	// 	std::cout << "file save error: (" << filename << ")" <<
	// 			  std::endl;
	
	
}

#define MAXBUFSIZE  ((int) 1e6)

void readMatrix ( Matrix &m, const char *filename ) {

	std::ifstream infile;
	infile.open ( filename );
	
	size_t row, col;
	
	row = 0;
	
	if ( infile.is_open() ) {
	
		while ( ! infile.eof() ) {
		
			std::string line;
			getline ( infile, line );
			
			std::stringstream stream ( line );
			
			col = 0;
			
			while ( ! stream.eof() ) {
				stream >> m ( row, col );
				col++;
			}
			
			row++;
			
		}
		
		infile.close();
		
		std::cout << "! Found " << row << " x " << col <<
				  " matrix: (" << filename << ")" << std::endl;
				  
	} else
	
		std::cout << "file read error: (" << filename << ")" <<
				  std::endl;
				  
};

void load_matrix_from_file ( Matrix &m,
							 std::string filename ) {
							 
	// assume that m has proper dimensions
	readMatrix ( m, filename.c_str() );
	
}

#endif /* __IO_H__ */
