


<snippet>
  <content><![CDATA[
# Array-LSTM
Technical Report: Recurrent Memory Array Structures
Arxiv 1610414, 11 July 2016
## Installation

to compile, run:

make PRECISE_MATH=0 cuda

or for precise FP64 (useful for debugging)
make PRECISE_MATH=1 cuda
 
CPU versions are outdated, OpenCL version is not fully implemented

## Usage
 
run like this
 
./deeplstm N B S GPU (test_every_seconds)
example: ./deeplstm 512 100 64 0 (512 hidden nodes, 100 BPTT steps, batchsize = 64, GPU id = 0)
 
## Contributing
1. Fork it!
2. Create your feature branch: `git checkout -b my-new-feature`
3. Commit your changes: `git commit -am 'Add some feature'`
4. Push to the branch: `git push origin my-new-feature`
5. Submit a pull request :D
## History
Author: Kamil M Rocki (kmrocki@us.ibm.com)
## License
 Copyright (c) 2016, IBM Corporation. All rights reserved.
 
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.
 
 You should have received a copy of the GNU General Public License
 along with this program; if not, write to the Free Software
 Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
]]></content>
  <tabTrigger>readme</tabTrigger>
</snippet>
