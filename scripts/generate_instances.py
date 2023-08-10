#!/usr/bin/env python3



# Copyright 2023 L3Harris Technologies, Inc.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
#


# add some relative paths so we can import sampling.py module.
import sys
sys.path.append("../")
sys.path.append("../src")
sys.path.append("./src")
sys_path = sys.path

import sampling


G = sampling.Graph()

# TODO: work in progress.
# n = number of nodes
# periodic lattice or free form glass graph
# other shape lattices
# 2D lattices
# 3D lattices
# edge weights:  ferromagnetic (computationally easy, but interesting), antiferromagnetic, glassy
# vary temperature parameter for each
# external field (none, all the same, all positive but variable, glassy)

# TODO: consider incorporating k_B and T into the sampling.Graph class.


print("done")


