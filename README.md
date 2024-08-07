# Graph randomisation methods
A repository to recreate the experiments from the paper "Comparative analysis of graph randomization: Tools,methods, pitfalls, and best practices", which can be found [here](https://arxiv.org/abs/2405.05400).
The main code runs in Julia (these experiments were run using Julia 1.9.3), or uses Julia to generate and run Python or C code.

The following tools or packages are included in this analysis:
* networkX (Python)
* igraph (Python/C)
* networkit (Python/C)
* graph-tool (Python/C)
* NEMtropy (Python)
* MaxEntropyGraphs.jl (Julia)


## Setup 
In order for all of this to work, the following prerequisites are expected to be present:
- conda pre-installed (for recreation of the python virtual environment)
- Julia 1.9.3 installed (for recreation of the Julia virtual environment)
- gcc should also be available in order to run the C codes

The setup.sh script takes care of most of the installation. In a nutshell, it does the following operations:
1. configure the python environment
2. expose this python environment to Julia
3. install additional Julia packages required

There are some additional specific issues that need to be addressed nevertheless, notably:
1. There is a circular import issue in graph_tool. You will probably get the an error similar to the one shown below:
    ```sh
    ImportError("cannot import name 'minimize_blockmodel_dl' from partially initialized module 'graph_tool.inference' (most likely due to a circular import) (CONDAPATH/graphrandomisation/lib/python3.11/site-packages/graph_tool/inference/__init__.py)")
    File "JULIAPATH/.julia/packages/PyCall/1gn3u/src/pyeval.jl", line 5, in <module>
        const _namespaces = Dict{Module,PyDict{String,PyObject,true}}()
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    File "CONDAPATH/graphrandomisation/lib/python3.11/site-packages/graph_tool/inference/__init__.py", line 330, in <module>
        from . base_states import *
    File "CONDAPATH/envs/graphrandomisation/lib/python3.11/site-packages/graph_tool/inference/base_states.py", line 33, in <module>
        import graph_tool.draw
    File "CONDAPATH/graphrandomisation/lib/python3.11/site-packages/graph_tool/draw/__init__.py", line 87, in <module>
        from .. inference import minimize_blockmodel_dl, BlockState, ModularityState
    ```
    This can be fixed by commenting the line that loads up the drawing functionalities (as we won't be using these anyway). This is located in "CONDAPATH/envs/graphrandomisation/lib/python3.11/site-packages/graph_tool/inference/base_states.py" file on line 33.
    By commenting the line, the problem is fixed:
    ```python
    #! /usr/bin/env python
    # -*- coding: utf-8 -*-
    #
    # graph_tool -- a general graph manipulation python module
    #
    # Copyright (C) 2006-2024 Tiago de Paula Peixoto <tiago@skewed.de>
    #
    # This program is free software; you can redistribute it and/or modify it under
    # the terms of the GNU Lesser General Public License as published by the Free
    # Software Foundation; either version 3 of the License, or (at your option) any
    # later version.
    #
    # This program is distributed in the hope that it will be useful, but WITHOUT
    # ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
    # FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
    # details.
    #
    # You should have received a copy of the GNU Lesser General Public License
    # along with this program. If not, see <http://www.gnu.org/licenses/>.

    from .. import Vector_size_t, group_vector_property, _parallel

    from . util import *

    import functools
    from abc import ABC, abstractmethod
    import inspect
    import functools
    import textwrap
    import math
    import numpy

    #import graph_tool.draw # comment this line to resolve the circular importation error.
    [...]
    ```
2. When running on a Mac, networkit can have some issues with segmentation faults, independent of this code (see also its issue list).

## Running the experiments
Essentially, you can run the file `experiments.sh`. This will do the following for the different models:
1. Generate the random graphs using the different methods. The randomization is spread over multiple workers to profit from distributed computing. The graphs will be stored in a dataframe, that will also be written to disk (in the `results` folder.)
2. Compute a set of relevant metrics.
3. Generate the plots and store them in the `plots` folder.

If you want to go more into detail, you can have a look in the different folders:
- `src` contains all the random graph generation and supporting functions
- `experiments` contains the code to run the experiments and generate the figures
- `plots` will contain the generated figures
- `results` will contain the generated data (graphs + metrics as a dataframe)


