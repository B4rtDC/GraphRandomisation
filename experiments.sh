#!/bin/bash

# Run the undirect experiments (Zachary karate club network)
julia --project=. -e 'include("./experiments/UndirectedExperiments.jl")'

# Run the directed experiments (Chesapeake Bay foodweb)
julia --project=. -e 'include("./experiments/DirectedExperiments.jl")'

# Run the bipartite experiments (Crime & escort networks)
julia --project=. -e 'include("./experiments/BipartiteExperiments.jl")'