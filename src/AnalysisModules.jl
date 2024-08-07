
module AnalysisModules
    using Pkg
    cd(joinpath(@__DIR__,".."))
    @info("working in $(pwd())")
    Pkg.activate(pwd())
    @info "Running AnalysisModules on $(Threads.nthreads()) threads, loading packages..."

    # Graph libraries
    using Graphs
    using MaxEntropyGraphs
    using PyCall # for external libraries through Python

    # Data libraries
    using DataFrames
    using CSV
    using JLD2

    # Computation / analysis libraries
    import Statistics: mean, std
    import StatsBase: countmap, sample
    import Combinatorics: combinations
    using Distributions
    using HypothesisTests: ApproximateTwoSampleKSTest

    # Plotting libraries
    using Plots
    using StatsPlots
    using Measures

    # Other
    using Dates
    using ProgressMeter
    using Logging

    # Parallelization
    import Distributed

    ## Exports Computation
    export 
    zscore,
    empirical_pvalue,
    KL_divergence,
    KS_metric,
    LP_metric,
    community_size_leiden,
    aggregated_zscore_pvalue
    
    ## Plotting stuff
    export highlevelplot

    ## Exports Simple Graphs
    export 
    networkx_configuration_model, 
    networkx_chung_lu_model, 
    networkx_random_degree_sequence_graph, 
    igraph_configuration_model,
    networkit_chung_lu_model,
    networkit_configuration_model_graph,
    networkit_curveball,
    networkit_edge_switching_markov_chain,
    graph_tool_configuration_model,
    NEMtropy_UBCM,
    MaxEntropyGraphs_UBCM

    ## Exports Simple Directed Graphs
    export
    networkx_directed_configuration_model,
    igraph_directed_configuration_model,
    MaxEntropyGraphs_DBCM,
    networkit_directed_curveball,
    networkit_directed_edge_switching_markov_chain,
    graph_tool_directed_configuration_model

    ## Exports Bipartite Graphs
    export 
    networkx_bipartite_configuration_model, 
    MaxEntropyGraphs_UBCM,
    bipartite_curveball,
    bipartite_chung_lu_fast


    """
        AnalysisModules

    This hypermodule contains the functions used to analyze and generate the to produce the figures in the paper.

    ## Usage
    For efficieny, it is suggested to run the analysis modules in parallel. 
    This can be done using the `Distributed` module.

    """
    AnalysisModules

    ## Submodules
    # Simple graphs
    include("./UndirectedSimpleGraphs.jl")
    using .UndirectedSimpleGraphs
    # Directed graphs
    include("./DirectedSimpleGraphs.jl")
    using .DirectedSimpleGraphs
    # Bipartite graphs
    include("./BipartiteGraphs.jl")
    using .BipartiteGraphs
    # Analysis helper functions
    include("./AnalysisHelpers.jl")
    using .AnalysisHelpers


    


    @info "Packages loaded"
end
 
# using .AnalysisModules

# AnalysisModules.zscore([1,2,3], 7)
# AnalysisModules.empirical_pvalue([1,2,3,4], 7)

# AnalysisModules.zscore([1,2,3], 2)
# G = AnalysisModules.Graphs.smallgraph(:karate)
# N = 20

# using Distributed
# # add workers
# addprocs(3)
# # load packages on workers
# @everywhere include() .AnalysisModules

# # Testing all options
# AnalysisModules.networkx_configuration_model(G, N)
# AnalysisModules.networkx_chung_lu_model(G, N)
# #AnalysisModules.networkx_random_degree_sequence_graph(G, N) # Very slow method
# AnalysisModules.igraph_configuration_model(G, N, method=:configuration)
# AnalysisModules.igraph_configuration_model(G, N, method=:fast_heur_simple)
# #AnalysisModules.igraph_configuration_model(G, N, method=:configuration_simple) # Very slow method
# AnalysisModules.igraph_configuration_model(G, N, method=:edge_switching_simple)
# AnalysisModules.igraph_configuration_model(G, N, method=:vl)
# AnalysisModules.networkit_chung_lu_model(G, N)
# AnalysisModules.networkit_configuration_model_graph(G, N)
# AnalysisModules.networkit_curveball(G, N)
# AnalysisModules.networkit_edge_switching_markov_chain(G, N)
# AnalysisModules.graph_tool_configuration_model(G, N, method=:configuration)
# AnalysisModules.NEMtropy_UBCM(G, N)
# AnalysisModules.MaxEntropyGraphs_UBCM(G, N)

