module UndirectedSimpleGraphs
    using ..Graphs
    using ..PyCall # for external libraries through Python
    import ..Distributed: myid
    using ..ProgressMeter
    using ..CSV
    using ..MaxEntropyGraphs

    const workingfolder = "./igraph_temp/" # working folder for igraph

    """
        networkx_configuration_model(d::Vector, N::Int; kwargs...)
        networkx_configuration_model(G::SimpleGraph, N::Int; kwargs...)

    From a given degree sequence `d` or graphs `G`, generate `N` random graphs using [NetworkX's configuration model](https://networkx.org/documentation/stable/reference/generated/networkx.generators.degree_seq.configuration_model.html).
    Parallel edges and self-loops will be removed from the resulting graphs. Stub-matching based.

    Returns a vector of `N` graphs to work in Julia. 
    """
    function networkx_configuration_model(d::Vector, N::Int; kwargs...)
        # python dependency
        nx = pyimport("networkx")
        # progress bar
        P = Progress(N; dt=1.0, desc="networkx_configuration_model sampling on worker $(myid()):")
        # pre-allocate
        res = Vector{SimpleGraph}(undef, N)
        for i in 1:N
            # create the random graph
            G = nx.configuration_model(d)
            # remove parallel edges and self-loops
            G = nx.Graph(G)
            G.remove_edges_from(nx.selfloop_edges(G))
            # make iterator for edges
            edge_iter = (Graphs.SimpleEdge(e[1]+1, e[2]+1) for e in nx.edges(G))
            # convert to Julia graph
            Gj = Graphs.SimpleGraphFromIterator(edge_iter)
            # check if vertices are all there
            while nv(Gj) ≠ length(d)
                add_vertex!(Gj)
            end
            # store the result
            res[i] = Gj
            # update progress bar
            next!(P)
        end

        return  res
    end

    networkx_configuration_model(G::SimpleGraph, N::Int; kwargs...) = networkx_configuration_model(Graphs.degree(G), N; kwargs...)


    """
        networkx_chung_lu_model(d::Vector, N::Int; kwargs...)
        networkx_chung_lu_model(G::SimpleGraph, N::Int; kwargs...)

    From a given degree sequence `d` or graphs `G`, generate a random graph using [NetworkX's Chung-Lu model](https://networkx.org/documentation/stable/reference/generated/networkx.generators.degree_seq.expected_degree_graph.html).
    Self-loops will be removed from the resulting graphs.

    For finite graphs this model doesn't produce exactly the given expected degree sequence.
    """
    function networkx_chung_lu_model(d::Vector, N::Int; kwargs...)
        # python dependency
        nx = pyimport("networkx")
        # progress bar
        P = Progress(N; dt=1.0, desc="networkx_chung_lu_model sampling on worker $(myid()):")
        # pre-allocate
        res = Vector{SimpleGraph}(undef, N)
        for i in 1:N
            # create the random graph
            G = nx.expected_degree_graph(d)
            # remove self-loops
            G.remove_edges_from(nx.selfloop_edges(G))
            # make iterator for edges
            edge_iter = (Graphs.SimpleEdge(e[1]+1, e[2]+1) for e in nx.edges(G))
            # convert to Julia graph
            Gj = Graphs.SimpleGraphFromIterator(edge_iter)
            # check if vertices are all there
            while nv(Gj) ≠ length(d)
                add_vertex!(Gj)
            end
            # store the result
            res[i] = Gj
            # update progress bar
            next!(P)
        end

        return  res
    end

    networkx_chung_lu_model(G::SimpleGraph, N::Int; kwargs...) = networkx_chung_lu_model(Graphs.degree(G), N; kwargs...)

    
    """
        networkx_random_degree_sequence_graph(d::Vector, N::Int; tries::Int=500, localattempts::Int=10, kwargs...)
        networkx_random_degree_sequence_graph(G::SimpleGraph, N::Int; tries::Int=500, localattempts::Int=10, kwargs...)

    From a given degree sequence `d`, generate a random graph using [NetworkX's model](https://networkx.org/documentation/stable/reference/generated/networkx.generators.degree_seq.random_degree_sequence_graph.html)

    This should exactly match the degree sequence and sample (near) uniformly from the set of graphs with that degree sequence.
    This algorithm is not guaranteed to produce a graph, which is why we have the `tries` and `localattempts` parameters.
    `tries` is the number of times to try to regenerate a graph, and 
    `localattempts` is the number of times to try to generate graph `i` out of `N`.
    """
    function networkx_random_degree_sequence_graph(d::Vector, N::Int; tries::Int=500, localattempts::Int=10, kwargs...)
        # python dependency
        nx = pyimport("networkx")
        # progress bar
        P = Progress(N; dt=1.0, desc="networkx_random_degree_sequence_graph sampling on worker $(myid()):")
        # pre-allocate
        res = Vector{SimpleGraph}(undef, N)
        for i in 1:N
            for j in 1:localattempts
                Gj = nothing
                # try to generate a graph
                try
                    # create the random graph
                    G = nx.random_degree_sequence_graph(d, tries=tries)
                    # remove self-loops
                    G.remove_edges_from(nx.selfloop_edges(G))
                    # make iterator for edges
                    edge_iter = (Graphs.SimpleEdge(e[1]+1, e[2]+1) for e in nx.edges(G))
                    # convert to Julia graph
                    Gj = Graphs.SimpleGraphFromIterator(edge_iter)
                    # check if vertices are all there
                    while nv(Gj) ≠ length(d)
                        add_vertex!(Gj)
                    end
                    # if we got here, we have a graph
                    # store the result
                    res[i] = Gj
                    # break out of the inner loop
                    break
                catch
                    continue
                end
            end
            # update progress bar
            next!(P)
        end

        return  res
    end

    networkx_random_degree_sequence_graph(G::SimpleGraph, N::Int; tries::Int=500, localattempts::Int=10, kwargs...) = networkx_random_degree_sequence_graph(Graphs.degree(G), N; tries=tries, localattempts=localattempts, kwargs...)


    """
        igraph_configuration_model(d::Vector, N::Int; method::Symbol=:configuration, kwargs...)
        igraph_configuration_model(G::SimpleGraph, N::Int; method::Symbol=:configuration, kwargs...)

    From a given degree sequence `d` or graph `G`, generate `N` random graphs using [igraph's configuration model](https://igraph.org/c/doc/igraph-Generators.html#igraph_degree_sequence_game).

    For downstream tasks, parallel edges and self-loops will be removed from the resulting graphs.
    # Arguments:
    - `d::Vector`: the degree sequence to use
    - `method::Symbol`: the method to use for generating the graph. The following methods are available:
        - `:configuration`: simple generator that implements the stub-matching configuration model. It may generate self-loops and multiple edges. This method does not sample multigraphs uniformly, but it can be used to implement uniform sampling for simple graphs by rejecting any result that is non-simple (i.e. contains loops or multi-edges).
        - `:fast_heur_simple`: similar to `:configuration` but avoids the generation of multiple and loop edges at the expense of increased time complexity. The method will re-start the generation every time it gets stuck in a configuration where it is not possible to insert any more edges without creating loops or multiple edges, and there is no upper bound on the number of iterations, but it will succeed eventually if the input degree sequence is graphical and throw an exception if the input degree sequence is not graphical. This method does not sample simple graphs uniformly.
        - `:configuration_simple`: similar to `:configuration` but rejects generated graphs if they are not simple. This method samples simple graphs uniformly.
        - `:edge_switching_simple`: an MCMC sampler based on degree-preserving edge switches. It generates simple undirected or directed graphs. The algorithm uses L{Graph.Realize_Degree_Sequence()} to construct an initial graph, then rewires it using L{Graph.rewire()}.
        - `:vl:`: a more sophisticated generator that can sample undirected, connected simple graphs approximately uniformly. It uses edge-switching Monte-Carlo methods to randomize the graphs. This generator should be favoured if undirected and connected graphs are to be generated and execution time is not a concern. igraph uses the original implementation of Fabien Viger; see the following [URL and the paper cited on it for the details of the details](https://www-complexnetworks.lip6.fr/~latapy/FV/generation.html).
    """
    function igraph_configuration_model(d::Vector, N::Int; method::Symbol=:configuration, kwargs...)
        # python dependency
        ig = pyimport("igraph")
        # progress bar
        P = Progress(N; dt=1.0, desc="igraph_configuration_model_$(method) sampling on worker $(myid()):")
        # pre-allocate
        res = Vector{SimpleGraph}(undef, N)
        for i in 1:N
            G = ig.Graph.Degree_Sequence(d, method=method)
            # remove self-loops and parallel edges
            G.simplify(multiple=true, loops=true, combine_edges=nothing)
            # make iterator for edges
            edge_iter = (Graphs.SimpleEdge(e.source + 1, e.target + 1) for e in G.es)
            # convert to Julia graph
            Gj = Graphs.SimpleGraphFromIterator(edge_iter)
            # check if vertices are all there
            while nv(Gj) ≠ length(d)
                add_vertex!(Gj)
            end
            # store the result
            res[i] = Gj
            # update progress bar
            next!(P)
        end

        return res
    end

    igraph_configuration_model(G::SimpleGraph, N::Int; method::Symbol=:configuration, kwargs...) = igraph_configuration_model(Graphs.degree(G), N; method=method, kwargs...)

    """
        igraph_chung_lu_approximation(G::Simplegraph, N::Int; kwargs...)

    From a given graph `G`, generate `N` random graphs using [igraph's static fitness game](https://igraph.org/c/html/latest/igraph-Generators.html#igraph_static_fitness_game).
    """
    function igraph_chung_lu_approximation(G::SimpleGraph, N::Int; loops=false, multiple=false, kwargs...) 
        # python dependency
        ig = pyimport("igraph")
        # progress bar
        P = Progress(N; dt=1.0, desc="igraph_chung_lu_approximation sampling on worker $(myid()):")
        # pre-allocate
        res = Vector{SimpleGraph}(undef, N)
        for i in 1:N
            g = ig.GraphBase.Static_Fitness(ne(G),fitness_out=Graphs.degree(G), loops=loops, multiple=multiple)
            # remove self-loops and parallel edges
            #G.simplify(multiple=true, loops=true, combine_edges=nothing)
            # make iterator for edges
            edge_iter = (Graphs.SimpleEdge(e[1] + 1, e[2] + 1) for e in g.get_edgelist())
            # convert to Julia graph
            Gj = Graphs.SimpleGraphFromIterator(edge_iter)
            # check if vertices are all there
            while nv(Gj) ≠ nv(G)
                add_vertex!(Gj)
            end
            # store the result
            res[i] = Gj
            # update progress bar
            next!(P)
        end

        return res
    end

    """
        networkit_chung_lu_model(d::Vector, N::Int; kwargs...)
        networkit_chung_lu_model(G::SimpleGraph, N::Int; kwargs...)

    From a given degree sequence `d` or graph `G`, generate `N` random graphs using [NetworKit's Chung-Lu model](https://networkit.github.io/dev-docs/python_api/generators.html).
    """
    function networkit_chung_lu_model(d::Vector, N::Int; kwargs...)
        # python dependency
        nk = pyimport("networkit")
        # progress bar
        P = Progress(N; dt=1.0, desc="networkit_chung_lu_model sampling on worker $(myid()):")
        # mapping for sorted degree sequence
        d_ind = sortperm(d, rev=true)

        # pre-allocate
        res = Vector{SimpleGraph}(undef, N)
        S = nk.generators.ChungLuGenerator(d)
        for i in 1:N
            G = S.generate()
            # make iterator for edges
            edge_iter = (Graphs.SimpleEdge(d_ind[e[1] + 1], d_ind[e[2] + 1]) for e in G.iterEdges()) # fix
            # convert to Julia graph
            res[i] = Graphs.SimpleGraphFromIterator(edge_iter)
            # check if vertices are all there
            while nv(res[i]) ≠ length(d)
                add_vertex!(res[i])
            end
            # update progress bar
            next!(P)
        end

        return res
    end

    networkit_chung_lu_model(G::SimpleGraph, N::Int; kwargs...) = networkit_chung_lu_model(Graphs.degree(G), N; kwargs...)


    """
        networkit_configuration_model_graph(d::Vector, N::Int; kwargs...)
        networkit_configuration_model_graph(G::SimpleGraph, N::Int; kwargs...)

    From a given degree sequence `d` or graph `G`, generate `N` random graphs using [networkit's configuration model](https://networkit.github.io/dev-docs/python_api/generators.html#networkit.generators.EdgeSwitchingMarkovChainGenerator).

    The method is based on the Edge-Switching Markov-Chain method and avoids self-loops and parallel edges.
    """
    function networkit_configuration_model_graph(d::Vector, N::Int; kwargs...)
        # python dependency
        nk = pyimport("networkit")
        # progress bar
        P = Progress(N; dt=1.0, desc="networkit_configuration_model_graph sampling on worker $(myid()):")
        # pre-allocate
        res = Vector{SimpleGraph}(undef, N)
        S = nk.generators.ConfigurationModelGenerator(d) # EdgeSwitchingMarkovChainGenerator
        for i in 1:N
            G = S.generate()
            # make iterator for edges
            edge_iter = (Graphs.SimpleEdge(e[1] + 1, e[2] + 1) for e in G.iterEdges())
            # convert to Julia graph
            res[i] = Graphs.SimpleGraphFromIterator(edge_iter)
            # check if vertices are all there
            while nv(res[i]) ≠ length(d)
                add_vertex!(res[i])
            end
            # update progress bar
            next!(P)
        end

        return res
    end

    networkit_configuration_model_graph(G::SimpleGraph, N::Int; kwargs...) = networkit_configuration_model_graph(Graphs.degree(G), N; kwargs...)


    """
        networkit_curveball(G::SimpleGraph, N::Int; kwargs...) 

    From a given graph `G`, generate `N` random graphs using [networkit's curveball model](https://networkit.github.io/dev-docs/notebooks/Randomization.html).

    """
    function networkit_curveball(G::SimpleGraph, N::Int; kwargs...) 
        # python dependency
        nk = pyimport("networkit")
        # progress bar
        P = Progress(N; dt=1.0, desc="networkit_curveball sampling on worker $(myid()):")
        # convert the graph to networkit
        Gnk = nk.Graph(nv(G), directed=false, weighted=false)
        for e in edges(G)
            Gnk.addEdge(e.src - 1, e.dst - 1) # networkit uses 0-based indexing
        end
        # pre-allocate result
        res = Vector{SimpleGraph}(undef, N)
        # do the work
        for i in 1:N
            # Initialize algorithm 
            globalCurve = nk.randomization.GlobalCurveball(Gnk)
            # Run algorithm 
            globalCurve.run()
            # Get result
            G_rand = globalCurve.getGraph()
            # reshape to Julia graph
            edge_iter = (Graphs.SimpleEdge(e[1] + 1, e[2] + 1) for e in G_rand.iterEdges())
            res[i] = Graphs.SimpleGraphFromIterator(edge_iter)
            # check if vertices are all there
            while nv(res[i]) ≠ nv(G)
                add_vertex!(res[i])
            end
            next!(P)
        end

        return res
    end


    """
        networkit_edge_switching_markov_chain(G::SimpleGraph, N::Int)

    From a given graph `G`, generate `N` random graphs using [networkit's edge switching markov chain model](https://networkit.github.io/dev-docs/notebooks/Randomization.html).
    """
    function networkit_edge_switching_markov_chain(G::SimpleGraph, N::Int; kwargs...)
        # python dependency
        nk = pyimport("networkit")
        # progress bar
        P = Progress(N; dt=1.0, desc="networkit_edge_switching_markov_chain sampling on worker $(myid()):")
        # convert the graph to networkit
        Gnk = nk.Graph(nv(G), directed=false, weighted=false)
        for e in edges(G)
            Gnk.addEdge(e.src - 1, e.dst - 1) # networkit uses 0-based indexing
        end
        # pre-allocate result
        res = Vector{SimpleGraph}(undef, N)
        # do the work
        for i in 1:N
            # Initialize algorithm 
            esr = nk.randomization.EdgeSwitching(Gnk)
            # Run algorithm 
            esr.run()
            # Get result
            G_rand = esr.getGraph()
            # reshape to Julia graph
            edge_iter = (Graphs.SimpleEdge(e[1] + 1, e[2] + 1) for e in G_rand.iterEdges())
            res[i] = Graphs.SimpleGraphFromIterator(edge_iter)
            # check if vertices are all there
            while nv(res[i]) ≠ nv(G)
                add_vertex!(res[i])
            end
            next!(P)
        end

        return res
    end

    

    """
        graph_tool_configuration_model(d::Vector, N::Int; method=:configuration)
        graph_tool_configuration_model(G::SimpleGraph, N::Int; method=:configuration)

    From a given degree sequence `d` or graph `G`, generate `N` random graphs using [random graph](https://graph-tool.skewed.de/static/doc/autosummary/graph_tool.generation.random_graph.html#graph_tool.generation.random_graph), with a given degree distribution and (optionally) vertex-vertex correlation.
    The graph will be randomized via the [random_rewire()](https://graph-tool.skewed.de/static/doc/autosummary/graph_tool.generation.random_rewire.html#graph_tool.generation.random_rewire) function, and any remaining parameters will be passed to that function. 

    The following statistical models can be chosen, which determine how the edges are rewired=
    - `:erdos`: The edges will be rewired entirely randomly, and the resulting graph will correspond to the Erdős-Rényi model.
    - `:configuration`: The edges will be rewired randomly, but the degree sequence of the graph will remain unmodified.
    - `:constrained-configuration`: The edges will be rewired randomly, but both the degree sequence of the graph and the vertex-vertex (in,out)-degree correlations will remain exactly preserved. If the block_membership parameter is passed, the block variables at the endpoints of the edges will be preserved, instead of the degree-degree correlation.
    - `:probabilistic-configuration`: This is similar to constrained-configuration, but the vertex-vertex correlations are not preserved, but are instead sampled from an arbitrary degree-based probabilistic model specified via the edge_probs parameter. The degree-sequence is preserved.
    - `:blockmodel-degree`: This is just like probabilistic-configuration, but the values passed to the edge_probs function will correspond to the block membership values specified by the block_membership parameter.
    - `:blockmodel`: This is just like blockmodel-degree, but the degree sequence is not preserved during rewiring.
    - `:blockmodel-micro`: This is like blockmodel, but the exact number of edges between groups is preserved as well.

    By default, this method does not allow for parallel edges or self-loops.
    """
    function graph_tool_configuration_model(d::Vector, N::Int; method=:configuration, kwargs...)
        # python wrapper for graph-tool
        py"""
        from graph_tool.generation import random_graph

        def configuration_model_graph_tool(d, model="configuration"):
            g = random_graph(len(d), lambda x : d[x], block_membership=None, directed=False, model=model, n_iter=10) # updated following Tiago's suggestion

            return g.iter_edges()

        """
        # progress bar
        P = Progress(N; dt=1.0, desc="graph_tool_model_$(method) sampling on worker $(myid()):")
        # pre-allocate
        res = Vector{SimpleGraph}(undef, N)
        # do the work
        for i in 1:N
            # make iterator for edges
            edge_iter = (Graphs.SimpleEdge(e[1] + 1, e[2] + 1) for e in py"configuration_model_graph_tool"(d, model=method))
            # convert to Julia graph
            Gj = Graphs.SimpleGraphFromIterator(edge_iter)
            # check if vertices are all there
            while nv(Gj) ≠ length(d)
                add_vertex!(Gj)
            end
            res[i] = Gj
            next!(P)
        end

        return res
    end

    function graph_tool_stub_matching(d::Vector, N::Int; kwargs...)
        return graph_tool_configuration_model(d, N; method=:configuration, kwargs...)
    end

    graph_tool_configuration_model(G::SimpleGraph, N::Int; method=:configuration, kwargs...) = graph_tool_configuration_model(Graphs.degree(G), N; method=method, kwargs...)


    """
        graph_tool_stub_matching(G::SimpleGraph, N::Int; kwargs...)

    Generate `N` random graphs with the same degree sequence as `G` using the stub matching approach from the graph-tool library.
    This solution was provided by Tiago Peixoto himself, the author of the graph-tool library.

    # See also:
    https://graph-tool.skewed.de/static/doc/autosummary/graph_tool.generation.generate_sbm.html
    """
    function graph_tool_stub_matching(G::SimpleGraph, N::Int; kwargs...)
        # Generate the edge list iterator
        edge_iter = ((e.src-1, e.dst-1) for e in edges(G))

        # python wrapper for graph-tool
        py"""
        from graph_tool import Graph
        import numpy as np
        from graph_tool.generation import generate_sbm
        # convert to graph-tool graph
        g = Graph($(edge_iter), directed=False)
        # sanity check
        assert g.num_vertices() == $(Graphs.nv(G))
        assert g.num_edges() == $(Graphs.ne(G))
        assert (g.get_out_degrees(g.get_vertices()) == $(Graphs.degree(G))).all()
        # block membership
        b = [0 for _ in range($(Graphs.nv(G)))] 
        # inter-block edges (all together)
        probs = np.matrix([[$(Graphs.ne(G))*2]])
        
        # generate random graph edge list iterator
        def get_rand_graph():
            return generate_sbm(b, probs, $(Graphs.degree(G)), directed=False, micro_ers=True, micro_degs=True).iter_edges()    
        """

        # progress bar
        P = Progress(N; dt=1.0, desc="graph_tool_stub_matching sampling on worker $(myid()):")
        # pre-allocate
        res = Vector{SimpleGraph}(undef, N)
        # do the work
        for i in 1:N
            # make iterator for edges
            edge_iter = (Graphs.SimpleEdge(e[1] + 1, e[2] + 1) for e in py"get_rand_graph"())
            # convert to Julia graph
            Gj = Graphs.SimpleGraphFromIterator(edge_iter)
            # check if vertices are all there
            while nv(Gj) ≠ Graphs.nv(G)
                add_vertex!(Gj)
            end
            res[i] = Gj
            next!(P) # update progress bar
        end
        return res
    end


    """
        graph_tool_chunglu(G::SimpleGraph, N::Int; kwargs...)

    Generate `N` random graphs with the same degree sequence as `G` using the Chung-Lu approach from the graph-tool library.
    This solution was provided by Tiago Peixoto himself, the author of the graph-tool library.

    # See also:
    https://graph-tool.skewed.de/static/doc/autosummary/graph_tool.generation.generate_maxent_sbm.html
    """
    function graph_tool_chunglu(G::SimpleGraph, N::Int; kwargs...)
    # Generate the edge list iterator
    edge_iter = ((e.src-1, e.dst-1) for e in edges(G))

    # python wrapper for graph-tool
    py"""
    # dependencies
    from graph_tool import Graph
    import numpy as np
    from graph_tool.generation import generate_maxent_sbm
    from graph_tool.inference import solve_sbm_fugacities # minimize_blockmodel_dl, 
    from graph_tool.spectral import adjacency
    #from graph_tool.inference import 
    # convert to graph-tool graph
    g = Graph($(edge_iter), directed=False)
    # sanity checks
    assert g.num_vertices() == $(Graphs.nv(G))
    assert g.num_edges() == $(Graphs.ne(G))
    assert (g.get_out_degrees(g.get_vertices()) == $(Graphs.degree(G))).all()

    ## setup
    # block membership
    b = [0 for _ in range($(Graphs.nv(G)))] 
    # inter-block edges (all together)
    probs = np.matrix([[$(Graphs.ne(G))*2]])
    # fugacities
    out_degs = $(Graphs.degree(G))
    mrs, out_theta = solve_sbm_fugacities(b, probs, out_degs=out_degs)

    # generate random graph edge list iterator
    def get_rand_graph():
        return generate_maxent_sbm(b, mrs, out_theta, directed=False).iter_edges()  # , multigraph=False
    
    """

    # progress bar
    P = Progress(N; dt=1.0, desc="graph_tool_chunglu sampling on worker $(myid()):")
    # pre-allocate
    res = Vector{SimpleGraph}(undef, N)
    # do the work
    for i in 1:N
        # make iterator for edges
        edge_iter = (Graphs.SimpleEdge(e[1] + 1, e[2] + 1) for e in py"get_rand_graph"())
        # convert to Julia graph
        Gj = Graphs.SimpleGraphFromIterator(edge_iter)
        # check if vertices are all there
        while nv(Gj) ≠ Graphs.nv(G)
            add_vertex!(Gj)
        end
        res[i] = Gj
        next!(P)
    end
    return res
    end

    """
        NEMtropy_UBCM(d::Vector, N::Int; kwargs...)    
        NEMtropy_UBCM(G::SimpleGraph, N::Int; kwargs...)

    From a given degree sequence `d` or graph `G`, generate `N` random graphs using NEMtropy's UBCM model.
    """
    function NEMtropy_UBCM(d::Vector, N::Int; kwargs...)
        # python dependency
        nm = pyimport("NEMtropy")
        # progress bar
        P = Progress(N; dt=1.0, desc="NEMtropy_UBCM sampling on worker $(myid()):")
        # build model
        model = nm.UndirectedGraph(degree_sequence=d)
        model.solve_tool(model="cm_exp", method="fixed-point", initial_guess="degrees_minor")
        # preallocate 
        res = Vector{SimpleGraph}(undef, N)

        # sample
        for i in 1:N
            # sample
            model.ensemble_sampler(1, cpu_n=1, output_dir="./sample/")
            # read in the sample in julia as an iterable
            edge_iter = (Graphs.SimpleEdge(parse(Int, e[1]) + 1, parse(Int, e[2]) + 1) for e in CSV.Rows("./sample/0.txt"))
            # return the graph  
            Gj = Graphs.SimpleGraphFromIterator(edge_iter)
            # check if vertices are all there
            while nv(Gj) ≠ length(d)
                add_vertex!(Gj)
            end

            res[i] = Gj
            next!(P)
        end

        return res
    end

    NEMtropy_UBCM(G::SimpleGraph, N::Int; kwargs...) = NEMtropy_UBCM(Graphs.degree(G), N; kwargs...)


    """
        MaxEntropyGraphs_UBCM(d::Vector, N::Int; kwargs...)

        From a given degree sequence `d` or graph `G`, generate `N` random graphs using MaxEntropyGraphs.jl's UBCM model.
    """
    function MaxEntropyGraphs_UBCM(d::Vector, N::Int; kwargs...)
        model = UBCM(d=d)
        solve_model!(model)
        
        return rand(model, N)
    end

    MaxEntropyGraphs_UBCM(G::SimpleGraph, N::Int; kwargs...) = MaxEntropyGraphs_UBCM(Graphs.degree(G), N; kwargs...)


    """
        igraph_degree_sequence_game(d, N; kwargs...)
        igraph_degree_sequence_game(G::Simplegraph, N; kwargs...)

    Call the igraph's degree sequence game function directly from the igraph C library. 
    This function will generate and compile the C code, and use it to generate the random graphs, which will be returned by the function.

    # Arguments:
    - `d::Vector` or `G::SimpleGraph`: the degree sequence or graph to use as a reference for generating the random graphs
    - `N::Int`: the number of graphs to generate
    - `method::Symbol`: the method to use for generating the graph. The following methods are available:
        - `:IGRAPH_DEGSEQ_CONFIGURATION`: simple generator that implements the stub-matching configuration model. It may generate self-loops and multiple edges. This method does not sample multigraphs uniformly, but it can be used to implement uniform sampling for simple graphs by rejecting any result that is non-simple (i.e. contains loops or multi-edges).
        - `:IGRAPH_DEGSEQ_CONFIGURATION_SIMPLE`: similar to `:IGRAPH_DEGSEQ_CONFIGURATION` but rejects generated graphs if they are not simple. This method samples simple graphs uniformly.
        - `:IGRAPH_DEGSEQ_FAST_HEUR_SIMPLE`: similar to `:IGRAPH_DEGSEQ_CONFIGURATION` but avoids the generation of multiple and loop edges at the expense of increased time complexity. The method will re-start the generation every time it gets stuck in a configuration where it is not possible to insert any more edges without creating loops or multiple edges, and there is no upper bound on the number of iterations, but it will succeed eventually if the input degree sequence is graphical and throw an exception if the input degree sequence is not graphical. This method does not sample simple graphs uniformly.
        - `:IGRAPH_DEGSEQ_EDGE_SWITCHING_SIMPLE`: an MCMC sampler based on degree-preserving edge switches. It generates simple undirected or directed graphs. The algorithm uses L{Graph.Realize_Degree_Sequence()} to construct an initial graph, then rewires it using L{Graph.rewire()}.
        - `:IGRAPH_DEGSEQ_VL`: a more sophisticated generator that can sample undirected, *connected* simple graphs approximately uniformly. It uses edge-switching Monte-Carlo methods to randomize the graphs. The algorithm uses the original implementation of Fabien Viger; see the following [URL and the paper cited on it for the details of the details](https://www-complexnetworks.lip6.fr/~latapy/FV/generation.html).

    - `header_folder::String`: the path to the igraph header folder
    - `library_folder::String`: the path to the igraph library folder
    - `temp_folder::String`: the path to the temporary folder to store the C code and the generated graphs
    - `keepraw::Bool`: whether to keep the raw network files generated by the C code
    """
    function igraph_degree_sequence_game(d::Vector, N::Int; method::Symbol=:IGRAPH_DEGSEQ_CONFIGURATION, 
                                            header_folder::String="/opt/homebrew/Cellar/igraph/0.10.12/include/igraph",
                                            library_folder::String="/opt/homebrew/Cellar/igraph/0.10.12/lib",
                                            temp_folder=joinpath(pwd(), workingfolder), 
                                            keepraw::Bool=false, kwargs...)
        
        @info "Running igraph_degree_sequence_game with method: $(method)"
        # preallocate output vector
        G_res = Vector{SimpleGraph}(undef, N)

        # convert the degree sequence to a string
        d_str = join(d, ", ")

        # check if the temp folder exists
        if !isdir(temp_folder)
            mkdir(temp_folder)
        end

        # create the C code
        template = """
        // to compile:
        // gcc -o $(joinpath(temp_folder,String(method)))_program $(joinpath(temp_folder,String(method)))_program.c -I$(header_folder) -L$(library_folder) -ligraph
        // to run:
        // ./$(joinpath(temp_folder,String(method)))_program
        #include <igraph.h>
        #include <stdio.h>


        int generatorfun(const igraph_vector_int_t *d, int n) {
            // loop n times
            for (int i = 0; i < n; i++) {
                // initialize the graph
                igraph_t graph;
                int result;

                // print the vector d using standard output
                //igraph_vector_int_print(d);
            
                // run the game
                result = igraph_degree_sequence_game(&graph, d, NULL, $(method));

                if (result != IGRAPH_SUCCESS) {
                    // decrement the loop counter
                    i--;
                    continue;  // continue to the next iteration if failed
                }

                // open an output file for each iteration with a unique name
                char fname[150];
                sprintf(fname, \"$(joinpath(temp_folder,"graph_%d_$(method).txt"))\", i);
                FILE *file = fopen(fname, "w");

                // print the graph to the file
                igraph_write_graph_edgelist(&graph, file);

                // close the file
                fclose(file);

                // destroy the graph to free memory
                igraph_destroy(&graph);
            }

            return IGRAPH_SUCCESS;
        }

        // run the generator function
        int main() {
            // allocate degree sequence vector
            igraph_vector_int_t d;

            // initialize the degree sequence vector
            igraph_vector_int_init(&d, $(length(d)));

            // set the initial values of the degree sequence vector
            int initial_values[] = {$(d_str)};
            for (int i = 0; i < $(length(d)); i++) {
                VECTOR(d)[i] = initial_values[i];
            }

            // run the generator function for $(N) iterations
            generatorfun(&d, $(N));

            return 0;
        }
        """

         # write the C code to a file
        open("$(joinpath(temp_folder,String(method)))_program.c", "w") do f
            write(f, template)
        end

        # compile the C code
        run(`gcc -o $(joinpath(temp_folder,String(method)))_program $(joinpath(temp_folder,String(method)))_program.c -I$(header_folder) -L$(library_folder) -ligraph`)

        # run the C code
        run(`$(joinpath(temp_folder,String(method)))_program`)

        # read the generated graphs
        Threads.@threads for i in 0:N-1
            # get filename
            fname = joinpath(temp_folder,"graph_$(i)_$(String(method)).txt")
            # open the file
            res = readlines(fname)
            # generate edge iterator
            edge_iter = (Graphs.SimpleEdge(parse(Int,split(e)[1])+1, parse(Int,split(e)[2])+1) for e in res)
            # graph from iterator
            G = SimpleGraphFromIterator(edge_iter)
    
            # check nodecount 
            while nv(G) < length(d)
                add_vertex!(G)
            end
            
            # add to results
            G_res[i+1] = G
    
            # close and remove the file
            keepraw ? nothing : rm(fname)
        end
    
        return G_res
    end

    igraph_degree_sequence_game(G::SimpleGraph, N::Int; method::Symbol=:IGRAPH_DEGSEQ_CONFIGURATION, 
                                            header_folder::String="/opt/homebrew/Cellar/igraph/0.10.12/include/igraph",
                                            library_folder::String="/opt/homebrew/Cellar/igraph/0.10.12/lib",
                                            temp_folder=joinpath(pwd(), workingfolder), 
                                            keepraw::Bool=false, kwargs...)  = igraph_degree_sequence_game(Graphs.degree(G), N, method=method, header_folder=header_folder, library_folder=library_folder, temp_folder=temp_folder, keepraw=keepraw, kwargs...)



    """
        igraph_static_fitness_game(d::Vector, E::Int, N::Int; loops::Bool=false, multiple::Bool=false, kwargs...)
        igraph_static_fitness_game(G::SimpleGraph, E::Int, N::Int; loops::Bool=false, multiple::Bool=false, kwargs...)

    From a given degree sequence `d` or graph `G`, generate `N` random graphs using [igraph's static fitness game](https://igraph.org/c/html/latest/igraph-Generators.html#igraph_static_fitness_game).

    # Arguments:
    - `d::Vector` or `G::SimpleGraph`: the degree sequence or graph to use as a reference for generating the random graphs
    - `E::Int`: the number of edges for the graph
    - `N::Int`: the number of graphs to generate
    - `loops::Bool`: whether to allow self-loops in the graph
    - `multiple::Bool`: whether to allow multiple edges in the graph
    - `header_folder::String`: the path to the igraph header folder
    - `library_folder::String`: the path to the igraph library folder
    - `temp_folder::String`: the path to the temporary folder to store the C code and the generated graphs
    - `keepraw::Bool`: whether to keep the raw network files generated by the C code
    """                                        
    function igraph_static_fitness_game(d::Vector, E::Int, N::Int; loops::Bool=false, multiple::Bool=false,
                                            header_folder::String="/opt/homebrew/Cellar/igraph/0.10.12/include/igraph",
                                            library_folder::String="/opt/homebrew/Cellar/igraph/0.10.12/lib",
                                            temp_folder=joinpath(pwd(), workingfolder), 
                                            keepraw::Bool=false, kwargs...)

        @info "Running igraph_static_fitness_game"

        # preallocate output vector
        G_res = Vector{SimpleGraph}(undef, N)

        # convert the degree sequence to a string
        d_str = join(d, ", ")

        # check if the temp folder exists
        if !isdir(temp_folder)
            mkdir(temp_folder)
        end

        # create the C code
        template = """
        // to compile:
        // gcc -o $(joinpath(temp_folder,"static_fitness_game"))_program $(joinpath(temp_folder,"static_fitness_game"))_program.c -I$(header_folder) -L$(library_folder) -ligraph
        // to run:
        // ./$(joinpath(temp_folder,"static_fitness_game"))_program
        #include <igraph.h>
        #include <stdio.h>


        int generatorfun(const igraph_vector_t *d, int e, int n) {
            // loop n times
            for (int i = 0; i < n; i++) {
                // initialize the graph
                igraph_t graph;
                int result;
            
                // run the game
                result = igraph_static_fitness_game(&graph, e, d, NULL, $(loops), $(multiple));

                if (result != IGRAPH_SUCCESS) {
                    // decrement the loop counter
                    i--;
                    continue;  // continue to the next iteration if failed
                }

                // open an output file for each iteration with a unique name
                char fname[150];
                sprintf(fname, \"$(joinpath(temp_folder,"graph_%d_static_fitness_game.txt"))\", i);
                FILE *file = fopen(fname, "w");

                // print the graph to the file
                igraph_write_graph_edgelist(&graph, file);

                // close the file
                fclose(file);

                // destroy the graph to free memory
                igraph_destroy(&graph);
            }

            return IGRAPH_SUCCESS;
        }

        // run the generator function
        int main() {
            // allocate degree sequence vector
            igraph_vector_t d;

            // initialize the degree sequence vector
            igraph_vector_init(&d, $(length(d)));

            // set the initial values of the degree sequence vector
            int initial_values[] = {$(d_str)};
            for (int i = 0; i < $(length(d)); i++) {
                VECTOR(d)[i] = initial_values[i];
            }

            // run the generator function for $(N) iterations
            generatorfun(&d, $(E), $(N));

            return 0;
        }
        """

            # write the C code to a file
        open("""$(joinpath(temp_folder,"static_fitness_game"))_program.c""", "w") do f
            write(f, template)
        end

        # compile the C code
        run(`gcc -o $(joinpath(temp_folder,"static_fitness_game"))_program $(joinpath(temp_folder,"static_fitness_game"))_program.c -I$(header_folder) -L$(library_folder) -ligraph`)

        # run the C code
        run(`$(joinpath(temp_folder,"static_fitness_game"))_program`)

        # read the generated graphs
        Threads.@threads for i in 0:N-1
            # get filename
            fname = joinpath(temp_folder,"graph_$(i)_static_fitness_game.txt")
            # open the file
            res = readlines(fname)
            # generate edge iterator
            edge_iter = (Graphs.SimpleEdge(parse(Int,split(e)[1])+1, parse(Int,split(e)[2])+1) for e in res)
            # graph from iterator
            G = SimpleGraphFromIterator(edge_iter)
    
            # check nodecount 
            while nv(G) < length(d)
                add_vertex!(G)
            end
            
            # add to results
            G_res[i+1] = G
    
            # close and remove the file
            keepraw ? nothing : rm(fname)
        end
    
        return G_res
    end


    igraph_static_fitness_game(G::SimpleGraph, N::Int; loops::Bool=false, multiple::Bool=false,
                                    header_folder::String="/opt/homebrew/Cellar/igraph/0.10.12/include/igraph",
                                    library_folder::String="/opt/homebrew/Cellar/igraph/0.10.12/lib",
                                    temp_folder=joinpath(pwd(), workingfolder), 
                                    keepraw::Bool=false, kwargs...) = igraph_static_fitness_game(Graphs.degree(G), Graphs.ne(G), N; loops=loops, multiple=multiple, header_folder=header_folder, library_folder=library_folder, temp_folder=temp_folder, keepraw=keepraw, kwargs...)

    

    
    ## Exports
    export 
        networkx_configuration_model, 
        networkx_chung_lu_model, 
        networkx_random_degree_sequence_graph, 
        igraph_configuration_model,
        igraph_chung_lu_approximation,
        networkit_chung_lu_model,
        networkit_configuration_model_graph,
        networkit_curveball,
        networkit_edge_switching_markov_chain,
        graph_tool_configuration_model,
        graph_tool_stub_matching,
        graph_tool_chunglu,
        NEMtropy_UBCM,
        MaxEntropyGraphs_UBCM,
        igraph_degree_sequence_game,
        igraph_static_fitness_game

end