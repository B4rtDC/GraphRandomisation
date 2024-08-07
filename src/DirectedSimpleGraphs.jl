module DirectedSimpleGraphs
    using ..Graphs
    using ..PyCall # for external libraries through Python
    import ..Distributed: myid
    using ..ProgressMeter
    using ..CSV
    using ..MaxEntropyGraphs


    const workingfolder = "./igraph_temp/"

    """
        networkx_directed_configuration_model(d_in::Vector, d_out::Vector, N::Int; kwargs...)
        networkx_directed_configuration_model(G::SimpleGraph, N::Int; kwargs...)

    From a given degree sequence `d` or graphs `G`, generate `N` random graphs using [NetworkX's configuration model](https://networkx.org/documentation/stable/reference/generated/networkx.generators.degree_seq.directed_configuration_model.html#networkx.generators.degree_seq.directed_configuration_model).
    Parallel edges and self-loops will be removed from the resulting graphs. Stub-matching based.

    Returns a vector of `N` graphs to work in Julia. 
    """
    function networkx_directed_configuration_model(d_in::Vector, d_out::Vector, N::Int; kwargs...)
        length(d_in) == length(d_out) || throw(ArgumentError("d_in and d_out must have the same length"))
        # python dependency
        nx = pyimport("networkx")
        # progress bar
        P = Progress(N; dt=1.0, desc="networkx_directed_configuration_model sampling on worker $(myid()):")
        # pre-allocate
        res = Vector{SimpleDiGraph}(undef, N)
        for i in 1:N
            # create the random graph
            G = nx.directed_configuration_model(d_in, d_out)
            # remove parallel edges and self-loops
            G = nx.DiGraph(G)
            G.remove_edges_from(nx.selfloop_edges(G))
            # make iterator for edges
            edge_iter = (Graphs.Edge(e[1]+1, e[2]+1) for e in nx.edges(G))
            # convert to Julia graph
            Gj = Graphs.SimpleDiGraphFromIterator(edge_iter)
            # check if vertices are all there
            while nv(Gj) ≠ length(d_in)
                add_vertex!(Gj)
            end
            # store the result
            res[i] = Gj
            # update progress bar
            next!(P)
        end

        return  res
    end

    networkx_directed_configuration_model(G::SimpleDiGraph, N::Int; kwargs...) = networkx_directed_configuration_model(Graphs.indegree(G), Graphs.outdegree(G), N; kwargs...)
    
    """
        igraph_directed_configuration_model(d_in::Vector, d_out::Vector, N::Int; method::Symbol=:configuration, kwargs...)
        igraph_directed_configuration_model(G::SimpleDiGraph, N::Int; method::Symbol=:configuration, kwargs...)

    From a given degree sequence `d` or graph `G`, generate `N` random graphs using [igraph's configuration model](https://igraph.org/c/doc/igraph-Generators.html#igraph_degree_sequence_game).

    For downstream tasks, parallel edges and self-loops will be removed from the resulting graphs.
    # Arguments:
    - `d::Vector`: the degree sequence to use
    - `method::Symbol`: the method to use for generating the graph. The following methods are available:
        - `:configuration`: simple generator that implements the stub-matching configuration model. It may generate self-loops and multiple edges. This method does not sample multigraphs uniformly, but it can be used to implement uniform sampling for simple graphs by rejecting any result that is non-simple (i.e. contains loops or multi-edges).
        - `:fast_heur_simple`: similar to `:configuration` but avoids the generation of multiple and loop edges at the expense of increased time complexity. The method will re-start the generation every time it gets stuck in a configuration where it is not possible to insert any more edges without creating loops or multiple edges, and there is no upper bound on the number of iterations, but it will succeed eventually if the input degree sequence is graphical and throw an exception if the input degree sequence is not graphical. This method does not sample simple graphs uniformly.
        - `:configuration_simple`: similar to `:configuration` but rejects generated graphs if they are not simple. This method samples simple graphs uniformly.
        - `:edge_switching_simple`: an MCMC sampler based on degree-preserving edge switches. It generates simple undirected or directed graphs. The algorithm uses L{Graph.Realize_Degree_Sequence()} to construct an initial graph, then rewires it using L{Graph.rewire()}.
    """
    function igraph_directed_configuration_model(d_in::Vector, d_out, N::Int; method::Symbol=:configuration, kwargs...)
        length(d_in) == length(d_out) || throw(ArgumentError("d_in and d_out must have the same length"))
        # python dependency
        ig = pyimport("igraph")
        # progress bar
        P = Progress(N; dt=1.0, desc="igraph_configuration_model_$(method) sampling on worker $(myid()):")
        # pre-allocate
        res = Vector{SimpleDiGraph}(undef, N)
        for i in 1:N
            G = ig.Graph.Degree_Sequence(d_out, d_in, method=method)
            # remove self-loops and parallel edges
            G.simplify(multiple=true, loops=true, combine_edges=nothing)
            # make iterator for edges
            edge_iter = (Graphs.Edge(e.source + 1, e.target + 1) for e in G.es)
            # convert to Julia graph
            Gj = Graphs.SimpleDiGraphFromIterator(edge_iter)
            # check if vertices are all there
            while nv(Gj) ≠ length(d_in)
                add_vertex!(Gj)
            end
            # store the result
            res[i] = Gj
            # update progress bar
            next!(P)
        end

        return res
    end

    igraph_directed_configuration_model(G::SimpleDiGraph, N::Int; method::Symbol=:configuration, kwargs...) = igraph_directed_configuration_model(Graphs.indegree(G), Graphs.outdegree(G), N; method=method, kwargs...)


    """
        networkit_directed_curveball(G::SimpleDiGraph, N::Int; kwargs...) 

    From a given graph `G`, generate `N` random graphs using [networkit's curveball model](https://networkit.github.io/dev-docs/notebooks/Randomization.html).

    """
    function networkit_directed_curveball(G::SimpleDiGraph, N::Int; kwargs...) 
        # python dependency
        nk = pyimport("networkit")
        # progress bar
        P = Progress(N; dt=1.0, desc="networkit_curveball sampling on worker $(myid()):")
        # convert the graph to networkit
        Gnk = nk.Graph(nv(G), directed=true, weighted=false)
        for e in edges(G)
            Gnk.addEdge(e.src - 1, e.dst - 1) # networkit uses 0-based indexing
        end
        # pre-allocate result
        res = Vector{SimpleDiGraph}(undef, N)
        # do the work
        for i in 1:N
            # Initialize algorithm 
            globalCurve = nk.randomization.GlobalCurveball(Gnk)
            # Run algorithm 
            globalCurve.run()
            # Get result
            G_rand = globalCurve.getGraph()
            # reshape to Julia graph
            edge_iter = (Graphs.Edge(e[1] + 1, e[2] + 1) for e in G_rand.iterEdges())
            res[i] = Graphs.SimpleDiGraphFromIterator(edge_iter)
            # check if vertices are all there
            while nv(res[i]) ≠ nv(G)
                add_vertex!(res[i])
            end
            next!(P)
        end

        return res
    end


    """
        networkit_directed_edge_switching_markov_chain(G::SimpleDiGraph, N::Int)

    From a given graph `G`, generate `N` random graphs using [networkit's edge switching markov chain model](https://networkit.github.io/dev-docs/notebooks/Randomization.html).
    """
    function networkit_directed_edge_switching_markov_chain(G::SimpleDiGraph, N::Int; kwargs...)
        # python dependency
        nk = pyimport("networkit")
        # progress bar
        P = Progress(N; dt=1.0, desc="networkit_edge_switching_markov_chain sampling on worker $(myid()):")
        # convert the graph to networkit
        Gnk = nk.Graph(nv(G), directed=true, weighted=false)
        for e in edges(G)
            Gnk.addEdge(e.src - 1, e.dst - 1) # networkit uses 0-based indexing
        end
        # pre-allocate result
        res = Vector{SimpleDiGraph}(undef, N)
        # do the work
        for i in 1:N
            # Initialize algorithm 
            esr = nk.randomization.EdgeSwitching(Gnk)
            # Run algorithm 
            esr.run()
            # Get result
            G_rand = esr.getGraph()
            # reshape to Julia graph
            edge_iter = (Graphs.Edge(e[1] + 1, e[2] + 1) for e in G_rand.iterEdges())
            res[i] = Graphs.SimpleDiGraphFromIterator(edge_iter)
            # check if vertices are all there
            while nv(res[i]) ≠ nv(G)
                add_vertex!(res[i])
            end
            next!(P)
        end

        return res
    end


    """
        graph_tool_directed_configuration_model(d_in::Vector, d_out::Vector, N::Int; method=:configuration)
        graph_tool_directed_configuration_model(G::SimpleDiGraph, N::Int; method=:configuration)

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
    function graph_tool_directed_configuration_model(d_in::Vector, d_out::Vector, N::Int; method=:configuration, kwargs...)
        # python wrapper for graph-tool
        py"""
        from graph_tool import all as gt

        def configuration_model_graph_tool(d, model="configuration"):
            g = gt.random_graph(len(d), lambda x : d[x], block_membership=None, directed=True, model=model)

            return g.iter_edges()

        """
        # progress bar
        P = Progress(N; dt=1.0, desc="graph_tool_model_$(method) sampling on worker $(myid()):")
        # pre-allocate
        res = Vector{SimpleDiGraph}(undef, N)
        # do the work
        for i in 1:N
            # make iterator for edges
            edge_iter = (Graphs.Edge(e[1] + 1, e[2] + 1) for e in py"configuration_model_graph_tool"(collect(zip(d_in, d_out)), model=method))
            # convert to Julia graph
            Gj = Graphs.SimpleDiGraphFromIterator(edge_iter)
            # check if vertices are all there
            while nv(Gj) ≠ length(d_in)
                add_vertex!(Gj)
            end
            res[i] = Gj
            next!(P)
        end

        return res
    end

    graph_tool_directed_configuration_model(G::SimpleDiGraph, N::Int; method=:configuration, kwargs...) = graph_tool_directed_configuration_model(Graphs.indegree(G), Graphs.outdegree(G),  N; method=method, kwargs...)

    """
        MaxEntropyGraphs_DBCM(d::Vector, N::Int; kwargs...)

        From a given degree sequence `d` or graph `G`, generate `N` random graphs using MaxEntropyGraphs.jl's DBCM model.
    """
    function MaxEntropyGraphs_DBCM(d_in::Vector, d_out::Vector, N::Int; kwargs...)
        model = DBCM(d_in=d_in, d_out=d_out)

        solve_model!(model)
        
        return rand(model, N)
    end

    MaxEntropyGraphs_DBCM(G::SimpleDiGraph, N::Int; kwargs...) = MaxEntropyGraphs_DBCM(Graphs.indegree(G), Graphs.outdegree(G), N; kwargs...)

    function NEMtropy_DBCM(G::SimpleDiGraph, N::Int; netname::String, kwargs...)
        # get degree sequences and write them to a file
        d_in = Graphs.indegree(G)
        d_out = Graphs.outdegree(G)

        # generate the python script to generate the samples fast
        content = """
        ## Run the script from its location
        ## This script is used to generate random biparte graphs from a given degree sequences
        ## using the NEMtropy package.
        ## The random graphs are saved in the OUT_FOLDER.
        ## For some reason, this is MUCH faster than invoking the NEMtropy package through PyCall in Julia.

        import NEMtropy as nem
        import numpy as np
        import os

        print('working in: {}'.format(os.getcwd()))
        # set var
        OUT_FOLDER = "./sample/directed_$(netname)/"
        N_SAMPLES = $(N)
        # check if folder exists and create it if not
        if not os.path.exists(OUT_FOLDER):
            os.makedirs(OUT_FOLDER)

        d = np.array($(vcat(d_out, d_in)))
        # model creation
        m = nem.DirectedGraph(degree_sequence=d)
        m.solve_tool(model="dcm_exp", method="fixed-point", initial_guess="degrees_minor")
        # generate the samples
        m.ensemble_sampler($(N), output_dir=OUT_FOLDER)
        
        """

        open("./sample/directed_$(netname).py", "w") do file
            write(file, content)
        end

        # run the python script
        #run(`/Users/bartdeclerck/miniconda3/envs/maxentropygraphsbis/bin/python ./sample/directed_$(netname).py`)

        # Load up the NEMtropy graphs from disk
        GG = Vector{SimpleDiGraph}()
        for filename in readdir("./sample/directed_$(netname)/")
            open("./sample/directed_$(netname)/$filename") do file
                e = Vector{Graphs.SimpleEdge}()
                for line in eachline(file)
                    # parse the line
                    u, v = split(line, " ")
                    # add the edge
                    push!(e, Graphs.SimpleEdge(parse(Int, u) + 1, parse(Int, v) + 1)) # +1 because of it comming from python
                end
                lg = Graphs.SimpleDiGraphFromIterator(e)
                while nv(lg) < nv(G)
                    add_vertex!(lg)
                end
                push!(GG, lg)
            end
        end

        return GG
    end

    """
        igraph_degree_sequence_game_directed(d_out, d_in, N; kwargs...)
        igraph_degree_sequence_game_directed(G::SimpleDigraph, N; kwargs...)

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
    function igraph_degree_sequence_game_directed(d_out::Vector, d_in::Vector,  N::Int; method::Symbol=:IGRAPH_DEGSEQ_CONFIGURATION, 
                                            header_folder::String="/opt/homebrew/Cellar/igraph/0.10.12/include/igraph",
                                            library_folder::String="/opt/homebrew/Cellar/igraph/0.10.12/lib",
                                            temp_folder=joinpath(pwd(), workingfolder), 
                                            keepraw::Bool=false, kwargs...)
        
        @info "Running igraph_degree_sequence_game with method: $(method)"
        # preallocate output vector
        G_res = Vector{SimpleDiGraph}(undef, N)

        # convert the degree sequence to a string
        d_out_str = join(d_out, ", ")
        d_in_str  = join(d_in,  ", ")

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


        int generatorfun(const igraph_vector_int_t *d_out, const igraph_vector_int_t *d_in, int n) {
            // loop n times
            for (int i = 0; i < n; i++) {
                // initialize the graph
                igraph_t graph;
                int result;
            
                // run the game
                result = igraph_degree_sequence_game(&graph, d_out, d_in, $(method));

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
            igraph_vector_int_t d_out;
            igraph_vector_int_t d_in;

            // initialize the degree sequence vector
            igraph_vector_int_init(&d_out, $(length(d_out)));
            igraph_vector_int_init(&d_in, $(length(d_in)));

            // set the initial values of the out-degree sequence vector
            int initial_out_values[] = {$(d_out_str)};
            int initial_in_values[]  = {$(d_in_str)};
            for (int i = 0; i < $(length(d_out)); i++) {
                VECTOR(d_out)[i] = initial_out_values[i];
                VECTOR(d_in)[i]  = initial_in_values[i];
            }

            // run the generator function for $(N) iterations
            generatorfun(&d_out, &d_in, $(N));

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
            edge_iter = (Graphs.Edge(parse(Int,split(e)[1])+1, parse(Int,split(e)[2])+1) for e in res)
            # graph from iterator
            G = SimpleDiGraphFromIterator(edge_iter)
    
            # check nodecount 
            while nv(G) < length(d_out)
                add_vertex!(G)
            end
            
            # add to results
            G_res[i+1] = G
    
            # close and remove the file
            keepraw ? nothing : rm(fname)
        end
    
        return G_res
    end

    
    igraph_degree_sequence_game_directed(G::SimpleDiGraph, N::Int; method::Symbol=:IGRAPH_DEGSEQ_CONFIGURATION, 
                                            header_folder::String="/opt/homebrew/Cellar/igraph/0.10.12/include/igraph",
                                            library_folder::String="/opt/homebrew/Cellar/igraph/0.10.12/lib",
                                            temp_folder=joinpath(pwd(), workingfolder), 
                                            keepraw::Bool=false, kwargs...)  = igraph_degree_sequence_game_directed(Graphs.outdegree(G), Graphs.indegree(G) , N, method=method, header_folder=header_folder, library_folder=library_folder, temp_folder=temp_folder, keepraw=keepraw, kwargs...)



    """
        igraph_static_fitness_game_directed(d_out::Vector, d_in::Vector, E::Int, N::Int; loops::Bool=false, multiple::Bool=false, kwargs...)
        igraph_static_fitness_game_directed(G::SimpleDiGraph, E::Int, N::Int; loops::Bool=false, multiple::Bool=false, kwargs...)

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
    function igraph_static_fitness_game_directed(d_out::Vector, d_in::Vector, E::Int, N::Int; loops::Bool=false, multiple::Bool=false,
                                            header_folder::String="/opt/homebrew/Cellar/igraph/0.10.12/include/igraph",
                                            library_folder::String="/opt/homebrew/Cellar/igraph/0.10.12/lib",
                                            temp_folder=joinpath(pwd(), workingfolder), 
                                            keepraw::Bool=false, kwargs...)

        @info "Running igraph_static_fitness_game"

        # preallocate output vector
        G_res = Vector{SimpleDiGraph}(undef, N)

        # convert the degree sequence to a string
        d_out_str = join(d_out, ", ")
        d_in_str  = join(d_in,  ", ")

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


        int generatorfun(const igraph_vector_t *d_out, const igraph_vector_t *d_in, int e, int n) {
            // loop n times
            for (int i = 0; i < n; i++) {
                // initialize the graph
                igraph_t graph;
                int result;
            
                // run the game
                result = igraph_static_fitness_game(&graph, e, d_out, d_in, $(loops), $(multiple));

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
            igraph_vector_t d_out;
            igraph_vector_t d_in;

            // initialize the degree sequence vector
            igraph_vector_init(&d_out, $(length(d_out)));
            igraph_vector_init(&d_in,  $(length(d_in)));

            // set the initial values of the degree sequence vector
            int initial_out_values[] = {$(d_out_str)};
            int initial_in_values[]  = {$(d_in_str)};
            for (int i = 0; i < $(length(d_out)); i++) {
                VECTOR(d_out)[i] = initial_out_values[i];
                VECTOR(d_in)[i]  = initial_in_values[i];
            }

            // run the generator function for $(N) iterations
            generatorfun(&d_out, &d_in, $(E), $(N));

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
            edge_iter = (Graphs.Edge(parse(Int,split(e)[1])+1, parse(Int,split(e)[2])+1) for e in res)
            # graph from iterator
            G = SimpleDiGraphFromIterator(edge_iter)
    
            # check nodecount 
            while nv(G) < length(d_out)
                add_vertex!(G)
            end
            
            # add to results
            G_res[i+1] = G
    
            # close and remove the file
            keepraw ? nothing : rm(fname)
        end
    
        return G_res
    end


    igraph_static_fitness_game_directed(G::SimpleDiGraph, N::Int; loops::Bool=false, multiple::Bool=false,
                                    header_folder::String="/opt/homebrew/Cellar/igraph/0.10.12/include/igraph",
                                    library_folder::String="/opt/homebrew/Cellar/igraph/0.10.12/lib",
                                    temp_folder=joinpath(pwd(), workingfolder), 
                                    keepraw::Bool=false, kwargs...) = igraph_static_fitness_game_directed(Graphs.outdegree(G), Graphs.indegree(G), Graphs.ne(G), N; loops=loops, multiple=multiple, header_folder=header_folder, library_folder=library_folder, temp_folder=temp_folder, keepraw=keepraw, kwargs...)

                                    
    
    

    
    
    """
        graph_tool_stub_matching_directed(G::SimpleGraph, N::Int; kwargs...)

    Generate `N` random directed graphs with the same degree sequence as `G` using the stub matching approach from the graph-tool library.
    This solution was provided by Tiago Peixoto himself, the author of the graph-tool library.

    # See also:
    https://graph-tool.skewed.de/static/doc/autosummary/graph_tool.generation.generate_sbm.html
    """
    function graph_tool_stub_matching_directed(G::SimpleDiGraph, N::Int; kwargs...)
        # Generate the edge list iterator
        edge_iter = ((e.src-1, e.dst-1) for e in edges(G))

        # python wrapper for graph-tool
        py"""
        from graph_tool import Graph
        import numpy as np
        from graph_tool.generation import generate_sbm
        # convert to graph-tool graph
        g = Graph($(edge_iter), directed=True)
        # sanity check
        assert g.num_vertices() == $(Graphs.nv(G))
        assert g.num_edges() == $(Graphs.ne(G))
        assert (g.get_out_degrees(g.get_vertices()) == $(Graphs.outdegree(G))).all()
        assert (g.get_in_degrees(g.get_vertices()) == $(Graphs.indegree(G))).all()
        # block membership
        b = [0 for _ in range($(Graphs.nv(G)))] 
        # inter-block edges (all together)
        probs = np.matrix([[$(Graphs.ne(G))]])
        
        # generate random graph edge list iterator
        def get_rand_graph():
            return generate_sbm(b, probs, $(Graphs.outdegree(G)), $(Graphs.indegree(G)), directed=True, micro_ers=True, micro_degs=True).iter_edges()    
        """

        # progress bar
        P = Progress(N; dt=1.0, desc="graph_tool_stub_matching sampling on worker $(myid()):")
        # pre-allocate
        res = Vector{SimpleDiGraph}(undef, N)
        # do the work
        for i in 1:N
            # make iterator for edges
            edge_iter = (Graphs.Edge(e[1] + 1, e[2] + 1) for e in py"get_rand_graph"())
            # convert to Julia graph
            Gj = Graphs.SimpleDiGraphFromIterator(edge_iter)
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
        graph_tool_chunglu_directed(G::SimpleDiGraph, N::Int; kwargs...)

    Generate `N` random graphs with the same degree sequence as `G` using the Chung-Lu approach from the graph-tool library.
    This solution was provided by Tiago Peixoto himself, the author of the graph-tool library.

    # See also:
    https://graph-tool.skewed.de/static/doc/autosummary/graph_tool.generation.generate_maxent_sbm.html
    """
    function graph_tool_chunglu_directed(G::SimpleDiGraph, N::Int; kwargs...)
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
        g = Graph($(edge_iter), directed=True)
        # sanity checks
        assert g.num_vertices() == $(Graphs.nv(G))
        assert g.num_edges() == $(Graphs.ne(G))
        assert (g.get_out_degrees(g.get_vertices()) == $(Graphs.outdegree(G))).all()
        assert (g.get_in_degrees(g.get_vertices())  == $(Graphs.indegree(G))).all()

        ## setup
        # block membership
        b = [0 for _ in range($(Graphs.nv(G)))] 
        # inter-block edges (all together)
        probs = np.matrix([[$(Graphs.ne(G))]])
        # fugacities
        out_degs = $(Graphs.outdegree(G))
        in_degs  = $(Graphs.indegree(G))
        mrs, out_theta, in_theta = solve_sbm_fugacities(b, probs, out_degs=out_degs, in_degs=in_degs)

        # generate random graph edge list iterator
        def get_rand_graph():
            return generate_maxent_sbm(b, mrs, out_theta, in_theta, directed=True).iter_edges()  # , multigraph=False
        
        """

        # progress bar
        P = Progress(N; dt=1.0, desc="graph_tool_chunglu sampling on worker $(myid()):")
        # pre-allocate
        res = Vector{SimpleDiGraph}(undef, N)
        # do the work
        for i in 1:N
            # make iterator for edges
            edge_iter = (Graphs.Edge(e[1] + 1, e[2] + 1) for e in py"get_rand_graph"())
            # convert to Julia graph
            Gj = Graphs.SimpleDiGraphFromIterator(edge_iter)
            # check if vertices are all there
            while nv(Gj) ≠ Graphs.nv(G)
                add_vertex!(Gj)
            end
            res[i] = Gj
            next!(P)
        end
        return res
    end
    
    ## Exports
    export
        networkx_directed_configuration_model,
        MaxEntropyGraphs_DBCM,
        igraph_directed_configuration_model,
        networkit_directed_curveball,
        networkit_directed_edge_switching_markov_chain,
        graph_tool_directed_configuration_model,
        graph_tool_stub_matching_directed,
        graph_tool_chunglu_directed,
        igraph_degree_sequence_game_directed,
        igraph_static_fitness_game_directed
end