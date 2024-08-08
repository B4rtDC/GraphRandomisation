module BipartiteGraphs
    using ..Graphs
    using ..PyCall # for external libraries through Python
    import ..Distributed: myid
    using ..ProgressMeter
    using ..CSV
    using ..MaxEntropyGraphs

    import StatsBase: sample
    import Random: shuffle!
    """
        networkx_bipartite_configuration_model(G::SimpleGraph, N::Int; kwargs...)

    From a given graph `G`, generate `N` random graphs using [NetworkX's configuration model](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.bipartite.generators.configuration_model.html#networkx.algorithms.bipartite.generators.configuration_model).
    Parallel edges and self-loops will be removed from the resulting graphs. Stub-matching based.

    Returns a vector of `N` graphs to work in Julia. 
    """
    function networkx_bipartite_configuration_model(G::Graphs.SimpleGraph, N::Int; kwargs...)
        # input validation
        Graphs.is_bipartite(G) || ArgumentError("Input graph must be bipartite")
        # membership vector
        membership = bipartite_map(G)
        # degree sequence and subvectors
        d = degree(G)
        d_a = d[membership .== 1]
        d_b = d[membership .== 2]
        reverter = vcat(findall(membership .== 1), findall(membership .== 2))

        # python dependency
        nx = pyimport("networkx")
        # progress bar
        P = Progress(N; dt=1.0, desc="networkx_bipartite_configuration_model sampling on worker $(myid()):")
        # pre-allocate
        res = Vector{SimpleGraph}(undef, N)
        for i in 1:N
            # create the random graph
            g = nx.bipartite.configuration_model(d_a, d_b)
            # convert to Julia graph
            Gj = Graphs.SimpleGraphFromIterator(Graphs.SimpleEdge(reverter[e[1] + 1], reverter[e[2] + 1]) for e in g.edges())
            # check if vertices are all there
            while nv(Gj) ≠ nv(G)
                add_vertex!(Gj)
            end
            # store the result
            res[i] = Gj
            # update progress bar
            next!(P)
        end

        return  res
    end


    """
        bipartite_curveball(G::Graphs.SimpleGraph, N::Int; n_iterations::Union{Int,Nothing}=nothing)

    Generate `N` random graphs based on `G` using the curveball algorithm
    
    This is a direct implementation of the bipartite curveball algorithm based on the supplementary material of https://www.nature.com/articles/ncomms5114

    """
    function bipartite_curveball(G::Graphs.SimpleGraph, N::Int; n_iterations::Union{Int,Nothing}=nothing)
        # establish membership
        membership = bipartite_map(G)
        # counts per partition
        na = sum(membership .== 1)
        nb = sum(membership .== 2)
        n_iterations = isnothing(n_iterations) ? 5*min(na,nb) : n_iterations
        sites = na <= nb ? findall(membership .== 1) : findall(membership .== 2)

        # preallocate
        G̃ = Vector{Graphs.SimpleGraph}(undef, N)
        for i = 1:N
            # fetch neighbors and change to sets
            neighs = Dict(n => neighbors(G, n) for n in sites)
            for _ in 1:n_iterations
                # Pair extraction
                res = sample(sites, 2; replace=false)
                # Compare to identify the set of species occurring in one list but not in the other and vice versa
                inter = intersect(neighs[res[1]], neighs[res[2]])
                diff1 = shuffle!(collect(setdiff(neighs[res[1]], neighs[res[2]])))
                diff2 = shuffle!(collect(setdiff(neighs[res[2]], neighs[res[1]])))
                # to trade
                maxtrades = min(length(diff1), length(diff2))
                if maxtrades == 0
                    continue
                end
                n_trade = rand(1:maxtrades)
                # @info """Pair sites: $(res)
                # neighbors: 
                # - $(neighs[res[1]]) 
                # - $(neighs[res[2]])
                
                # to trade: $(n_trade)
                # """
                # trade
                neighs[res[1]] = vcat(inter, diff2[1:n_trade], diff1[n_trade+1:end])
                neighs[res[2]] = vcat(inter, diff1[1:n_trade], diff2[n_trade+1:end])

                # @info """neighbors post trade :
                # - $(neighs[res[1]])
                # - $(neighs[res[2]])"
                # ""
                
            end
            # generate the graph
            iter = (Graphs.Edge(e, v) for e in keys(neighs) for v in neighs[e] )

            G̃[i] = Graphs.SimpleGraphFromIterator(iter)
        end

        return G̃
    end

    import Distributions: Categorical

    """
        bipartite_chung_lu_fast(G::Graphs.SimpleGraph, N::Int)

    Generate `N` random graphs based on `G` using the fast Chung-Lu algorithm

    This is a direct implementation of the fast bipartite Chung-Lu algorithm of https://arxiv.org/abs/1607.08673
    Multi-edges will be discarded, so the number of edges in the generated graphs may be less than the number of edges in the original graph.
    """
    function bipartite_chung_lu_fast(G::Graphs.SimpleGraph, N::Int)
        # establish membership
        membership = bipartite_map(G)
        # counts per partition
        a_members = findall(membership .== 1)
        na = length(a_members)
        b_members = findall(membership .== 2)
        nb = length(b_members)
        # total number of edges
        m = ne(G)
        # distributions
        da = Categorical(degree(G, a_members) ./ m)
        db = Categorical(degree(G, b_members) ./ m)

        # preallocate
        G̃ = Vector{Graphs.SimpleGraph}(undef, N)
        for i = 1:N
            iter = (Graphs.Edge(a_members[rand(da)], b_members[rand(db)]) for _ in 1:m)
            Grand = Graphs.SimpleGraphFromIterator(iter)
            while nv(Grand) < nv(G)
                add_vertex!(Grand)
            end
            G̃[i] = Grand
        end
        # preallocate
        #G̃ = Vector{Graphs.SimpleGraph}(undef, N)
        return G̃
    end

    """
        NEMtropy_BiCM(G::Graphs.SimpleGraph, N::Int; kwargs...)

    Generate `N` samples from a given graph `G` using NEMtropy's BiCM method.

    The function works by generating a python script that generates the samples and then runs it.
    This is much faster than invoking the NEMtropy package through PyCall in Julia.
    We then load up the samples from disk and return them in the JuliaGraphs format.
    """
    function NEMtropy_BiCM(G::Graphs.SimpleGraph, N::Int; netname::String, conda_env_name::String="graphrandomisation", kwargs...)
        # establish membership
        membership = bipartite_map(G)
        a_members = findall(membership .== 1)
        N_a = length(a_members)
        b_members = findall(membership .== 2)
        N_b = length(b_members)
        # reverter
        reverter = vcat(findall(membership .== 1), findall(membership .== 2))

        # get degree sequence and write to file
        d_a = degree(G, a_members)
        d_b = degree(G, b_members)

        # generate the python script to generate the samples fast.
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
        OUT_FOLDER = "./sample/bipartite_$(netname)/"
        N_SAMPLES = $(N)
        # check if folder exists and create it if not
        if not os.path.exists(OUT_FOLDER):
            os.makedirs(OUT_FOLDER)

        da = np.array($(d_a))
        db = np.array($(d_b))
        # model creation
        m = nem.BipartiteGraph(degree_sequences=(da,db))
        avg_mat = m.get_bicm_matrix()

        for i in range(N_SAMPLES):
            # numpy bi-adjacency matrix
            rand_mat = nem.network_functions.sample_bicm(avg_mat)
            # loop over the matrix and write the non-zero entries to a file
            with open(OUT_FOLDER + "{}.txt".format(i), "w") as f:
                for i in range(rand_mat.shape[0]):
                    for j in range(rand_mat.shape[1]):
                        if rand_mat[i,j] > 0:
                            f.write(str(i) + " " + str(j) + "\\n")


        """

        # write out the graph
        open("./sample/bipartite_$(netname).py", "w") do file
            write(file, content)
        end

        # run the python script (using conda env)
        # - list all conda envs
        candidates = split(read(`conda env list`, String), '\n')
        # find the one matching
        myenvpath = split(candidates[findfirst(occursin.(conda_env_name, candidates))])[end]
        run(`$(myenvpath)/bin/python ./sample/bipartite_$(netname).py`)

        # Generate the samples from the generated python script (can take some time...) - deactivate if already done
        run(`/Users/bartdeclerck/miniconda3/envs/maxentropygraphsbis/bin/python ./sample/bipartite_$(netname).py`)

        # Load up the NEMtropy graphs from disk
        GG = Vector{SimpleGraph}()
        for filename in readdir("./sample/bipartite_$(netname)/")
            open("./sample/bipartite_$(netname)/$filename") do file
                e = Vector{Graphs.SimpleEdge}()
                for line in eachline(file)
                    # parse the line
                    u, v = split(line, " ")
                    # add the edge
                    push!(e, Graphs.SimpleEdge(reverter[parse(Int, u)+1], reverter[parse(Int, v)+1+N_a])) # +1 because of it comming from python
                end
                lg = Graphs.SimpleGraphFromIterator(e)
                while nv(lg) < nv(G)
                    add_vertex!(lg)
                end
                push!(GG, lg)
            end
        end

        # return the samples
        return GG
    end

    function MaxEntropyGraphs_BiCM(G::Graphs.SimpleGraph, N::Int; kwargs...)
        # model definition
        model = MaxEntropyGraphs.BiCM(G)
        # model fitting
        solve_model!(model)

        return rand(model, N)
    end


    export 
    networkx_bipartite_configuration_model, 
    MaxEntropyGraphs_BiCM,
    bipartite_curveball,
    NEMtropy_BiCM,
    bipartite_chung_lu_fast

end