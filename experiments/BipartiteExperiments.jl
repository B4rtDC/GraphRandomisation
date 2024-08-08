begin
    using Pkg
    cd(joinpath(@__DIR__,".."))
    @info "working in $(pwd())"
    Pkg.activate(pwd())

    using DataFrames
    using CSV
    using JLD2
    using Distributed

    using Plots
    using LaTeXStrings
    using Measures
    using Dates
    import StatsBase: mean, countmap
    using Graphs
    using MaxEntropyGraphs
    using Distributions
    
    ## variables for the study
    # add the workers we need (limited by default to the number logical CPU cores in the machine)
    addprocs(Sys.CPU_THREADS)

    # number of samples per randomisation method/algorithm
    const N_samples = 10

    # conda environment name (required for running the python scripts properly, using hte right environment)
    CONDA_ENV_NAME = "graphrandomisation"
    
    nothing 
end

# Load AnalysisModules on workers (required for distributed graph generation)
@everywhere include(joinpath(pwd(), "src","AnalysisModules.jl"))

#  ____ ____  ___ __  __ _____    ____ ____      _    ____  _   _ 
# / ___|  _ \|_ _|  \/  | ____|  / ___|  _ \    / \  |  _ \| | | |
#| |   | |_) || || |\/| |  _|   | |  _| |_) |  / _ \ | |_) | |_| |
#| |___|  _ < | || |  | | |___  | |_| |  _ <  / ___ \|  __/|  _  |
# \____|_| \_\___|_|  |_|_____|  \____|_| \_\/_/   \_\_|   |_| |_|
                                                                



# Graph to be used for the analysis
@info "$(now()) - Loading the original crime graph"
begin # Network generation and data loading
    e = Vector{Graphs.SimpleEdge}()
    open("./experiments/data/moreno_crime/out.moreno_crime_crime") do file
        NE, NV_p, NV_c = 0, 0, 0
        for line in eachline(file)
            if line[1] == '%'
                sp = split(line, " ")
                try
                    NE = parse(Int, sp[2])
                    NV_p = parse(Int, sp[3])
                    NV_c = parse(Int, sp[4])
                catch
                    continue
                end
            else
                # parse the line
                u, v = split(line, " ")
                # add the edge
                push!(e, Graphs.SimpleEdge(parse(Int, u), parse(Int, v) + NV_p))
            end
        end
    end
    G = Graphs.SimpleGraphFromIterator(e)
    # projections of the graph
    G_bot = MaxEntropyGraphs.project(G; layer=:bottom, method=:weighted)
    G_top = MaxEntropyGraphs.project(G; layer=:top,    method=:weighted)
    # degree sequence per layer
    da = Graphs.degree(G, findall(Graphs.bipartite_map(G) .== 1))
    db = Graphs.degree(G, findall(Graphs.bipartite_map(G) .== 2))
    # some quality checks
    @assert !Graphs.has_self_loops(G)
    @assert is_bipartite(G)
    @assert nv(G) == 1380
    @assert ne(G) == 1476
    @assert sum(Graphs.bipartite_map(G) .== 1) == 829
    @assert sum(Graphs.bipartite_map(G) .== 2) == 551

    # OK!

    nothing
end


# Generate the dataset (using distributed computing)
# Define the samplers and their arguments (descriptor, sampler, kwargs)
tasks = [   ("networkx_bipartite_configuration_model",  AnalysisModules.networkx_bipartite_configuration_model, nothing);
            ("bipartite_curveball",                     AnalysisModules.bipartite_curveball,                    nothing);
            ("bipartite_chung_lu_fast",                 AnalysisModules.bipartite_chung_lu_fast,                nothing);
            ("NEMtropy_BiCM",                           AnalysisModules.NEMtropy_BiCM,                          Dict(:netname =>"crime", :conda_env_name => CONDA_ENV_NAME));
            ("MaxEntropyGraphs_BiCM",                   AnalysisModules.MaxEntropyGraphs_BiCM,                  nothing)

]

# Run the random graph generation
res = pmap(x -> (x[1] => x[2](G, N_samples; (isnothing(x[3]) ? Dict() : x[3])...)), tasks);


# Transform the results into a DataFrame
df = DataFrame( :randomisation_method => vcat([repeat([sample.first], length(sample.second)) for sample in res]...),
                :graph                => vcat([sample.second for sample in res]...))


# Compute some metrics
begin
    # compute the KL divergence
    transform!(df, :graph => ByRow(g -> AnalysisModules.KL_divergence(g, G, α=1e-5))   => :KL_divergence)
    # compute the number of squares
    transform!(df, :graph => ByRow(g -> MaxEntropyGraphs.squares(g))  => :squares)

    # helper function for projections
    """
        validated_projection(G::T, GS::Vector{T}; α=0.05) where T<:AbstractGraph

    Given a graph `G` and a set of graphs `GS`, this function computes the p-values for each edge in `G` and corrects 
    them for multiple testing using the Benjamini-Hochberg method. 
    The function returns the edges that are significant at level `α`.
    """
    function validated_projection(G::T, S::Vector{T}; α=0.05) where T<:AbstractGraph
        # initialise p-values
        p = zeros(length(edges(G)))
        # compute p-values for each edge
        for (i,e) in enumerate(edges(G))
            e_w   = e.weight
            e_w_s = MaxEntropyGraphs.SimpleWeightedGraphs.get_weight.(S, e.src, e.dst)
            #e_w_s = map(s -> s.weights[e.src, e.dst], GS)
            p[i] = AnalysisModules.empirical_pvalue(e_w_s, e_w)
        end
        # correct p-values for multiple testing
        p_corrected = MaxEntropyGraphs.MultipleTesting.adjust(p, MaxEntropyGraphs.MultipleTesting.BenjaminiHochberg())
        # get significant edges
        validated_edges = filter(x -> x[1] < α, collect(zip(p_corrected, edges(G))))

        return validated_edges
    end

    # obtain the projected graphs
    @info "$(now()) - Projecting the original graph"
    G_bot = MaxEntropyGraphs.project(G; layer=:bottom, method=:weighted)
    G_top = MaxEntropyGraphs.project(G; layer=:top, method=:weighted)

    # compute projection for each graph
    membership = Graphs.bipartite_map(G)
    bottom = findall(membership .== 1)
    top = findall(membership .== 2)

    @info "$(now()) - Computing the bottom projections"
    df[!,:proj_bot] = pmap(g -> MaxEntropyGraphs.project(g, membership, bottom, top; layer=:bottom, method=:weighted, skipchecks=true), df[!, :graph])
    @info "$(now()) - Computing the top projections"
    df[!,:proj_top] = pmap(g -> MaxEntropyGraphs.project(g, membership, bottom, top; layer=:top,    method=:weighted, skipchecks=true), df[!, :graph])


    # determine the projections of the random graphs
    resdf = DataFrame(:method => unique(df.randomisation_method))
    for (graph, column) in [(G_bot, :proj_bot), (G_top, :proj_top)]
        counts = []
        @info "Validating the projections for the $column column (original edges: $(length(edges(graph)))"
        for method in unique(df.randomisation_method)
            S = filter(row -> row.randomisation_method == method, df)[!,column]
            res = validated_projection(graph, S)
            push!(counts, length(res)/ Graphs.ne(graph))
            # write out the results as a message
            @info """Method: $method
            - Number of significant edges: $(length(res))
            
            """   
        end
        resdf[!, Symbol("$(column)_edges")] = counts
    end
    
end

# Write the dataframe to disk
save("./results/Bipartite_crime.jld2", "df", df)


# Different plots
begin
    # KL-divergence plot
    begin
        @info "$(now()) - Making a plot of the KL divergence"
        # make the plot
        p = AnalysisModules.highlevelplot(df, :KL_divergence, ylabel="Kullback-Leibler divergence")
        plot!(p, tickfontsize=18, bottom_margin=20mm, left_margin=12mm, ylabelfontsize=20, size=(1000,400))
        plot!(p, tickfontsize=13, labelfontsize=17, bottom_margin=10mm, left_margin=7mm, right_margin=5mm, top_margin=4mm,size=(1200,400),legendfontsize=11)
        savefig(p, "./plots/BiCM_crime_KL.pdf")
        p
    end

    # Number of squares plot
    begin
        p = AnalysisModules.highlevelplot(df, :squares, title="", X⁺=MaxEntropyGraphs.squares(G),
        ylabel="Number of squares", left_margin=8mm, tickfontsize=13, labelfontsize=14)
        plot!(p, tickfontsize=13, labelfontsize=17, bottom_margin=10mm, left_margin=7mm, right_margin=5mm, top_margin=4mm, size=(1200,400),legendfontsize=11)
        savefig(p, "./plots/BiCM_crime_squares.pdf")
        p
    end

    # Ratio of significant edges plot
    begin
        @info "Illustrating the project results"
        p = plot()
        markers = [:circle, :diamond, :rect, :star5, :hexagon, :octagon, :star4, :star3, :star6]
        i = 1
        for row in eachrow(resdf)
            scatter!(p,[1; 2], [row.proj_bot_edges; row.proj_top_edges] , label=replace(AnalysisModules.AnalysisHelpers.methodmapper[row.method],"\n"=>" "), marker= markers[i], markeralpha=0.5, markersize=6)
            i+=1
        end
        xticks!([1, 2], ["Bottom layer", "Top layer"])
        xlims!(0.5, 2.5)
        ylims!(0.0, 1)
        ylabel!("Ratio of significant edges")
        xlabel!(" ")
        plot!(p, tickfontsize=11, labelfontsize=14, bottom_margin=5mm, left_margin=5mm, right_margin=5mm, size=(500,400),legendfontsize=11)
    
        savefig(p, "./plots/BiCM_crime_projections.pdf")
        p
    end

    # Ratio of significant edges in function of the number of curveball trades plot
    begin 
        # computation part
        begin
            membership = Graphs.bipartite_map(G)
            bottom = findall(membership .== 1)
            top = findall(membership .== 2)
            sample_size = 100
            r_bot = Float64[]
            r_top = Float64[]
            N_iter_range = round.(Int, 10 .^ range(3, 4, step=0.1))
            default_N_iter = 5*min(length(bottom),length(top))
            ## insert the default number of iterations between the values to its left and right
            insert!(N_iter_range, findfirst(x -> x > default_N_iter, N_iter_range), default_N_iter)
        
            for N_iters in N_iter_range
                @info "$(now()) - Running the curveball algorithm with $N_iters iterations (sample of $sample_size)"
                SS = AnalysisModules.bipartite_curveball(G, sample_size, n_iterations=N_iters)
                @info "$(now()) - Computing the projections"
                G_bot_curveball = pmap(g -> MaxEntropyGraphs.project(g, membership, bottom, top; layer=:bottom, method=:weighted, skipchecks=true), SS)
                G_top_curveball = pmap(g -> MaxEntropyGraphs.project(g, membership, bottom, top; layer=:top, method=:weighted, skipchecks=true),  SS)
                @info "$(now()) - Validating the projections"
                v_bot = validated_projection(G_bot, G_bot_curveball)
                v_top = validated_projection(G_top, G_top_curveball)
                @info "$(now()) - Number of significant edges in the bottom layer: $(length(v_bot) / Graphs.ne(G_bot))"
                @info "$(now()) - Number of significant edges in the top layer: $(length(v_top) / Graphs.ne(G_top))"
                push!(r_bot, length(v_bot) / Graphs.ne(G_bot))
                push!(r_top, length(v_top) / Graphs.ne(G_top))
            end
        end
        # plotting part
        begin
            p = plot()
            plot!(p, N_iter_range, r_bot, label="Bottom layer", marker=:circle, markeralpha=0.5)
            plot!(p, N_iter_range, r_top, label="Top layer", marker=:diamond, markeralpha=0.5)#, xscale=:log10)
            xlabel!("Number of curveball trades")
            ylabel!("Ratio of significant edges")
            plot!(p, tickfontsize=11, labelfontsize=14, bottom_margin=5mm, left_margin=5mm, right_margin=5mm, size=(500,400),legendfontsize=11)
            xtickpositions = range(1e3, 1e4, length=5)
            ylims!(0, 1)
            
            # add vertical line add default number of iterations
            vline!(p, [2755], label="Default number of trades", linestyle=:dash, color=:black)
            savefig(p, "./plots/BiCM_crime_curveball_projection.pdf")
            p
        end
    end
end


#  ____  _______  ____        _____  ____  _  _______ ____     ____ ____      _    ____  _   _ 
# / ___|| ____\ \/ /\ \      / / _ \|  _ \| |/ / ____|  _ \   / ___|  _ \    / \  |  _ \| | | |
# \___ \|  _|  \  /  \ \ /\ / / | | | |_) | ' /|  _| | |_) | | |  _| |_) |  / _ \ | |_) | |_| |
#  ___) | |___ /  \   \ V  V /| |_| |  _ <| . \| |___|  _ <  | |_| |  _ <  / ___ \|  __/|  _  |
# |____/|_____/_/\_\   \_/\_/  \___/|_| \_\_|\_\_____|_| \_\  \____|_| \_\/_/   \_\_|   |_| |_|



# Graph to be used for the analysis
@info "$(now()) - Loading the original sex workers graph"
begin 
    df = CSV.read("./experiments/data/escorts/data.escorts", DataFrame)
    
    n_b = length(unique(df.buyer))
    n_e = length(unique(df.escort))
    
    # some checks for coherence
    @assert  first(size(df)) == 50632
    @assert n_b == 10106
    @assert n_e == 6624

    # get the ratings for each buyer/escort
    r_b = combine(groupby(df, :buyer), :rating => mean).rating_mean
    r_e = combine(groupby(df, :escort), :rating => mean).rating_mean
    ratings = vcat(r_b, r_e)

    # add the mean rating to the dataframe
    transform!(df, :buyer => ByRow(b -> r_b[b]) => :buyer_rating)
    transform!(df, :escort => ByRow(e -> r_e[e]) => :escort_rating)

    # build the actual graph edge list
    transform!(df, [:buyer, :escort] => ByRow((x,y) -> Edge(x, y + n_b)) => :edgelist)

    # build the graph
    G = SimpleGraphFromIterator(df.edgelist)

    # some checks
    @assert is_bipartite(G)
    @assert !Graphs.has_self_loops(G)
    @assert nv(G) == n_b + n_e
    @assert sum(Graphs.bipartite_map(G) .== 1) == n_b
    @assert sum(Graphs.bipartite_map(G) .== 2) == n_e

    # OK!
    nothing
end

# Generate the dataset (using distributed computing)
# Define the samplers and their arguments (descriptor, sampler, kwargs)
tasks = [   ("networkx_bipartite_configuration_model",  AnalysisModules.networkx_bipartite_configuration_model, nothing);
            ("bipartite_curveball",                     AnalysisModules.bipartite_curveball,                    nothing);
            ("bipartite_chung_lu_fast",                 AnalysisModules.bipartite_chung_lu_fast,                nothing);
            ("NEMtropy_BiCM",                           AnalysisModules.NEMtropy_BiCM,                          Dict(:netname =>"crime", :conda_env_name => CONDA_ENV_NAME));
            ("MaxEntropyGraphs_BiCM",                   AnalysisModules.MaxEntropyGraphs_BiCM,                  nothing)

]

# Run the random graph generation
res = pmap(x -> (x[1] => x[2](G, N_samples; (isnothing(x[3]) ? Dict() : x[3])...)), tasks);


# Transform the results into a DataFrame
df = DataFrame( :randomisation_method => vcat([repeat([sample.first], length(sample.second)) for sample in res]...),
                :graph                => vcat([sample.second for sample in res]...))


# Compute some metrics
begin
    # rating assortativity function
    import Statistics: cor
    rating_assortativity(G, ratings) = cor(vcat([[ratings[e.src] ratings[e.dst]] for e in edges(G)]...), dims=1)[1,2]
    
    # compute rating assortativity
    transform!(df, :graph => ByRow(g -> rating_assortativity(g, ratings))  => :rating_assortativity)

end

# Write the dataframe to disk
save("./results/Bipartite_escorts.jld2", "df", df)

# Different plots
begin
    # Rating assortativity plot
    begin
        p = AnalysisModules.highlevelplot(df, :rating_assortativity, ylabel="Rating assortativity", X⁺=rating_assortativity(G, ratings))
        plot!(p, tickfontsize=13, labelfontsize=17, bottom_margin=10mm, left_margin=7mm, right_margin=5mm, top_margin=4mm, size=(1200,400),legendfontsize=11)
        savefig(p, "./plots/BiCM_escorts_rating_assortativity.pdf")
        p
    end

    # Rating assortativity evluation in function of number of curveball trades plot
    begin
        # compute part
        begin
            ngraphs = N_samples
            mixingops = [10; 100; 10000; 100000; 1000000; 500000]
            res = Matrix{SimpleGraph}(undef, ngraphs, length(mixingops) + 1)
            for (i, N) in enumerate(mixingops)
                @info "Mixing operations: ($i,$N)"
                res[:,i] = AnalysisModules.bipartite_curveball(G, ngraphs, n_iterations = N)
            end

            # add default swaps
            res[:, end] = AnalysisModules.bipartite_curveball(G, ngraphs)
            # establish membership
            membership = bipartite_map(G)
            # counts per partition
            na = sum(membership .== 1)
            nb = sum(membership .== 2)
            n_iterations_default =  5*min(na,nb)
            push!(mixingops, n_iterations_default)
        end
        # plot part
        begin
            using Printf
            p = plot()
            xticklabels = String[]
            for (i,j) in enumerate([1;2;3;4;6;5;7])
                assval = map(g -> rating_assortativity(g, ratings), res[:,j])
                boxplot!(p, fill(i, ngraphs), assval, alpha=0.5, color=palette(:tab10)[1])
                m = floor(log10(mixingops[j]))
                a = mixingops[j] / 10^m
                @info m, a
                #@sprintf(L"%$(round(a))x10^{%$(Int(m))}", Int(a, Int(m))
                push!(xticklabels,  L"%$(Int(round(a)))\times10^{%$(Int(m))}")
            end
            m = floor(log10(n_iterations_default))
            a = n_iterations_default / 10^m
            xticklabels[end] = L"""default
            $(%$(round(a,digits=2))\times10^{%$(round(Int,m))})$"""#, n_iterations_default
            plot!(p, xlabel="Number of trades", ylabel="Rating assortativity", legend=false, size=(1200,400), left_margin=12mm, right_margin=8mm, bottom_margin=20mm)
            plot!(p, xticks=(collect(1:length(xticklabels)), xticklabels), tickfontsize=18, labelfontsize=20)
            plot!(p, tickfontsize=13, labelfontsize=17, bottom_margin=12mm, left_margin=7mm, right_margin=5mm, top_margin=2mm, size=(1200,400),legendfontsize=11)
            savefig(p, "./plots/BiCM_escorts_rating_assortativity_evolution.pdf")
            p
        end
    end
end