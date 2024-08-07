module AnalysisHelpers
    import Statistics: mean, std
    import StatsBase: countmap
    import HypothesisTests: ApproximateTwoSampleKSTest
    using ..Graphs
    using ..PyCall # for external libraries through Python

    using ..Distributions

    # Data libraries
    using ..DataFrames

    # Plotting libraries
    using ..Plots
    using ..StatsPlots
    using ..Measures
    using ..Dates
    

    # Helper constant for the plots
    const methodmapper = Dict(
        "networkx_configuration_model" => "NetworkX\n(configuration\nmodel)",
        "networkx_chung_lu_model" => "NetworkX\n(Chung-Lu\nmodel)",
        "networkx_random_degree_sequence_graph" => "NetworkX\n(random\ndegree\nsequence\ngraph)",
        "igraph_configuration_model" => "igraph\n(configuration\nmodel)",
        "igraph_configuration_model_fast_heur_simple" => "igraph\n(configuration\nmodel\nfast\nheuristic)",
        "igraph_configuration_model_edge_switching_simple" => "igraph\n(configuration\nmodel\nedge\nswitching)",
        "igraph_configuration_model_vl" => "igraph\n(configuration\nmodel\nVL)",
        "igraph_chung_lu_approximation" => "igraph\n(Chung-Lu\napproximation)",
        "networkit_chung_lu_model" => "NetworKit\n(Chung-Lu\nmodel)",
        "networkit_configuration_model_graph" => "NetworKit\n(configuration\nmodel)",
        "networkit_curveball" => "NetworKit\n(curveball)",
        "networkit_edge_switching_markov_chain"=> "NetworKit\n(edge\nswitching)",
        "graphtool_configuration_model"=> "graph-tool\n(LRA)",
        "graphtool_chung_lu"=> "graph-tool\n(Chung-Lu\nmodel)",
        "graphtool_stub_matching"=> "graph-tool\n(stub\nmatching)",
        "NEMtropy_UBCM" => "NEMtropy\n(UBCM)",
        "MaxEntropyGraphs_UBCM" => "MaxEntropy\nGraphs\n(UBCM)",
        "MaxEntropyGraphs_DBCM" => "MaxEntropyGraphs\n(DBCM)",
        "networkx_directed_configuration_model" => "NetworkX\n(directed\nconfiguration\nmodel)",
        "igraph_directed_configuration_model" => "igraph\n(directed\nconfiguration\nmodel)",
        "igraph_directed_configuration_model_edge_switching_simple" => "igraph\n(directed\nconfiguration\nmodel\nedge\nswitching)",
        "networkit_directed_curveball" => "NetworKit\n(directed\ncurveball)",
        "networkit_directed_edge_switching_markov_chain"=> "NetworKit\n(directed\nedge\nswitching)",
        "graphtool_directed_configuration_model" => "graph-tool\n(directed\nconfiguration\nmodel)",
        "networkx_bipartite_configuration_model" => "NetworkX\n(bipartite configuration\nmodel)",
        "bipartite_curveball" => "bipartite\ncurveball",
        "bipartite_chung_lu_fast" => "bipartite\nChung-Lu\n(fast)",
        "NEMtropy_BiCM" => "NEMtropy\n(BiCM)",
        "MaxEntropyGraphs_BiCM" => "MaxEntropyGraphs\n(BiCM)",
        )

    ## Compute the statistical significance, according to different methods
    ## --------------------------------------------------------------------

    """
        zscore(v::Vector, ref::T) where T<:Number

    Compute the z-score of a reference value `ref` in a vector `v`.
    This is done by subtracting the mean of `v` from `ref` and dividing by the standard deviation of `v`.
    Use with caution, especially when `v` is small or if `v` is not normally distributed.
    """
    zscore(v::V, ref::T) where {V<:AbstractVector,T<:Number} = (ref - mean(v)) / std(v) 

    """
        empirical_pvalue(v::Vector, ref::T) where T<:Number

    Compute the empirical p-value of a reference value `ref` in a vector `v`. 
    This is done by counting the number of values in `v` that are greater than or equal to `ref` and dividing by the length of `v`.

    An offset of 1 is added to both the numerator and denominator following 
    "A Note on the Calculation of Empirical P Values from Monte Carlo Procedures"
    """
    empirical_pvalue(v::V, ref::T) where {V<:AbstractVector,T<:Number} = (1 + sum(v .≥ ref)) / (1 + length(v))

    
    """
        KL_divergence(p::Vector, q::Vector; α=0.0)
        KL_divergence(G::T, H::T; α=0.0, method::Function=Graphs.degree) where T<:AbstractGraph

    Compute the Kullback-Leibler (KL) divergence between the distributions of two observed vectors `p` and `q` with Laplace smoothing.
    
    If we use P and Q to denote two probability distributions of p and q respectively,  
    then the Kullback-Leibler divergence Dₖₗ(P ‖ Q)can be interpreted as the average difference 
    of the number of bits required for encoding samples of P using a code optimized for Q rather than one optimized for P.
    Usually, P represents the data, the observations, or a measured probability distribution. 
    Q represents instead a theory, a model, a description or an approximation of P.
    α is the smoothing parameter (default is 0.0, which corresponds to no smoothing and can lead to NaNs in the result, as a guideline use (10 k_{max}^{-1})
    """
    function KL_divergence(p::T, q::T; α=0.0) where T<:AbstractVector
        # obtain the distributions for both as a dictionary value => absolute frequency
        P = countmap(p)
        Q = countmap(q)
        
        # apply Laplace smoothing
        all_keys = union(keys(P), keys(Q))
        P_smoothed = Dict(deg => get(P, deg, zero(Float64)) + α for deg in all_keys)
        Q_smoothed = Dict(deg => get(Q, deg, zero(Float64)) + α for deg in all_keys)
        
        # compute the total count after smoothing
        total_P = sum(values(P_smoothed))
        total_Q = sum(values(Q_smoothed))
        
        # compute the KL-divergence with smoothed probabilities
        res = zero(Float64)
        for key in all_keys
            p = P_smoothed[key] / total_P
            q = Q_smoothed[key] / total_Q
            if !iszero(p)
                if iszero(q)
                    return Inf
                else
                    res += p * log(p / q)
                end
            end
            #res += p * log(p / q)
        end

        return res
    end

    KL_divergence(G::T, H::T; α=0.0, method::Function=Graphs.degree) where T<:AbstractGraph = KL_divergence(method(G), method(H), α=α)

    """
        KS_metric(p::T, q::T) where T<:AbstractVector
        KS_metric(G::T, H::T, method::Function=Graphs.degree) where T<:AbstractGraph

    Compute the Kolmogorov-Smirnov metric between two between the distributions of two observed vectors `p` and `q`.
    Also has a version for `Graphs`, where we first compute the degree sequence of the graphs and then compute the metric.
    """
    KS_metric(p::T, q::T) where T<:AbstractVector = ApproximateTwoSampleKSTest(p, q).δ
    KS_metric(G::T, H::T, method::Function=Graphs.degree) where T<:Graphs.AbstractGraph = ApproximateTwoSampleKSTest(method(G), method(H)).δ
    

    """
        LP_metric(a::T; b::T; p) where T<:AbstractVector
        KS_metric(G::T, H::T, method::Function=Graphs.degree) where T<:AbstractGraph

    Compute the Lₚ metric between the distributions of two observed vectors `a` and `b`.
    Also has a version for `Graphs`, where we first compute the degree sequence of the graphs and then compute the metric.
    """
    function LP_metric(a::T, b::T; p::Int=2) where T<:AbstractVector 
        # obtain the distributions for both as a dictionary value => absolute frequency
        fa = countmap(a)
        fb = countmap(b)

        # compute the total count
        total_A = sum(values(fa))
        total_B = sum(values(fb))

        # compute the Lp metric
        res = zero(Float64)
        for key in union(keys(fa), keys(fb))
            A = get(fa, key, zero(Float64)) / total_A
            B = get(fb, key, zero(Float64)) / total_B
            res += abs(A - B) ^ p
        end

        return res ^ (1/p)
    end

    LP_metric(G::T, H::T; p::Int=2, method::Function=Graphs.degree) where T<:AbstractGraph = LP_metric(method(G), method(H), p=p)



    function community_size_leiden(G::T) where T<:Graphs.AbstractGraph
        # compute the community structure
        communities = Graphs.community_leiden(G)
        # compute the size of each community
        return length.(communities)
    end

    la = pyimport("leidenalg")
    ig = pyimport("igraph")
    """
        community_size_leiden(G::SimpleGraph)

    Compute the number of communities in a graph `G` using the Leiden algorithm.

    Relies on the `igraph` package and `leidenalg` packages from Python.
    """
    function community_size_leiden(G::Graphs.SimpleGraph)
        return length(la.find_partition(ig.GraphBase.Adjacency(Graphs.adjacency_matrix(G)), la.ModularityVertexPartition).sizes())
    end


    """
        aggregated_zscore_pvalue(df::DataFrame, field::Symbol, X⁺::T; groupfield::Symbol=:randomisation_method) where T<:Number

    Compute the z-score and experimental p-value for a given field `field` in the dataframe `df` with respect to the reference value `X⁺`.
    The associated p-value with respect to the z-value is also computed assuming a normal distribution.
    """
    function aggregated_zscore_pvalue(df::DataFrame, field::Symbol, X⁺::T; groupfield::Symbol=:randomisation_method) where T<:Number
        # group per randomisation method
        gdf = groupby(df, :randomisation_method)
        # compute the z-scores and p-values
        res = combine(gdf, [field] .=> [(x -> zscore(x, X⁺)), 
                                        (x -> empirical_pvalue(x, X⁺))] .=> [Symbol("zscore_$(field)"), Symbol("p_empirical_$(field)")]) 
        # compute infered p-value base on the z-score
        res[!, Symbol("p_infered_$(field)")] = 1. .- cdf.(Normal(), res[!, Symbol("zscore_$(field)")])
        # compute infered z-score based on the empirical p-value
        res[!, Symbol("z_inferrred_$(field)")] = quantile.(Normal(),1. .-  res[!,Symbol("p_empirical_$(field)")])

        return res
    end


    ## Plotting functions
    """
        highlevelplot(df::DataFrame, field::Symbol;
                                groupfield::Symbol=:randomisation_method,
                                X⁺::Union{Nothing, M}=nothing,
                                filename::String,
                                kwargs...)
    
    Make a plot of the field `field` of the DataFrame `df` grouped by `groupfield`.
    Both a boxplot and a violin plot are shown (all in the same color).
    If desired, the keyword argument `X⁺` can be used to add a reference line to the plot.
    The additional keyword arguments are passed to the `plot` function.
    The plot is saved to a file with the filename `\$(field)_\$(Dates.format(now(),"YYYY-mm-dd-HH:MM")).pdf`.

    `groupfield` defaults to `:randomisation_method` and is mapped to a more readable name (see `methodmapper`).
    """
    function highlevelplot(df::DataFrame, field::Symbol;
                            groupfield::Symbol=:randomisation_method,
                            X⁺::Union{Nothing, M}=nothing,
                            plotsettings::Dict=Dict(:size => (1400, 600), 
                                                    :xlabel => "", 
                                                    :legend => false,
                                                    :bottom_margin => 12mm),
                            savefig::Bool=false,
                            filename = "$(field)_$(Dates.format(now(),"YYYY-mm-dd-HH:MM")).pdf",
                            kwargs...) where M <: Number # {T <: AbstractGraph, }
        # group per randomisation method
        gdf = groupby(df, groupfield)
        xtickpos = Vector{Int}(undef, length(gdf))
        xtickstr = Vector{String}(undef, length(gdf))
        # initialise the plot
        p = plot()
        for (i, sdf) in enumerate(gdf)
            # data
            boxplot!(p, fill(i, length(getproperty(sdf,field))), getproperty(sdf,field), alpha = 1,  color=palette(:tab10)[1])
            violin!(p, fill(i, length(getproperty(sdf,field))), getproperty(sdf,field), alpha= 0.5, color=palette(:tab10)[1])
            xtickpos[i] = i
            xtickstr[i] = get(methodmapper, sdf[1, :randomisation_method], sdf[1, :randomisation_method])
        end

        # reference line (if numerical value is given)
        if X⁺ ≠ nothing
            hline!(p, [X⁺], linestyle=:dash, color=:black)
        end

        # setting 
        plot!(p; plotsettings..., xticks=(xtickpos, xtickstr), kwargs...)

        savefig ? savefig(p, filename) : nothing
        
        return p
    end



    # function highlevelhistogram(df::DataFrame, field::Symbol; X⁺::Union{Nothing, M}=nothing,
    #                             plotsettings::Dict=Dict(:size => (1400, 600), 
    #                                                     :legend => false,
    #                                                     :bottom_margin => 12mm,
    #                                                     :left_margin => 12mm)
    #                             ) where M <: Number
    #     # group per randomisation method
    #     gdf = groupby(df, :randomisation_method)

    #     # initialise the plot
    #     p = plot()
    #     for (i, sdf) in enumerate(gdf)
    #         # data
    #         label = replace(methodmapper[sdf.randomisation_method[1]], "\n" =>  " ")
    #         histogram!(p, getproperty(sdf,field), label=label, bins=20, normalize=:pdf, alpha=0.5)
    #         #his(p, fill(i, length(getproperty(sdf,field))), getproperty(sdf,field), alpha = 1,  color=palette(:tab10)[1])
    #         #violin!(p, fill(i, length(getproperty(sdf,field))), getproperty(sdf,field), alpha= 0.5, color=palette(:tab10)[1])
    #         #xtickpos[i] = i
    #         #xtickstr[i] = methodmapper[sdf[1, :randomisation_method]]# replace(sdf[1, :randomisation_method], " " => "\n")
    #     end

    #     # reference line (if numerical value is given)
    #     if X⁺ ≠ nothing
    #         vline!(p, [X⁺], linestyle=:dash, color=:black, label="Observered")
    #     end

    #     # setting 
    #     plot!(p; plotsettings..., legend=:topright)

    #     # labels
    #     xlabel!(p, "$(field)")
    #     ylabel!(p, "Density")
    #     #savefig(p, "$(field)_$(Dates.format(now(),"YYYY-mm-dd-HH:MM")).pdf")
    #     return p
    # end







    export 
        zscore,
        empirical_pvalue,
        KL_divergence,
        KS_metric,
        LP_metric,
        community_size_leiden,
        highlevelplot ,
        aggregated_zscore_pvalue
end