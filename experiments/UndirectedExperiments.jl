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
    
    nothing 
end


# Load AnalysisModules on workers (required for distributed graph generation)
@everywhere include(joinpath(pwd(), "src","AnalysisModules.jl"))

# Graph to be used for the analysis
G = AnalysisModules.Graphs.smallgraph(:karate)

# Generate the dataset (using distributed computing)
# Define the samplers and their arguments (descriptor, sampler, kwargs)
tasks = [   ("networkx_configuration_model",            AnalysisModules.networkx_configuration_model,           nothing),
            ("networkx_chung_lu_model",                 AnalysisModules.networkx_chung_lu_model,                nothing),
            ("networkx_random_degree_sequence_graph",   AnalysisModules.networkx_random_degree_sequence_graph,  nothing),
            ("igraph_configuration_model",              AnalysisModules.igraph_configuration_model,             Dict(:method => :configuration)),
            ("igraph_configuration_model_fast_heur_simple",AnalysisModules.igraph_configuration_model,          Dict(:method => :fast_heur_simple)),
            ("igraph_configuration_model_edge_switching_simple",AnalysisModules.igraph_configuration_model,     Dict(:method => :edge_switching_simple)),
            ("igraph_configuration_model_vl",           AnalysisModules.igraph_configuration_model,             Dict(:method => :vl)),
            ("igraph_chung_lu_approximation",           AnalysisModules.igraph_chung_lu_approximation,          nothing),
            ("networkit_chung_lu_model",                AnalysisModules.networkit_chung_lu_model,               nothing), # Note: possible issues on Mac (segfault due to networkit)
            ("networkit_configuration_model_graph",     AnalysisModules.networkit_configuration_model_graph,    nothing),
            ("networkit_curveball",                     AnalysisModules.networkit_curveball,                    nothing),
            ("networkit_edge_switching_markov_chain",   AnalysisModules.networkit_edge_switching_markov_chain,  nothing),
            ("graphtool_configuration_model",           AnalysisModules.graph_tool_configuration_model,         nothing),
            ("graphtool_stub_matching",                 AnalysisModules.graph_tool_stub_matching,               nothing),
            ("graphtool_chung_lu",                      AnalysisModules.graph_tool_chunglu,                     nothing), # Possible issue with the circular import in Python, cf. readme.md
            ("NEMtropy_UBCM",                           AnalysisModules.NEMtropy_UBCM,                          nothing),
            ("MaxEntropyGraphs_UBCM",                   AnalysisModules.MaxEntropyGraphs_UBCM,                  nothing)];

# Run the random graph generation
res = pmap(x -> (x[1] => x[2](G, N_samples; (isnothing(x[3]) ? Dict() : x[3])...)), tasks);

# Transform the results into a DataFrame
df = DataFrame( :randomisation_method => vcat([repeat([sample.first], length(sample.second)) for sample in res]...),
                :graph                => vcat([sample.second for sample in res]...))

# Compute some metrics on the graphs (applied on the dataframe)
begin
    # compute the KL-divergence
    transform!(df, :graph => ByRow(g -> AnalysisModules.KL_divergence(g, G, α=1e-6))   => :KL_divergence)
    # compute degree
    transform!(df, :graph => ByRow(g -> AnalysisModules.Graphs.degree(g))   => :degree)
    # compute number of components
    transform!(df, :graph => ByRow(g -> length(AnalysisModules.Graphs.connected_components(g)) )   => :components)
    # assortativity coefficient 
    transform!(df, :graph => ByRow(g -> AnalysisModules.Graphs.assortativity(g)) => :assortativity)
    # compute the number of triangles
    transform!(df, :graph => ByRow(g -> sum(AnalysisModules.Graphs.triangles(g)) .÷ 3) => :triangles)
    # compute the ANND (for a specific node_id)
    node_id = 34
    field = :ANND_1
    transform!(df, :graph => ByRow(g -> MaxEntropyGraphs.ANND(g, node_id)) => field)
end

# Write the dataframe to disk
save("./results/UBCM_karateclub.jld2", "df", df)

# Different plots
begin
    # KL-divergence plot
    begin 
        p = AnalysisModules.highlevelplot(df, :KL_divergence, ylabel="Kullback-Leibler\ndivergence", left_margin=8mm)
        plot!(p, ytickfontsize=13, ylabelfontsize=14, bottom_margin=13mm, left_margin=12mm, right_margin=0mm, top_margin=1mm,size=(1600,300),legendfontsize=11, xtickfontsize=8)
        savefig(p, "./plots/UBCM_Karate_KL_divergence.pdf")
        p
    end

    # Degree adherence plot
    begin
        methodmapper = AnalysisModules.AnalysisHelpers.methodmapper

        p = plot(xlabel="Node ID", ylabel="Degree",size=(1400, 600), legend=:top,
                xtickfontsize=15, xlabelfontsize=15, ytickfontsize=15, ylabelfontsize=15, 
                legendfontsize=15,
                bottom_margin=12mm, left_margin=12mm, xticks=collect(1:34))
        
        # obtain only grand canonical models
        dfc = filter(:randomisation_method => x -> x ∈ Set(["networkx_configuration_model", 
                                                        "igraph_configuration_model",
                                                        "networkx_chung_lu_model",
                                                        "networkit_chung_lu_model",  
                                                        "graphtool_stub_matching",
                                                        "graphtool_chung_lu",
                                                        "NEMtropy_UBCM", 
                                                        "MaxEntropyGraphs_UBCM"]), df)
        # group by method
        dfcg =  groupby(dfc, :randomisation_method)
        markers = [:circle, :square, :diamond, :utriangle, :pentagon, :dtriangle, :star5, :hexagon]
        for (i,sdf) in enumerate(dfcg)
            # plot the degree sequence
            M = hcat(sdf.degree ...)
            # compute the mean and standard deviation
            μ = reshape(mean(M, dims=2),:)
            scatter!(p, μ, label="""$(replace(methodmapper[sdf.randomisation_method[1]], "\n" => " "))""", alpha=0.65, marker=markers[i], markersize=10)
        end
        # # plot the degree sequence
        # M = hcat(dfcg[1].degree ...)
        # # compute the mean and standard deviation
        # μ = reshape(mean(M, dims=2),:)
        # scatter!(p, μ, label="Mean $(dfcg[1].randomisation_method[1])", color=:black)
        scatter!(p, AnalysisModules.Graphs.degree(G), label="Observed", marker=:cross, color=:black,markersize=10)
        
        # update settings
        plot!(p, ytickfontsize=13, ylabelfontsize=14, bottom_margin=12mm, left_margin=7mm, right_margin=0mm, top_margin=4mm,size=(1400,375),legendfontsize=11, ylims=(0, 18), legendlocation=:top)

        # save the plot
        savefig(p, "./plots/UBCM_Karate_degree_sequence.pdf")
        p
    end

    # Number of components plot
    begin
        p = AnalysisModules.highlevelplot(df, :components, ylabel="Number of components", left_margin=8mm)
        plot!(p, ytickfontsize=13, ylabelfontsize=14, bottom_margin=12mm, left_margin=11mm, right_margin=0mm, top_margin=3mm,size=(1600,300),legendfontsize=11)
        ylims!(p, (0, 10))
        hline!(p, [1], label="", color=:black, linestyle=:dash)
        savefig(p, "./plots/UBCM_Karate_components_new.pdf")
        p
    end

    # Assortativity coefficient plot
    begin
        p = AnalysisModules.highlevelplot(df, :assortativity, X⁺=AnalysisModules.Graphs.assortativity(G),
                    ylabel="Assortativity coefficient", left_margin=8mm)
    
        plot!(p, ytickfontsize=13, ylabelfontsize=14, bottom_margin=12mm, left_margin=11mm, right_margin=0mm, top_margin=3mm,size=(1600,300),legendfontsize=11)        
        savefig(p, "./plots/UBCM_Karate_assortativity.pdf")
        p
    end

    # Triangle count and z-score plots
    begin
        p = AnalysisModules.highlevelplot(df, :triangles, title="", X⁺=sum(AnalysisModules.Graphs.triangles(G))÷3,
                                        ylabel="Number of triangles", left_margin=8mm)
        plot!(p, ytickfontsize=13, ylabelfontsize=14, bottom_margin=12mm, left_margin=11mm, right_margin=0mm, top_margin=1mm,size=(1400,300),legendfontsize=11)        
        savefig(p, "./plots/UBCM_Karate_triangles.pdf")
        p
    end
    begin
        adf = AnalysisModules.aggregated_zscore_pvalue(df, :triangles, sum(AnalysisModules.Graphs.triangles(G)) .÷ 3)
        # z-score computation plot
        xticks = collect(1:length(adf.randomisation_method))
        xticklabels = [AnalysisModules.AnalysisHelpers.methodmapper[adf.randomisation_method[i]] for i in xticks]
        p = plot(ylims=(-8, 8), ylabel="z-score triangles")
        # Z-scores for triangles, assuming normal distribution
        scatter!(p, adf.zscore_triangles, label="z-scores", color=palette(:tab10)[1], markersize=8)
        scatter!(p, adf.z_inferrred_triangles, label="z-scores from empirical p-value", color=palette(:tab10)[2],markersize=8, marker=:square, alpha=0.75)
        # add exlusion threshold at -2, 2
        hline!(p, [2; -2], label=false, color=palette(:tab10)[1], linestyle=:dash)
        p2 = twinx()
        plot!(p2, ylabel="p-value", yscale=:log10, ylims=(1e-4,1))

        # plot the p-values
        scatter!(p2, adf.p_empirical_triangles, label=false, markersize=8, color=palette(:tab10)[3], marker=:utriangle)
        scatter!([-1],[0], color=palette(:tab10)[3], marker=:utriangle, label="empirical p-value")
        # plot the infered p-values
        scatter!(p2, adf.p_infered_triangles, label=false, markersize=8, color=palette(:tab10)[4], marker=:dtriangle, alpha=0.75)
        scatter!([-1],[0], color=palette(:tab10)[4], marker=:dtriangle, label="infered p-value", alpha=0.75)

        # final cleanup
        plot!(p,  xlims=(0.75,length(xticks)+0.5), xticks=(xticks, xticklabels), legend=:bottom)
        plot!(p, ytickfontsize=13, ylabelfontsize=14, bottom_margin=12mm, left_margin=7mm, right_margin=7mm, top_margin=2mm,size=(1400,300),legendfontsize=11)        
        savefig(p, "./plots/UBCM_Karate_triangles_zscore_pvalue.pdf")
        p
    end
    
    # ANND and the associated z-score and p-values plots
    begin
        # z-score computation ANND_node_id
        adf = AnalysisModules.aggregated_zscore_pvalue(df,  field, MaxEntropyGraphs.ANND(G, node_id))
        # z-score computation plot
        xticks = collect(1:length(adf.randomisation_method))
        xticklabels = [AnalysisModules.AnalysisHelpers.methodmapper[adf.randomisation_method[i]] for i in xticks]
        p = plot(ylims=(-8, 8), ylabel="z-score ANND node $(node_id)")
        
        # Z-scores for triangles, assuming normal distribution
        scatter!(p,getproperty(adf, Symbol("zscore_$(field)")), label="z-scores", color=palette(:tab10)[1], markersize=8)
        scatter!(p,getproperty(adf, Symbol("z_inferrred_$(field)")), label="z-scores from empirical p-value", color=palette(:tab10)[2],markersize=8, marker=:square, alpha=0.75)
        # add exlusion threshold at -2, 2
        hline!(p, [2; -2], label=false, color=palette(:tab10)[1], linestyle=:dash)
        ylims!(p, (-8,2.1))
        ## Empirical p-values for triangles on second y-axis
        # setup second vertical axis
        p2 = twinx()
        plot!(p2, ylabel="p-value", ylabelfontsize=15,  ylims=(1e-1,1.2))

        # plot the p-values
        scatter!(p2, getproperty(adf, Symbol("p_empirical_$(field)")), label=false, markersize=8, color=palette(:tab10)[3], marker=:utriangle)
        scatter!([-1],[0], color=palette(:tab10)[3], marker=:utriangle, label="empirical p-value")
        # plot the infered p-values
        scatter!(p2, getproperty(adf, Symbol("p_infered_$(field)")), label=false, markersize=8, color=palette(:tab10)[4], marker=:dtriangle, alpha=0.75)
        scatter!([-1],[0], color=palette(:tab10)[4], marker=:dtriangle, label="infered p-value", alpha=0.75)

        # final cleanup
        plot!(p,  xlims=(0.75,length(xticks)+0.5), xticks=(xticks, xticklabels), legend=:bottomright, legendfontsize=12)
        plot!(p, ytickfontsize=13, ylabelfontsize=14, bottom_margin=13mm, left_margin=8mm, right_margin=8mm, top_margin=2mm,size=(1600,300),legendfontsize=11)        
        
        savefig(p, "./plots/UBCM_Karate_ANND_$(node_id)_zscore_pvalue.pdf")

        p
    end
end