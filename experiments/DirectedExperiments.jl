begin
    using Pkg
    cd(joinpath(@__DIR__,".."))
    @info "working in $(pwd())"
    Pkg.activate(pwd())

    using DataFrames
    using CSV
    using JLD2
    using MaxEntropyGraphs
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

# Graph to be used for the analysis (assuring no self-loops)
G = MaxEntropyGraphs.chesapeakebay()
for e in edges(G)
    if e.src == e.dst
        rem_edge!(G, e)
    end
end
@assert !Graphs.has_self_loops(G)

# Generate the dataset (using distributed computing)
# Define the samplers and their arguments (descriptor, sampler, kwargs)
tasks = [   ("networkx_directed_configuration_model",               AnalysisModules.networkx_directed_configuration_model,      nothing),
            ("igraph_directed_configuration_model",                 AnalysisModules.igraph_directed_configuration_model,        Dict(:method => :configuration)),
            ("igraph_directed_configuration_model_fast_heur_simple",AnalysisModules.igraph_directed_configuration_model,        Dict(:method => :fast_heur_simple)), # can be very slow
            ("igraph_directed_configuration_model_configuration_simple",AnalysisModules.igraph_directed_configuration_model,    Dict(:method => :configuration_simple)), # can be very slow
            ("igraph_directed_configuration_model_edge_switching_simple",AnalysisModules.igraph_directed_configuration_model,   Dict(:method => :edge_switching_simple)),
            ("igraph_degree_sequence_game_directed",                AnalysisModules.igraph_degree_sequence_game_directed,       nothing),
            ("igraph_static_fitness_game_directed",                 AnalysisModules.igraph_static_fitness_game_directed,        nothing),
            ("networkit_directed_curveball",                        AnalysisModules.networkit_directed_curveball,               nothing), # Note: possible issues on Mac (segfault due to networkit)
            ("networkit_directed_edge_switching_markov_chain",      AnalysisModules.networkit_directed_edge_switching_markov_chain, nothing),
            ("graph_tool_directed_configuration_model",             AnalysisModules.graph_tool_directed_configuration_model,    nothing), # can have some issues with libpango-1.0.0.dylib 
            ("graph_tool_stub_matching_directed",                   AnalysisModules.graph_tool_stub_matching_directed,          nothing),
            ("graph_tool_chunglu_directed",                         AnalysisModules.graph_tool_chunglu_directed,                nothing),
            ("NEMtropy_DBCM",                                       AnalysisModules.NEMtropy_DBCM,                              Dict(:netname => "chesapeake", :conda_env_name => CONDA_ENV_NAME)),
            ("MaxEntropyGraphs_DBCM",                               AnalysisModules.MaxEntropyGraphs_DBCM,                      nothing)];


# Run the random graph generation
res = pmap(x -> (x[1] => x[2](G, N_samples; (isnothing(x[3]) ? Dict() : x[3])...)), tasks);

# Transform the results into a DataFrame
df = DataFrame( :randomisation_method => vcat([repeat([sample.first], length(sample.second)) for sample in res]...),
                :graph                => vcat([sample.second for sample in res]...))

# Compute some metrics on the graphs (applied on the dataframe)
begin
    # Compute the motifs
    transform!(df,  # motif counts
        :graph => ByRow(g -> M1(g))   => :M1,
        :graph => ByRow(g -> M2(g))   => :M2,
        :graph => ByRow(g -> M3(g))   => :M3,
        :graph => ByRow(g -> M4(g))   => :M4,
        :graph => ByRow(g -> M5(g))   => :M5,
        :graph => ByRow(g -> M6(g))   => :M6,
        :graph => ByRow(g -> M7(g))   => :M7,
        :graph => ByRow(g -> M8(g))   => :M8,
        :graph => ByRow(g -> M9(g))   => :M9,
        :graph => ByRow(g -> M10(g))   => :M10,
        :graph => ByRow(g -> M11(g))   => :M11,
        :graph => ByRow(g -> M12(g))   => :M12,
        :graph => ByRow(g -> M13(g))   => :M13)
end


# Write the dataframe to disk
save("./results/ChesapeakeBay.jld2", "df", df)

# Different plots
begin
    # Motif z-score plot
    begin
        # setup storage
        res = Dict(method => Dict(  "z-score" => Vector{Float64}(undef, 13),
                                    "inferred z-score" => Vector{Float64}(undef, 13),
                                    "p-value" => Vector{Float64}(undef, 13),
                                    "inferred p-value" => Vector{Float64}(undef, 13))  for method in unique(df.randomisation_method))
        
        # Computation part
        for i in 1:13
            ex = :($(MaxEntropyGraphs.directed_graph_motif_function_names[i])(G))
            adf = AnalysisModules.aggregated_zscore_pvalue(df, MaxEntropyGraphs.directed_graph_motif_function_names[i], eval(ex))
            for row in eachrow(adf)
                res[row.randomisation_method]["z-score"][i] = getproperty(row, Symbol("zscore_$(MaxEntropyGraphs.directed_graph_motif_function_names[i])"))
                res[row.randomisation_method]["inferred z-score"][i] = getproperty(row, Symbol("z_inferrred_$(MaxEntropyGraphs.directed_graph_motif_function_names[i])"))
                res[row.randomisation_method]["p-value"][i] = getproperty(row, Symbol("p_empirical_$(MaxEntropyGraphs.directed_graph_motif_function_names[i])"))
                res[row.randomisation_method]["inferred p-value"][i] = getproperty(row, Symbol("p_infered_$(MaxEntropyGraphs.directed_graph_motif_function_names[i])"))
            end
        end
    
        # Plotting part
        modelmap = Dict(
                    "NEMtropy_DBCM"                                             => "NEMtropy (DBCM)",
                    "graph_tool_directed_configuration_model"                    => "graph-tool (directed configuration model)",
                    "networkx_directed_configuration_model"                     => "networkx (directed configuration model)",
                    "MaxEntropyGraphs_DBCM"                                     => "MaxEntropyGraphs (DBCM)",
                    "networkit_directed_curveball"                              => "Networkit (directed curveball)",
                    "networkit_directed_edge_switching_markov_chain"            => "Networkit (directed edge switching)",
                    "igraph_directed_configuration_model_edge_switching_simple" => "igraph (directed configuration model edge switching)",
                    "igraph_directed_configuration_model"                       => "igraph (directed configuration model)",
                    "igraph_degree_sequence_game_directed"                      => "igraph (degree sequence game)",
                    "igraph_static_fitness_game_directed"                       => "igraph (static fitness game)",
                    "graph_tool_stub_matching_directed"                         => "graph-tool (stub matching)",
                    "graph_tool_chunglu_directed"                               => "graph-tool (chung-lu)",
                    "igraph_directed_configuration_model_fast_heur_simple"      => "igraph (fast_heur_simple)",
                    "igraph_directed_configuration_model_configuration_simple"  => "igraph (configuration, simple)",
                    )

        # setup the plot
        xticks = collect(1:13)
        xticklabels = MaxEntropyGraphs.directed_graph_motif_function_names
        p = plot(size=(1200, 600), 
                    left_margin=8mm, right_margin=8mm, bottom_margin=10mm,
                    legend=:bottomleft, 
                    xtickfontsize=14, xlabel="Motif", xlabelfontsize=15,
                    xlims=(0,length(xticks)+1), xticks=(xticks, xticklabels),
                    ytickfontsize=14, ylabel="z-score", ylabelfontsize=15,
                    ylims=(-15, 15))
    
        for (i, method) in enumerate(unique(df.randomisation_method))
            label = modelmap[method]#replace(AnalysisModules.AnalysisHelpers.methodmapper[method], "\n" => " ")
            plot!(p,[0],[0], label=label, color=palette(:tab10)[i])
            plot!(p, res[method]["z-score"],          label="", color=palette(:tab10)[i], markersize=8, marker=:circle)
            plot!(p, res[method]["inferred z-score"], label="",     color=palette(:tab10)[i], markersize=8, marker=:square, markeralpha=0.5, line=:dash)
            # check if there are Inf values
            minus_inf = findall(x -> x == -Inf, res[method]["inferred z-score"])
            plus_inf = findall(x -> x == Inf, res[method]["inferred z-score"])
            # ADD STAR MARKER FOR -INFINITY VALUES
            if !isempty(minus_inf)
                scatter!(p, minus_inf, res[method]["z-score"][minus_inf], label="", color=palette(:tab10)[i], markersize=20, marker=:star, markeralpha=0.5)
            end
            # ADD PENTAGON MARKER FOR +INFINITY VALUES
            if !isempty(plus_inf)
                scatter!(p, plus_inf, res[method]["z-score"][plus_inf], label="", color=palette(:tab10)[i], markersize=20, marker=:hexagon, markeralpha=0.5)
            end
        end
        # add exlusion threshold at -2, 2
        hline!(p, [2; -2], label=false, color=:black, linestyle=:dash)
        
        # setup readability
        plot!(p, legendfontsize=12)
    
        # save the plot
        savefig(p, "./plots/Chesapeakebay_motifs_zscore.pdf")
    end

    # Motif density plots
    begin
        # density plot per motif
        gdf = groupby(df, :randomisation_method);
        mycolor = Dict()
        for mot in MaxEntropyGraphs.directed_graph_motif_function_names
            p = plot()
            for subgroup in gdf
                obs = countmap(subgroup[!,mot])
                x = collect(minimum(keys(obs)):maximum(keys(obs)))
                histogram!(p, subgroup[!,mot], normalize=:pdf, alpha=0.2, label="", linecolor=:match, linealpha=0.2) # , bar_width=1, bins=x.-0.5
                mycolor[subgroup.randomisation_method[1]] = p.series_list[end][:linecolor]
            end
            for subgroup in gdf
                obs = countmap(subgroup[!,mot])
                x = collect(minimum(keys(obs)):maximum(keys(obs)))
                d = fit(Normal, subgroup[!,mot])
                plot!(p, x, pdf(d, x), label="", color = mycolor[subgroup.randomisation_method[1]], linewidth=2)
            end
            vline!(p, [eval(mot)(G)], label="", color=:black, style=:dash, linewidth=2)
            xlims!(p, minimum(df[!,mot])-1, maximum(df[!,mot])+1)
            plot!(p, size=(800,600), legend=false, xlabel="$(mot)", ylabel="Density", bottom_margin=5mm, xlabelfontsize=20, xtickfontsize=16, ylabelfontsize=20, ytickfontsize=16)
            savefig(p, "./plots/Chesapeakebay_$(mot).pdf")
        end

        # legend per motif
        p = plot()
        for subgroup in gdf
            plot!(p, [1], [1], label=modelmap[subgroup.randomisation_method[1]], color=mycolor[subgroup.randomisation_method[1]], linewidth=2)
            plot!(p, size=(800,600), legend=:top, xlabel="", ylabel="", grid=false, bottom_margin=5mm, xlabelfontsize=20, xtickfontsize=16, ylabelfontsize=20, ytickfontsize=16, axis=false, showgrid=false)
            savefig(p, "./plots/Chesapeakebay_motifs_legend.pdf")
        end
    end
end

