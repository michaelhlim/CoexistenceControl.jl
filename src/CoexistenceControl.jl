#= Coexistence Control module. 

    Our module supports the following functionalities:
    - Generating assemblages and transitions for a given N, T
    - Running A* path finding algorithm on the graph of assemblage transitions,
    which takes advantage of natural transitions.
    - Find suitable generative model parameters for A, r (Normal or LogNormal),
    which we can use to test our A* path finding algorithm net gain.
        
=# 
module CoexistenceControl

using DataFrames
using Distributions
using LinearAlgebra
using Distances
using Random
using LightGraphs
using SimpleWeightedGraphs
using SparseArrays
using ProgressMeter
using StatsBase

# Export statement for lv_dynamics.jl
export
    LVParams,
    LVParamsFull,
    LVState,
    AStarStruct,
    determine_feasibility,
    determine_stability,
    generate_state_vec_set,
    generate_assemblages,
    generate_params,
    generate_assemblage_transitions,
    portion_stable_feasible,
    ratio_desired_configurations

include("lv_dynamics.jl")

# Export statement for utils.jl
export
    split_idx,
    state_str_to_idx,
    state_idx_to_vec,
    state_idx_to_str,
    state_LV_to_idx,
    state_idx_to_LV,
    operation_distance,
    state_distance,
    net_species_gain

include("utils.jl")

# Export statement for generate_network.jl
export
    make_network,
    make_graph

include("generate_network.jl")

end # module
