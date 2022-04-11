########################### Setup for the experiments ###########################
# Loading packages
using ClusterManagers
using Distributed
using Printf
using Random
using DataFrames
using Distributions
using LinearAlgebra
using CoexistenceControl
using LightGraphs
using SimpleWeightedGraphs
using SparseArrays
using StatsBase
using Dates
using IterTools
using SharedArrays
using ProgressMeter
using CSV

# Set up directory -- SLURM uses different directory access
if "SLURM_JOBID" in keys(ENV)
	directory_string = "./scripts/helper"
else 
	directory_string =  "./helper" 
end

# Load configs
include(directory_string * "/experiment_config.jl")

# Loading packages for every compute node to parallelize
if parallelize
	using Pkg
	Pkg.activate(".")
	Pkg.instantiate()

	worker_ids = if "SLURM_JOBID" in keys(ENV)
		wids = ClusterManagers.addprocs_slurm(parse(Int, ENV["SLURM_NTASKS"]))
		wids
	else
		addprocs(procs_num)
	end  

	Distributed.@everywhere begin
		using Pkg
		Pkg.activate(".")
		Pkg.instantiate()
	end

	Distributed.@everywhere begin
		using DataFrames
		using Distributions
		using LinearAlgebra
		using CoexistenceControl
		using Random
		using LightGraphs
		using SimpleWeightedGraphs
		using SparseArrays
		using StatsBase
		using Printf
		using Dates
		using IterTools
		using SharedArrays
		using ProgressMeter
		using CSV
	end
	@info "Started $(Distributed.nworkers()) workers..."
	@info "Precompiling simulation code..."

end


########################### Parameter Search Setup ###########################
# Set up parameters for Cross-Entropy Method
start_mean = [-0.7, 0.2, 1.7, 0.5, 0.5, 5.0]
start_cov = diagm([0.5^2, 0.1^2, 1.0^2, 0.1^2, 0.05^2, 5.0^2])
d = MvNormal(start_mean, start_cov)
A_dist_type = "Normal"
r_dist_type = "Normal"
param_max = [2.0, 10.0, 10.0, 10.0, 0.6, 100.0]
param_min = [-2.0, 0.01, 0.01, 0.01, 0.4, 1.0]

datestring = Dates.format(now(), "e_d_u_Y_HH_MM")
results_csv = DataFrame(
	iter=Int64[], ratio=Float64[],
	A_mean=Float64[], A_sig=Float64[], r_mean=Float64[], r_sig=Float64[], 
	portion=Float64[], scaling=Float64[])
state_vec_set = generate_state_vec_set(n_species)

println(stderr, "System N: ", n_species)
println(stderr, "System T: ", n_temps)
println(stderr, "Start mean: ", start_mean)
println(stderr, "Start cov: ", start_cov)


########################### Search via CEM ###########################
try
	for i in 1:cem_iters
		params = Vector{Float64}[]
		println(stderr,"Creating $k_params simulation sets...")

		# Push in generated hyper parameters
		for k in 1:k_params
			p = rand(d)
			p = min.(p, param_max)
			p = max.(p, param_min)
			append!(p, k) # random seed of MersenneTwister
			push!(params, p)
		end
		@assert length(params) == k_params

		# Run the experiments
		if parallelize
			results = @showprogress pmap(param -> ratio_desired_configurations(
				n_species, n_temps, state_vec_set, A_dist_type, r_dist_type, param, 
				n_sims, ratio_bounds, max_samples), params)
		else
			results = @showprogress map(param -> ratio_desired_configurations(
				n_species, n_temps, state_vec_set, A_dist_type, r_dist_type, param, 
				n_sims, ratio_bounds, max_samples), params)
		end
		
		order = sortperm(results)
		elite = [e[1:length(start_mean)] for e in params[order[k_params - m_elites:end]]]
		elite_matrix = Matrix{Float64}(undef, length(start_mean), m_elites)
		for k in 1:m_elites
			elite_matrix[:,k] = elite[k]
		end
		try
			global d = fit(typeof(d), elite_matrix)
		catch ex
			if ex isa PosDefException
				println(stderr,"pos def exception")
				global d = fit(typeof(d), elite_matrix += 0.01*randn(size(elite_matrix)))
			else
				rethrow(ex)
			end
		end
		println(stderr, "Iteration $i")
		println(stderr, "Portion of within bound samples (mean): ", mean(results))
		println(stderr, "Mean: ", mean(d))
		println(stderr, "Cov (det): ", det(cov(d)))
		ev = eigvals(cov(d))
		println(stderr, "Cov (eig): ", ev)
		for j in 1:length(ev)
			println(stderr, "Eigvecs: ", eigvecs(cov(d))[:,j])
		end
		row_data = [i, mean(results)]
		append!(row_data, mean(d))
		push!(results_csv, row_data)
		if save_data
			fname_csv = joinpath(@__DIR__, 
				"../data/dataset/synthetic", 
				"cem_n_"*string(n_species)*"_t_"*string(n_temps)*"_"*datestring*".csv")
			CSV.write(fname_csv, results_csv)
			println(stderr,"Saving results to ", fname_csv)
		end

		# Early exit if generated A, r make desired portions within the bounds 95% of the time
		if mean(results) > 0.95
			break
		end
	end
finally
	Distributed.rmprocs(worker_ids)
end
