########################### Setup for the experiments ###########################
# Loading packages
using ClusterManagers
using Distributed

# Set up directory -- SLURM uses different directory access
if "SLURM_JOBID" in keys(ENV)
	directory_string = "./helper"
	dataset_string = "./data/dataset"
else 
	directory_string =  "./helper" 
	dataset_string = "./data/dataset"
end

# Load configs & helpers
include(directory_string * "/AStar_helper.jl")
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
		# Set up directory -- Need to do this again...
		if "SLURM_JOBID" in keys(ENV)
			Distributed.@everywhere directory_string = "./helper"
			Distributed.@everywhere dataset_string = "./data/dataset"
		else 
			Distributed.@everywhere directory_string =  "./helper" 
			Distributed.@everywhere dataset_string = "./data/dataset"
		end

		# Load helper functions
		include(directory_string * "/AStar_helper.jl")
	end
	@info "Started $(Distributed.nworkers()) workers..."
	@info "Precompiling simulation code..."

end


########################### Simulation ###########################
# Run A* for all dataset
try
	include(directory_string * "/load_data.jl") 
    # Go through each dataset
    for analysis_idx = 1:nrow(experimental_data_set)
        # Set parameters for the dataset
        analysis = experimental_data_set[analysis_idx, :]
        n_species = analysis.n
        n_temps = analysis.t
        data_name = analysis.names
        temperature_list = analysis.temp
        total_replicates = analysis.rep
        total_candidates = analysis.candidates
        data_string = analysis.raw_names
        experimental_data = (data_name != "generated")

        # Load up system parameters
        (A_matrices, r_vectors, A_diag, A_sampler, r_sampler, 
            portion, scaling, species_names) = load_data(
                n_species, n_temps, data_name, 
                temperature_list, total_replicates, 
                total_candidates, dataset_string)
        prog_bar = Progress(total_replicates; 
            showspeed = true,
            enabled = !experimental_data
            )

        if !experimental_data
            println("==================================")
            println(("Running " * string(total_replicates) * 
                " synthetic experiments for:"))
            println("- N: ", n_species)
            println("- T: ", n_temps)
            println("- A parameters: ", A_sampler)
            println("- r parameters: ", r_sampler)
        end

        for reps = 1:total_replicates
            # Setup the environment
            rng = MersenneTwister(reps)
            params_astar = get_data(
                n_species, n_temps, data_string, A_sampler, r_sampler, portion, scaling, 
                add_cost, del_cost, wait_cost, temp_cost, rng, experimental_data; 
                A_diag = A_diag, A_matrices = A_matrices, r_vectors = r_vectors,
                show_print_statements = experimental_data
                )
            (assemblage, transitions_natural, transitions_mat) = get_assemblage(
                n_species, n_temps, params_astar; 
                show_print_statements = experimental_data)
            (proportion, candidate_states, candidate_states_pairs) = get_candidates(
                assemblage, total_candidates;
                show_print_statements = experimental_data
                )

            # Run the experiments
            results_table = run_AStar(
                n_species, n_temps, params_astar, transitions_mat, 
                rng, parallelize, candidate_states_pairs;
                show_print_statements = experimental_data
                )
            results_stats = get_statistics(
                results_table;
                show_print_statements = experimental_data
                )
            
            # Save Data
            if save_data
                # Get the relative path
                if experimental_data
                    relative_path = joinpath(@__DIR__, "../data/results/experimental")
                else
                    relative_path = joinpath(
                        @__DIR__, ("../data/results/synthetic/n" * 
                                    string(n_species) * "_t" * string(n_temps))
                        )
                end

                # Make directory if it does not exist
                isdir(relative_path) || mkdir(relative_path)

                # Save all the data
                save_all_data(
                    n_species, n_temps, portion, scaling, A_sampler, r_sampler, 
                    params_astar, data_string, species_names, temperature_list, 
                    assemblage, transitions_natural, results_table, results_stats, 
                    candidate_states, candidate_states_pairs, experimental_data, 
                    relative_path; 
                    replicate = reps, show_print_statements = experimental_data
                    )
            end
            
            next!(prog_bar)
        end
	end
finally
	if parallelize
		Distributed.rmprocs(worker_ids)
	end
end