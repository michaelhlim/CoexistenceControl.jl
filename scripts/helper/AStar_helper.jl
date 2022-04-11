# Importing all the packages needed for the helper functions
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

function optimistic_distance(n_species::Int64, 
    idx1::Int64, 
    idx2::Int64, 
    add_cost::Float64, 
    del_cost::Float64)
    #= Get optimistic distance heuristic between state 1 and 2. 
    (single temperature)

    Optimistic distance is the heuristic cost required for A*
    in order to transition from state 1 to state 2, assuming 
    one natural transition can eliminate all deletion species.
        
    Args:
        n_species: Number of species in the system
        idx1: Raw index of state 1
        idx2: Raw index of state 2
        add_cost: Cost of addition
		del_cost: Cost of deletion
    
    Return:
        Float object of optimistic distance
    =#
    # Return 0 if same
    if idx1 == idx2
        return 0
    end

    # Make them into arrays of species
    state1 = state_idx_to_vec(n_species, idx1)
    state2 = state_idx_to_vec(n_species, idx2)
    state_diff = state1 - state2

    add_ops = count(i->(i==-1), state_diff)
    del_ops = (any(state_diff .== 1) ? 1 : 0)

    return (add_cost * add_ops + del_cost * del_ops)
end

function optimistic_distance(n_species::Int64, 
    n_temps::Int64, 
    idx1::Int64, 
    idx2::Int64, 
    add_cost::Float64, 
    del_cost::Float64, 
    temp_cost::Float64)
    #= Get optimistic distance heuristic between state 1 and 2. 
    (multi temperature)

    Optimistic distance is the heuristic cost required for A*
    in order to transition from state 1 to state 2, assuming 
    one natural transition can eliminate all deletion species.
        
    Args:
        n_species: Number of species in the system
        n_temps: Number of temperature settings in the system
        idx1: Raw index of state 1
        idx2: Raw index of state 2
        add_cost: Cost of addition
		del_cost: Cost of deletion
		temp_cost: Cost of temperature transition
    
    Return:
        Float object of optimistic distance
    =#
    # Get the indices
    (idx_species1, idx_temp1) = split_idx(n_species, n_temps, idx1)
    (idx_species2, idx_temp2) = split_idx(n_species, n_temps, idx2)

    # Get temp-less state distance
    dist = optimistic_distance(
        n_species, 
        idx_species1, 
        idx_species2, 
        add_cost::Float64, 
        del_cost::Float64)

    # Check how much temperature distance is there
    temp_op = abs(idx_temp1 - idx_temp2)
    
    return dist + temp_op * temp_cost
end

function load_data(n_species::Int64, 
	n_temps::Int64, 
	data_name::String,
	temperature_list::Vector{String},
	total_replicates::Int64,
	total_candidates::Int64,
	data_directory::String)
	#= Load up experimental or synthetic data.
		
	Args:
		n_species: Number of species in the system
		n_temps: Number of temperature settings in the system
		data_name: Name of dataset
		temperature_list: Temperatures for the dataset (empty string 
				means no specific temperature)
		total_replicates: Number of replicates for the study
		total_candidates: Number of candidate pairs for the study
		data_directory: String for the relative/absolute directory
	
	Return:
		A_matrices: If specified, A matrices for all temperatures
				will be set directly to A_matrices
		r_vectors: If specified, r vectors for all temperatures 
				will be set directly to r_vectors
		A_diag: If specified, diagonals of A matrix will be set to 
				the A_diag vector
		A_sampler: Distribution object which we sample the A matrix from
		r_sampler: Distribution object which we sample the r vector from
		portion: For varying temperatures, we sample 'portion' amount 
				of indices in which we scale up/down for
		scaling: The scaling factor for varying temperatures
		species_names: Names of the species
	=#
	# Create A, r lists
	A_matrices = Matrix{Float64}[]
	r_vectors = Vector{Float64}[]

	if data_name == "generated"
		# Load synthetic data
		synthetic_data = CSV.read(
			data_directory * "/synthetic/CEM_Final_Values.csv", 
			DataFrame)

		# Get the index corresponding to the parameters
		idx = findall(((synthetic_data[!, :N] .== n_species) .& 
				(synthetic_data[!, :T] .== n_temps)))

		# Obtain the parameters 
		# These are one element vectors, so take "first" element
		mu_r = first(synthetic_data[idx, :mu_r])
		sigma_r = first(synthetic_data[idx, :sigma_r])
		mu_A = first(synthetic_data[idx, :mu_A])
		sigma_A = first(synthetic_data[idx, :sigma_A])
		portion = first(synthetic_data[idx, :portion])
		scaling = first(synthetic_data[idx, :scaling])

		# Setting relevant variables
		A_diag = repeat([-1.0], n_species)
		A_sampler = Normal(mu_A, sigma_A)
		r_sampler = Normal(mu_r, sigma_r)
		A_matrices = nothing
		r_vectors = nothing
		species_names = [""]
	else
		# Load experimental data
		# Iterate over all temperatures
		for temp = temperature_list
			# Prepare dataset name
			data_string = data_directory * "/" * data_name
			if temp != ""
				data_string *= "/" * temp
			end

			# Process and push into the list
			A_df = CSV.read(data_string * "/a_matrix.csv", DataFrame)
			r_df = CSV.read(data_string * "/r_vector.csv", DataFrame)
			A = Matrix(A_df)
			r = Vector(r_df[1, :])
			species_names = names(A_df)

			push!(A_matrices, A)
			push!(r_vectors, r)
		end

		# Loading dummy variables
		A_diag = repeat([-1.0], n_species)
		A_sampler = Normal(0, 1)
		r_sampler = Normal(0, 1)
		portion = 1.0
		scaling = 1.0
	end

	return (A_matrices, r_vectors, A_diag, A_sampler, r_sampler, 
		portion, scaling, species_names)
end

function get_data(n_species::Int64, 
	n_temps::Int64, 
	data_string::String,
	A_sampler::Distribution, 
	r_sampler::Distribution, 
	portion::Float64, 
	scaling::Float64, 
	add_cost::Float64, 
	del_cost::Float64, 
	wait_cost::Float64, 
	temp_cost::Float64, 
	rng::AbstractRNG,
	experimental_data::Bool;
	A_diag = nothing, 
	A_matrices = nothing, 
	r_vectors = nothing,
	show_print_statements = true)
	#= Load up LV parameter settings.
		
	Args:
		n_species: Number of species in the system
		n_temps: Number of temperature settings in the system
		data_string: String that indicates the data source
		A_sampler: Distribution object which we sample the A matrix from
		r_sampler: Distribution object which we sample the r vector from
		portion: For varying temperatures, we sample 'portion' amount 
				of indices in which we scale up/down for
		scaling: The scaling factor for varying temperatures
		add_cost: Cost of addition action
		del_cost: Cost of deletion action
		wait_cost: Cost of wait action
		temp_cost: Cost of temperature change action
		rng: RNG object that can define the seed
		experimental_data: Bool for checking if we are loading
				experimental or synthetic data

	Keyword Args:
		A_diag: If specified, diagonals of A matrix will be set to 
				the A_diag vector
		A_matrices: If specified, A matrices for all temperatures
				will be set directly to A_matrices
		r_vectors: If specified, r vectors for all temperatures 
				will be set directly to r_vectors
		show_print_statements: If true, shows all print statements
	
	Return:
		params_astar: An LVParamsFull object that contains A, r for all temperatures
	=#
	if experimental_data 
		params_astar = generate_params(
			n_species, n_temps, A_sampler, r_sampler, 
			portion, scaling, 
			add_cost, del_cost, wait_cost, temp_cost, rng; 
			A_diag = A_diag, A_matrices = A_matrices, r_vectors = r_vectors)
		
		if show_print_statements
			println("==================================")
			println("Setting up LV system with N = " 
				* string(n_species) * ", T = " * string(n_temps) * "...")
			println("Loading A, r from data: "*data_string)
		end
	else
		if show_print_statements
			println("==================================")
			println("Setting up LV system with N = " 
				* string(n_species) * ", T = " * string(n_temps) * "...")
			println("Generating A, r from parameters...")
		end
		params_astar = generate_params(
			n_species, n_temps, A_sampler, r_sampler, 
			portion, scaling, 
			add_cost, del_cost, wait_cost, temp_cost, rng; 
			A_diag = A_diag)
	end

	if show_print_statements
		println("==================================")
		println("A matrices:")
		for i=1:n_temps
			println("(T = ", i, ")")
			show(stdout, "text/plain", params_astar.A_matrices[i])
			println()
		end
		
		println("\n==================================")
		println("r vectors:")
		for i=1:n_temps
			println("(T = ", i, ")")
			show(stdout, "text/plain", params_astar.r_vectors[i])
			println()
		end
	end

	return params_astar
end

function get_assemblage(n_species::Int64, 
	n_temps::Int64, 
	params_astar::LVParamsFull;
	show_print_statements = true)
	#= Generate assemblage and transition
		
	Args:
		n_species: Number of species in the system
		n_temps: Number of temperature settings in the system
		params_astar: An LVParamsFull object that contains 
				A, r for all temperatures
	
	Keyword Args:
		show_print_statements: If true, shows all print statements

	Return:
		assemblage: Assemblage object including all transitions
		transitions_natural: DataFrame of only natural transitions
		transitions_mat: Matrix object used to make AStar graph,
				which includes all transitions and costs
				1st column is from_index
				2nd column is to_index
				3rd column is cost/weight
	=#
	# Generate transition networks
	if show_print_statements
		println("\n==================================")
		println("Generating assemblages and networks...")
	end
	assemblage = generate_assemblages(n_species, n_temps)
	assemblage = generate_assemblage_transitions(
		n_species, n_temps, assemblage, params_astar;
		show_print_statements = show_print_statements
		)
	transitions = make_network(
		n_species, n_temps, assemblage, params_astar;
		show_print_statements = show_print_statements
		)
	transitions_mat = Matrix(
		transitions[:, [:from_idx, :to_idx, :weight]])
	transitions_natural = make_network(
		n_species, n_temps, assemblage, params_astar;
		include_single = false, 
		show_print_statements = show_print_statements
		)

	return (assemblage, transitions_natural, transitions_mat)
end

function get_candidates(assemblage::DataFrame,
	total_candidates::Int64;
	show_print_statements = true)
	#= Generate assemblage and transition
		
	Args:
		assemblage: Assemblage object including all transitions
		total_candidates: Max number of candidate pairs generated
	
	Keyword Args:
		show_print_statements: If true, shows all print statements

	Return:
		proportion: Proportion of states that are candidates 
				(feasible and stable) among all states
		candidate_states: Vector of all candidate states
		candidate_states_pairs: Vector of all candidate states pairs 
				(i, j) (from i to j)
	=#
	# Test how much improvement there is by testing all the 
	# path finding between the stable population arrangements
	if show_print_statements
		println("==================================")
		println("Finding suitable candidates that are stable "
			* "and feasible and iterating over all path combinations...")
	end

	candidate_states = (findall((assemblage[!, :stable] .== true) 
		.& (assemblage[!, :feasible] .== true)))
	proportion = length(candidate_states)*1.0/nrow(assemblage)
	
	if show_print_statements
		println("Proportion of candidate states: ", 
			proportion)
		println("Candidate states: ", length(candidate_states))
		println("Total states: ", nrow(assemblage))
	end

	# Pick a subset of candidate states pairs when the space is large
	candidate_states_pairs = Vector{Int64}[]
	if length(candidate_states)^2 < total_candidates
		for i = 1:length(candidate_states)
			for j = 1:length(candidate_states)
				if i != j
					push!(
						candidate_states_pairs, 
						[candidate_states[i], candidate_states[j]])
				end
			end
		end
	else
		for i = 1:total_candidates
			push!(
				candidate_states_pairs, 
				sample(candidate_states, 2, replace = false))
		end
	end

	return (proportion, candidate_states, candidate_states_pairs)
end

function run_AStar(n_species::Int64, 
	n_temps::Int64, 
	params_astar::LVParamsFull, 
	transitions_mat::Matrix{Float64},  
	rng::AbstractRNG,
	parallelize::Bool,
	candidate_states_pairs::Vector{Vector{Int64}};
	show_print_statements = true)
	#= Run A* experiment for a given parameter set.
		
	Args:
		n_species: Number of species in the system
		n_temps: Number of temperature settings in the system
		params: LVParamsFull object
		transitions_mat: Matrix of all the transitions ("network")
		rng: RNG object that can define the seed
		parallelize: Boolean for parallelizing
		candidate_states_pairs: Vector of all candidate states pairs 
				(i, j)
	
	Keyword Args:
		show_print_statements: If true, shows all print statements

	Return:
		results_table: A DataFrame that contains the
			results of the A* search for going from i to j.
	=#
	# Run the experiments
	if parallelize
		# Note: Unless you have many cores to burn (>100+), it's not worth
		# parallelzing since making the graph takes up majority of the time.
		# We cannot avoid making the graph in each core, since once the graph
		# becomes larger, the processes will inevitably face segfault errors.
		# As far as I'm concerned, there's no good way to create the graphs
		# or sparse matrices to be shared in parallel threads in 
		# Julia currently... I'd encourage you to just run it 
		# without parallelization; it doesn't take too long.
		if show_print_statements
			results = @showprogress pmap(states -> solve_AStar(
				n_species, n_temps, states, params_astar, 
				transitions_mat, rng), candidate_states_pairs
				)
		else
			results = pmap(states -> solve_AStar(
				n_species, n_temps, states, params_astar, 
				transitions_mat, rng), candidate_states_pairs
				)
		end
	else
		# If not parallelizing, we can save the graph_astar
		# calculation for each node
		graph_astar = make_graph(n_species, n_temps, transitions_mat)

		if show_print_statements
			results = @showprogress map(states -> solve_AStar(
				n_species, n_temps, states, params_astar, 
				graph_astar, rng), candidate_states_pairs
				)
		else
			results = map(states -> solve_AStar(
				n_species, n_temps, states, params_astar, 
				graph_astar, rng), candidate_states_pairs
				)
		end
	end

	# Concatenate results
	results_table = vcat(results...)

	return results_table
end

function solve_AStar(n_species::Int64, 
	n_temps::Int64, 
	idx_pair::Vector{Int64}, 
	params::LVParamsFull, 
	graph_astar::AStarStruct,  
	rng::AbstractRNG)
	#= Function to run a single A* experiment 
	between states i and j.
		
	Args:
		n_species: Number of species in the system
		n_temps: Number of temperature settings in the system
		idx_pair: [from, to] The indices to be tested 
		params: LVParamsFull object
		graph_astar: Graph object for A* search 
				(includes natural transitions)
		rng: RNG object that can define the seed
	
	Return:
		results: A DataFrame (with one row) that contains the
			results of the A* search for going from i to j.
	=#
	# Make a data frame for this one parallel thread
	results = DataFrame(
		from = String[], 
		to = String[], 
		net_species_gain = Int64[],
		path_astar = String[], 
		path_nominal = String[],
		path_length = Int64[],  
		path_length_astar = Int64[],
		net_length_improvement = Int64[], 
		net_cost = Float64[], 
		net_cost_astar = Float64[],
		net_cost_improvement = Float64[],
		astar_time = Float64[])

	i = idx_pair[1]
	j = idx_pair[2]

	# Define the heuristic function which calculates the 
	# optimistic addition/deletion distance
	optimistic_distance_to(idx1) = optimistic_distance(
		n_species, n_temps, idx1, j, 
		params.add_cost, params.del_cost, params.temp_cost)

	# A* search
	astar_time = @elapsed path_a_star = a_star(
		graph_astar.graph, i, j, graph_astar.distmx, optimistic_distance_to)

	# Get path information for nominal and A*
	(path_astar, net_cost_astar, path_length_astar) = astar_path(
		n_species, n_temps, path_a_star, params, graph_astar)
	path_nominal = nominal_path(n_species, n_temps, i, j)

	# Collect informations
	net_cost = state_distance(
		n_species, n_temps, i, j, 
		params.add_cost, params.del_cost, params.temp_cost)
	path_length = operation_distance(n_species, n_temps, i, j)
	net_species_diff = net_species_gain(n_species, n_temps, i, j)
	net_cost_improvement = round(net_cost - net_cost_astar, digits = 1)
	net_length_improvement = path_length - path_length_astar

	push!(results, [
		state_idx_to_str(n_species, n_temps, i), 
		state_idx_to_str(n_species, n_temps, j),
		net_species_diff,
		path_astar, 
		path_nominal, 
		path_length, 
		path_length_astar,
		net_length_improvement, 
		net_cost, 
		net_cost_astar,
		net_cost_improvement,
		astar_time])

	return results
end

solve_AStar(n_species::Int64, 
	n_temps::Int64, 
	idx_pair::Vector{Int64}, 
	params::LVParamsFull, 
	transitions_mat::Matrix{Float64},  
	rng::AbstractRNG) = solve_AStar(
		n_species, n_temps, idx_pair, params,
		make_graph(n_species, n_temps, transitions_mat),
		rng)

function astar_path(n_species::Int64, 
	n_temps::Int64, 
	path_a_star, 
	params::LVParamsFull, 
	graph_astar::AStarStruct)
	#= Function to extract the A* search path information.
		
	Args:
		n_species: Number of species in the system
		n_temps: Number of temperature settings in the system
		path_a_star: Path obtained by a_star function
		params: LVParamsFull object
		graph_astar: Graph object for A* search 
				(includes natural transitions)
		
	Return:
		path_astar: String of A* search path
		net_cost_astar: Net cost of A* search path
		path_length_astar: Length of A* search path
	=#
	# Collect informations
	path_astar = "(" * state_idx_to_str(
		n_species, n_temps, src(path_a_star[1])) * ")"
	net_cost_astar = 0.0
	path_length_astar = 0
	
	# Get path informations with natural transitions
	# TODO: this needs a better logic for checking which kind of
	# transition it is... if any costs are equal, then we are in trouble
	for edge in path_a_star
		edge_cost = graph_astar.distmx[src(edge), dst(edge)]
		transition_str = ""
		if edge_cost == params.add_cost
			transition_str = "+"
		elseif edge_cost == params.del_cost
			transition_str = "-"
		elseif edge_cost == params.wait_cost
			transition_str = ">"
		else
			transition_str = "="
		end

		path_astar *= (transition_str * "(" 
			* state_idx_to_str(n_species, n_temps, dst(edge)) * ")")
		net_cost_astar += edge_cost
		path_length_astar += 1
	end
	
	return (path_astar, net_cost_astar, path_length_astar)
end

function nominal_path(n_species::Int64, 
    n_temps::Int64, 
    idx1::Int64, 
    idx2::Int64)
    #= Function to get the nominal path between states 1 and 2.

    Args:
        n_species: Number of species in the system
        n_temps: Number of temperature settings in the system
        idx1: Raw index of state 1
        idx2: Raw index of state 2
    
    Return:
        path_nominal: String object of the nominal path
    =#
	curr = state_idx_to_LV(n_species, n_temps, idx1)
	goal = state_idx_to_LV(n_species, n_temps, idx2)
	path_nominal = "(" * state_idx_to_str(n_species, n_temps, idx1) * ")"

	while curr != goal
		# Check temperature match
		if curr.t != goal.t
			curr = LVState(curr.s, curr.t + sign(goal.t - curr.t))
			path_nominal *= "=" * "(" * state_idx_to_str(
				n_species, n_temps, state_LV_to_idx(
					n_species, n_temps, curr)) * ")"
			continue
		end

		# Check state difference
		diff = goal.s - curr.s
		add_any = any(x->x>0, diff)
		del_any = any(x->x<0, diff)

		# Check deletion match
		if del_any
			first_idx = findall(x->x<0, diff)[1]
			new_s = copy(curr.s)
			new_s[first_idx] = 0
			curr = LVState(new_s, curr.t)
			path_nominal *= "-" * "(" * state_idx_to_str(
				n_species, n_temps, state_LV_to_idx(
					n_species, n_temps, curr)) * ")"
			continue
		end

		# Check addition match
		if add_any
			first_idx = findall(x->x>0, diff)[1]
			new_s = copy(curr.s)
			new_s[first_idx] = 1
			curr = LVState(new_s, curr.t)
			path_nominal *= "+" * "(" * state_idx_to_str(
				n_species, n_temps, state_LV_to_idx(
					n_species, n_temps, curr)) * ")"
		end
	end
    
    return path_nominal
end

function get_statistics(results_table::DataFrame;
	show_print_statements = true)
	#= Calculate statistics for the A* experiment
		
	Args:
		results_table: A DataFrame that contains the
			results of the A* search for going from i to j.
	
	Keyword Args:
		show_print_statements: If true, shows all print statements
	
	Return:
		results_stats: A Dictionary that contains the
			relevant statistics of the results.
	=#
	# Calculate statistics
	length_imp = countmap(results_table[!, :net_length_improvement])
	cost_imp = countmap(results_table[!, :net_cost_improvement])
	length_imp_val = [k for (k,v) in length_imp]
	length_imp_count = [v for (k,v) in length_imp]
	cost_imp_val = [k for (k,v) in cost_imp]
	cost_imp_count = [v for (k,v) in cost_imp]

	avg_length_imp = dot(length_imp_count, length_imp_val)/sum(length_imp_count)
	avg_cost_imp = dot(cost_imp_count, cost_imp_val)/sum(cost_imp_count)

	if show_print_statements
		println("==================================")
		println("Printing some interesting statistics...")
		println("* Mean steps saved by using A* compared to nominal path distance:")
		println("(positive means A* took less steps)")
		println(avg_length_imp)
		println("\n* Mean cost difference between nominal and A*: ")
		println("(positive means A* did better)")
		println(avg_cost_imp)
		println("\n* A* Planning time: ")
		println(
			"Mean = ", 
			round(mean(results_table[!, :astar_time]), digits = 6), " (s) | SD = ", 
			round(std(results_table[!, :astar_time]), digits = 6), " (s)")
	end
	
	results_stats = Dict(
		"length_imp" => length_imp,
		"cost_imp" => cost_imp,
		"length_imp_val" => length_imp_val,
		"length_imp_count" => length_imp_count,
		"cost_imp_val" => cost_imp_val,
		"cost_imp_count" => cost_imp_count,
		"avg_length_imp" => avg_length_imp,
		"avg_cost_imp" => avg_cost_imp,
	)

	return results_stats
end

function save_all_data(n_species::Int64, 
	n_temps::Int64, 
	portion::Float64, 
	scaling::Float64, 
	A_sampler::Distribution, 
	r_sampler::Distribution, 
	params_astar::LVParamsFull,
	data_string::String,
	species_names::Vector{String},
	temperature_list::Vector{String},
	assemblage::DataFrame,
	transitions_natural::DataFrame,
	results_table::DataFrame,
	results_stats::Dict,
	candidate_states::Vector{Int64},
	candidate_states_pairs::Vector{Vector{Int64}},
	experimental_data::Bool,
	relative_path::String;
	replicate = nothing,
	show_print_statements = true)
	#= Save the relevant data for the experiment
		
	Args:
		n_species: Number of species in the system
        n_temps: Number of temperature settings in the system
		portion: For varying temperatures, we sample 'portion' amount 
				of indices in which we scale up/down for
		scaling: The scaling factor for varying temperatures
		A_sampler: Distribution object which we sample the A matrix from
		r_sampler: Distribution object which we sample the r vector from
		params_astar: An LVParamsFull object that contains 
				A, r for all temperatures
		data_string: String that indicates the data source
		species_names: Vector of strings that show names of species
		temperature_list: Vector of strings of temperature values used
		assemblage: Assemblage object including all transitions
		transitions_natural: DataFrame of only natural transitions
		results_table: A DataFrame that contains the
			results of the A* search for going from i to j.
		results_stats: A Dictionary that contains the
			relevant statistics of the results.
		candidate_states: Vector of all candidate states
		candidate_states_pairs: Vector of all candidate states pairs 
				(i, j) (from i to j)
		experimental_data: Bool for checking if we are loading
				experimental or synthetic data
		relative_path: Relative path to store the data into
	
	Keyword Args:
		replicate: Replicate number that is used for large scale
				synthetic experiments
		show_print_statements: If true, shows all print statements

	=#
	# Formatting output file name
	datestring = Dates.format(now(), "e_d_u_Y_HH_MM")
	if experimental_data
		fname_string = joinpath(
			relative_path, 
			"astar_results_"*data_string*"_"*datestring)
	else
		fname_string = joinpath(
			relative_path, 
			("astar_results_synthetic_n" * 
			string(n_species) * "_t" * string(n_temps) * 
			"_i" * string(replicate) * "_" * datestring))
	end
	CSV.write(fname_string*".csv", results_table)
	if show_print_statements
		println(stderr,"\n* Saving data to ", fname_string*".csv")
	end

	# Writing output files
	open(fname_string*".txt", "a") do io
		println(io, "==================================")
		println(io, "Setup:")
		println(io, "N: ", n_species)
		println(io, "T: ", n_temps)
		println(io, "add: ", params_astar.add_cost)
		println(io, "del: ", params_astar.del_cost)
		println(io, "nat: ", params_astar.wait_cost)
		println(io, "temp: ", params_astar.temp_cost)
		if experimental_data
			println(io, "Source: Data set")
			println(io, "Data set: ", data_string)
			println(io, "Species names: ", species_names)
			println(io, 
				"Species label: ", 
				[x for x = 1:length(species_names)])
			println(io, "Temperature names: ", temperature_list)
			println(io, 
				"Temperature label: ", 
				[x for x = 1:length(temperature_list)])
		else
			println(io, "Source: Generated")
			println(io, "A parameters: ", A_sampler)
			println(io, "r parameters: ", r_sampler)
			println(io, "portion: ", portion)
			println(io, "scaling: ", scaling)
		end
		println(io)

		println(io, "==================================")
		println(io, "Statistics:")
		println(io, 
			"Proportion of candidate states: ", 
			length(candidate_states)*1.0/nrow(assemblage))
		println(io, "Candidate states: ", length(candidate_states))
		println(io, "Total states: ", nrow(assemblage))
		println(io, "Total candidate pairs evaluated: ", 
			length(candidate_states_pairs))
		println(io, "Total natural transitions: ", nrow(transitions_natural))
		println(io)
		println(io, "Path length diff avg: ", results_stats["avg_length_imp"])
		println(io, "Path length diff val: ", results_stats["length_imp_val"])
		println(io, "Path length diff count: ", results_stats["length_imp_count"])
		println(io)
		println(io, "Cost improvement avg: ", results_stats["avg_cost_imp"])
		println(io, "Cost improvement val: ", results_stats["cost_imp_val"])
		println(io, "Cost improvement count: ", results_stats["cost_imp_count"])
		println(io)

		println(io, "==================================")
		println(io, "A matrices:")
		for i=1:n_temps
			println(io,"(T = ", i, ")")
			show(io, "text/plain", params_astar.A_matrices[i])
			println(io)
		end

		println(io, "\n==================================")
		println(io, "r vectors:")
		for i=1:n_temps
			println(io, "(T = ", i, ")")
			show(io, "text/plain", params_astar.r_vectors[i])
			println(io)
		end

		# # Uncomment this if you want to print out all natural transitions
		# println(io, "\n==================================")
		# println(io, "All natural transitions:")
		# for i=1:nrow(transitions_natural)
		# 	println(
		# 		io,
		# 		"(", transitions_natural[i,:from], ")", "->",  
		# 		"(", transitions_natural[i,:to], ")")
		# end
	end

	if show_print_statements
		println(stderr, "* Saving summary to ", fname_string*".txt")
	end
end
