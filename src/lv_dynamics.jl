#= Main script that contains all the generation and configuration for assemblages.
=#
struct LVParams
	#= LVParams struct that contains the A, r matrices and costs 
		for a given system (fixed T).
		
	Args:
		A_matrix: A matrix
		r_vector: r vector
		collabels: Label of each columns, usually useful for determining 
				which species are actually present when dealing with 
				subsystem of A, r
		add_cost: Cost of addition
		del_cost: Cost of deletion
		wait_cost: Cost of wait (no artificial action)
	=#
	A_matrix::Array{Float64, 2}
	r_vector::Array{Float64, 1}
	collabels::Array{Float64, 1}
	add_cost::Float64
	del_cost::Float64
	wait_cost::Float64
end

struct LVParamsFull
	#= LVParamsFull struct that contains the A, r matrices and costs 
		for a given system (multiple T).
		
	Args:
		A_matrices: Array of A matrices for each temperature
		r_vectors: Array of r vectors for each temperature
		collabels: Label of each columns, usually useful for determining 
				which species are actually present when dealing with 
				subsystem of A, r
		add_cost: Cost of addition
		del_cost: Cost of deletion
		wait_cost: Cost of wait (no artificial action)
		temp_cost: Cost of temperature transition
	=#
	A_matrices::Array{Array{Float64, 2}, 1}
	r_vectors::Array{Array{Float64, 1}, 1}
	collabels::Array{Float64, 1}
	add_cost::Float64
	del_cost::Float64
	wait_cost::Float64
	temp_cost::Float64
end

struct LVState
    #= LVState struct that contains the state and temperature.
		
	Args:
		s: Current state, a binary vector that indicates which species are present
		t: Current temperature, discretized into T number of bins
	=#
    s::Array{Int64, 1}
    t::Int64
end

struct AStarStruct
	#= AStarStruct struct that contains the graph and distance matrix 
		of an assemblage transition network.
		
	Args:
		graph: Graph with all transitions present
		distmx: Distance matrix for all transitions in graph
	=#
	graph::AbstractGraph
	distmx::Array{Float64, 2}
end

struct FeasibilityStruct
	#= FeasibilityStruct struct that contains the information used to 
		calculate feasibility of a state.
		
	Args:
		feasibility: Boolean for indicating whether the struct is feasible
		new_state: State vector in which the species have nonzero survival
		abundance_mean: Mean species abundance for existing species
		abundance_sd: Stdev of species abundance for existing species
	=#
	feasibility::Bool
	new_state::Array{Int64, 1}
	abundance_mean::Float64
	abundance_sd::Float64
end

struct StabilityStruct
	#= StabilityStruct struct that contains the information used to 
		calculate stability of a state.
		
	Args:
		stability: Boolean for indicating whether the struct is stable
		lambda: Eigenvalue of D(x^*)*A, where x^* is the equilibrium point.
				x^* = -A^-1*r = 1/T \int_0^T x(t) dt
		tau: The time constant, which is inverse of the biggest eigenvalue 
				if exists
	=#
	stability::Bool
	lambda::Array{Float64, 1}
	tau::Float64
end

function generate_state_vec_set(n_species::Int64)
	#= Create all possible 2^n_species state vectors (single temperature).
		
	Args:
		n_species: Number of species in the system
	
	Return:
		state_vec_set: An Array of all possible states
	=#
	state_vec_set = Array{Int64,1}[]
	for i = 0:2^(n_species)-1
		s = bitstring(i)
		s = reverse(s[end-n_species+1:end])
		s_vec = [parse(Int64, ss) for ss in split(s, "")]
		push!(state_vec_set, s_vec)
	end
	
	return state_vec_set
end

function generate_assemblages(n_species::Int64, n_temps::Int64)
	#= Create assemblage of all possible (2^n_species) * n_temps states 
	(multi temperature), as well as their properties.
		
	Args:
		n_species: Number of species in the system
		n_temps: Number of temperature settings in the system
	
	Return:
		assemblage: A DataFrame of assemblage containing all possible states
				and their properties
	=#
	# Create all possible 2^n_species states for single temperature
	state_vec_set = generate_state_vec_set(n_species)

	# Set up state vectors and labels
	state_vec_set = transpose(hcat(state_vec_set...))
	state_vec_set = repeat(state_vec_set, n_temps)
	state_vec_label = [string("sp", i) for i in 1:n_species]

	new_state_vec_set = zeros(Int64, size(state_vec_set))
	new_state_vec_label = [string("sp", i, "_new") for i in 1:n_species]

	# Create assemblage data frame
	assemblage = DataFrame(state_vec_set, :auto)
	rename!(assemblage, state_vec_label)
	temp_col = DataFrame(temp = repeat(1:n_temps, inner=[2^n_species]))
	assemblage = hcat(assemblage, temp_col)

	assemblage_new = DataFrame(new_state_vec_set, :auto)
	rename!(assemblage_new, new_state_vec_label)

	# Create relevant columns
	insertcols!(assemblage, :stable => false)
	insertcols!(assemblage, :feasible => false)
	insertcols!(assemblage, :tau => 0.0)
	insertcols!(assemblage, :richness => 0)
	insertcols!(assemblage, :abundance_mean => 0.0)
	insertcols!(assemblage, :abundance_sd => 0.0)

	# Combine the old state and new state
	assemblage = hcat(assemblage, assemblage_new)

	for i = 1:nrow(assemblage)
		assemblage[i,:richness] = sum(assemblage[i,1:n_species])
	end

	return assemblage
end

function generate_params(n_species::Int64, 
	n_temps::Int64, 
	A_sampler::Distribution, 
	r_sampler::Distribution, 
	portion::Float64, 
	scaling::Float64, 
	add_cost::Float64, 
	del_cost::Float64, 
	wait_cost::Float64, 
	temp_cost::Float64, 
	rng::AbstractRNG; 
	A_diag = nothing, 
	A_matrices = nothing, 
	r_vectors = nothing)
	#= Generate LV parameters according to sampler distributions.
	If A and r are provided, then return LV parameters directly.
		
	Args:
		n_species: Number of species in the system
		n_temps: Number of temperature settings in the system
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

	Keyword Args:
		A_diag: If specified, diagonals of A matrix will be set to 
				the A_diag vector
		A_matrices: If specified, A matrices for all temperatures
				will be set directly to A_matrices
		r_vectors: If specified, r vectors for all temperatures 
				will be set directly to r_vectors
	
	Return:
		params: An LVParamsFull object that contains A, r for all temperatures
	=#
	# Create A and r if none are provided:
	if isnothing(A_matrices) && isnothing(r_vectors)
		# Create A and r lists
		A_matrices = Matrix{Float64}[]
		r_vectors = Vector{Float64}[]

		# Generate A and r according to the sampler distributions,
		# medium temperature first
		A = collect(reshape(
			rand(rng, A_sampler, n_species*n_species), 
			n_species, n_species))
		r = rand(rng, r_sampler, n_species)

		# Set diagonal if requested
		if !isnothing(A_diag)
			A[diagind(A)] = A_diag
		end

		# Generate n_temps amount of A, r
		for i = 1:n_temps
			push!(A_matrices, A)
			push!(r_vectors, r)
		end

		# Generate hotter and colder configurations
		(A_matrices, r_vectors) = _augment_hot_and_cold(
			n_species, n_temps, A_matrices, r_vectors, portion, scaling, rng; 
			A_diag = A_diag)
	end

	# Return params
	params = LVParamsFull(A_matrices, 
		r_vectors, 
		1:n_species, 
		add_cost, 
		del_cost, 
		wait_cost, 
		temp_cost)

	return params
end

function _augment_hot_and_cold(n_species::Int64, 
	n_temps::Int64, 
	A_matrices::Vector{Matrix{Float64}},
	r_vectors::Vector{Vector{Float64}},
	portion::Float64, 
	scaling::Float64,
	rng::AbstractRNG;
	A_diag = nothing)
	#= Augment the hot and cold temperatures.
	For all temperature indices, it samples 'portion' amount of 
	random indices of A, r vectors and multiply (hot) or 
	divide (cold) the entries by 'scaling'.
	Skips the half_temperature, which will keep the original
	A, r value.
		
	Args:
		n_species: Number of species in the system
		n_temps: Number of temperature settings in the system
		A_matrices: A matrices for all temperatures (pre-adjustment)
		r_vectors: r vectors for all temperatures (pre-adjustment)
		portion: For varying temperatures, we sample 'portion' amount 
				of indices in which we scale up/down for
		scaling: The scaling factor for varying temperatures
		rng: RNG object that can define the seed

	Keyword Args:
		A_diag: If specified, diagonals of A matrix will be set to 
				the A_diag vector
	
	Return:
		A_matrices: A matrices for all temperatures (post-adjustment)
		r_vectors: r vectors for all temperatures (post-adjustment)
	=#
	# Determine half temperature index, and hot/cold temp range
	half_temp = Int64(floor((n_temps+1)/2))
	
	# Iterate over all temperatures to augment the A, r
	# Skip the half_temp index
	for j = 1:n_temps
		if j > half_temp
			# Hot temperature - scale up
			i = j
			alt_scaling = scaling
			A = copy(A_matrices[i-1])
			r = copy(r_vectors[i-1])
		elseif j < half_temp
			# Cold temperature - scale down
			i = half_temp - j
			alt_scaling = 1 / scaling
			A = copy(A_matrices[i+1])
			r = copy(r_vectors[i+1])
		else
			# Half temperature - do nothing
			continue
		end

		# Sample portion amount of indices to scale compared to the previous temperature
		A_idx = StatsBase.sample(
			rng, 
			1:n_species^2, 
			Int64(floor(n_species^2*portion)), 
			replace=false)
		r_idx = StatsBase.sample(
			rng, 
			1:n_species, 
			Int64(floor(n_species*portion)), 
			replace=false)
		A[A_idx] .= A[A_idx] * alt_scaling
		r[r_idx] .= r[r_idx] * alt_scaling

		# Do not touch the diagonals
		if !isnothing(A_diag)
			A[diagind(A)] = A_diag
		end

		A_matrices[i] = A
		r_vectors[i] = r
	end

	return (A_matrices, r_vectors)
end

function determine_feasibility(params::LVParams)
	#= Determine feasibility of the (sub)parameter set.
	Feasibility is calculated by seeing if the resulting
	equilibrium state x^* = -A^-1*r has all positive values.
	Also returns the equilibrium state itself, as well as
	the abundance mean and sd.
		
	Args:
		params: LVParams object of (sub)parameters

	Return:
		FeasibilityStruct that contains the feasibility 
		information of the parameter set
	=#
	# Catch if the r vector is null
	if length(params.r_vector) > 0
		# New abundance vector via LV calculation
		x_vector = -1.0 * inv(params.A_matrix) * params.r_vector

		# Check if the solution has all positive values
		feasibility = !any(x->x<=0, x_vector)
		
		# Get the new state where the species 
		# have positive abundance
		new_state_idx = findall(x -> x>0, x_vector)
		new_state = params.collabels[new_state_idx]

		# Get mean and sd of residents
		abundance_mean = mean(x_vector[new_state_idx])
		abundance_sd = std(x_vector[new_state_idx])
	else
		feasibility = true
		new_state = Float64[]
		abundance_mean = NaN
		abundance_sd = NaN
	end

	return FeasibilityStruct(
		feasibility, 
		new_state, 
		abundance_mean, 
		abundance_sd)
end

function determine_stability(params::LVParams)
	#= Determine stability of the (sub)parameter set.
	Stability is calculated by seeing if the eigenvalues of
	D(x^*)*A are all negative, with equilibrium state
	x^* = -A^-1*r and D() the diagonalization operator.
	Also returns the tau, time constant.
		
	Args:
		params: LVParams object of (sub)parameters

	Return:
		Stability struct that contains the stability 
		information of the parameter set
	=#
	if length(params.r_vector) > 0
		# Equilibrium state
		x_vector = -1.0 * inv(params.A_matrix) * params.r_vector

		# Calculate eigenvalues and see if 
		# max(Re(lambda)) < 0 for all lambda's
		lambda = eigvals(Diagonal(x_vector) * params.A_matrix)
		if typeof(lambda[1]) == Complex{Float64}
			lambda = [z.re for z in lambda]
		end
		stability = !any(x->x>=0, lambda)

		# Time constant is inverse of 
		# the biggest eigenvalue, if exists
		if stability
			tau = -1/maximum(lambda) 
		else
			tau = 0
		end
	else
		lambda = Float64[]
		stability = true
		tau = NaN
	end

	return StabilityStruct(stability, lambda, tau)
end

function generate_assemblage_transitions(n_species::Int64, 
	n_temps::Int64, 
	assemblage, 
	params::LVParamsFull;
	show_print_statements = true)
	#= Generate all assembly transitions for a given assemblage.
		
	Args:
		n_species: Number of species in the system
		n_temps: Number of temperature settings in the system
		assemblage: DataFrame object that contains 
				all assemblage information needed 
				to generate transitions
		params: An LVParamsFull object that contains 
				A, r for all temperatures
	
	Keyword Args:
		show_print_statements: If true, shows all print statements
	
	Return:
		assemblage: DataFrame object that contains 
				all assemblage information with transition
	=#
	if show_print_statements
		println(("Generating assemblage transitions with N = " 
				* string(n_species) * " and T = " * string(n_temps) * "...")
		)
	end
	prog_bar = Progress(nrow(assemblage); 
		showspeed = true,
		enabled = show_print_statements
		)
	
	for i = 1:nrow(assemblage) # Skip no species assemblage
		params_this = _generate_subparameters(
			assemblage[i,1:n_species], 
			_subparams(Int64(assemblage[i,n_species+1]), params)
			)
    
    	# Determine stability and feasibility
	    f = determine_feasibility(params_this)
	    s = determine_stability(params_this)
	    
	    assemblage[i, :stable] = s.stability
	    assemblage[i, :feasible] = f.feasibility
	    assemblage[i, :tau] = s.tau
	    assemblage[i, :abundance_mean] = f.abundance_mean
	    assemblage[i, :abundance_sd] = f.abundance_sd
	    
	    # Insert the new states determined by LV transition
	    if length(f.new_state) > 0
	    	for j in f.new_state
	    		assemblage[i, string("sp", j, "_new")] = 1
	    	end
	    end
		next!(prog_bar)
	end

	return assemblage
end

function portion_stable_feasible(n_species::Int64, 
	n_temps::Int64, 
	state_vec_set, 
	A_sampler::Distribution, 
	r_sampler::Distribution, 
	portion::Float64, 
	scaling::Float64, 
	A_diag::Vector{Float64}, 
	max_samples::Int64, 
	rng::AbstractRNG)
	#= Calculate the portion of feasible & stable configurations.
		
	Args:
		n_species: Number of species in the system
		n_temps: Number of temperature settings in the system
		state_vec_set: An Array of all possible states
		A_sampler: Distribution object which we sample the A matrix from
		r_sampler: Distribution object which we sample the r vector from
		portion: For varying temperatures, we sample 'portion' amount 
				of indices in which we scale up/down for
		scaling: The scaling factor for varying temperatures
		A_diag: Diagonals of A matrix
		max_samples: Maximum number of state configurations we want to
				determine whether they are feasible & stable
		rng: RNG object that can define the seed
	
	Return:
		Float in between [0, 1], representing the 
		proportion of states that are feasible & stable over
		all possible configurations or maximum sample 
		number of configurations
	=#
	# Generate parameters (don't need to worry about costs, set to 0.0)
	params = generate_params(
		n_species, 
		n_temps, 
		A_sampler, 
		r_sampler, 
		portion, 
		scaling, 
		0.0, 
		0.0, 
		0.0, 
		0.0, 
		rng; 
		A_diag = A_diag)
	
	# Subsample if asked
	n_states = length(state_vec_set)
	counts = 0.0

	if n_states * n_temps < max_samples
		# Find all species that are stable
		for i = 1:n_states
			for j = 1:n_temps
				params_this = _generate_subparameters(
					state_vec_set[i], j, params)
		    
		    	# Determine stability and feasibility
			    f = determine_feasibility(params_this)
			    s = determine_stability(params_this)
			    
			    if s.stability && f.feasibility
			    	counts += 1.0
			    end
			end
		end

		return counts / (n_states * n_temps)
	else
		# Find subsampled species that are stable
		states = StatsBase.sample(rng, 1:n_states, max_samples)
		temps = StatsBase.sample(rng, 1:n_temps, max_samples)
		
		for i = 1:max_samples
			params_this = _generate_subparameters(
				state_vec_set[states[i]], temps[i], params)
		    
	    	# Determine stability and feasibility
		    f = determine_feasibility(params_this)
		    s = determine_stability(params_this)
		    
		    if s.stability && f.feasibility
		    	counts += 1.0
		    end
		end

		return counts / max_samples
	end
end

function ratio_desired_configurations(n_species::Int64, 
	n_temps::Int64, 
	state_vec_set, 
	A_dist_type::AbstractString, 
	r_dist_type::AbstractString, 
	A_mean::Float64, 
	A_sig::Float64, 
	r_mean::Float64, 
	r_sig::Float64, 
	portion::Float64, 
	scaling::Float64, 
	n_iter::Int64, 
	ratio_bounds::Vector{Float64}, 
	max_samples::Int64, 
	rng::AbstractRNG)
	#= Calculate the ratio of desired configurations.
	
	Desired configuration here means that the portion of 
	stable and feasible configurations lie in between the
	'ratio_bounds'. Thus, it returns the ratio of times
	a parameter set generates desired parameter configurations.
		
	Args:
		n_species: Number of species in the system
		n_temps: Number of temperature settings in the system
		state_vec_set: An Array of all possible states
		A_dist_type: String for distribution type of A 
				(choice of "Normal" and "LogNormal")
		r_dist_type: String for distribution type of r 
				(choice of "Normal" and "LogNormal")
		A_mean: Mean of A entries
		A_sig: Standard deviation of A entries
		r_mean: Mean of r entries
		r_sig: Standard deviation of r entries
		portion: For varying temperatures, we sample 'portion' amount 
				of indices in which we scale up/down for
		scaling: The scaling factor for varying temperatures
		n_iter: Number of iterations to calculate the ratio of times
				we obtain a desired configuration
		ratio_bounds: [ratio_min, ratio_max] that determines the
				minimum and maximum desired ratio
		max_samples: Maximum number of state configurations we want to
				determine whether they are feasible & stable
		rng: RNG object that can define the seed
	
	Return:
		Float in between [0, 1], representing the 
		ratio of parameter configurations that are
		desired configurations
	=#
	# Set up distributions
	if A_dist_type == "Normal"
		A_sampler = Normal(A_mean, A_sig)
	else
		A_sampler = LogNormal(A_mean, A_sig)
	end
	if r_dist_type == "Normal"
		r_sampler = Normal(r_mean, r_sig)
	else
		r_sampler = LogNormal(r_mean, r_sig)
	end
	A_diag = repeat([-1.0], n_species)

	# Count up how many samples generated from the 
	# given parameter configuration lies within the bounds
	counts = 0.0
	for i in 1:n_iter
		ratio = portion_stable_feasible(
			n_species, 
			n_temps, 
			state_vec_set, 
			A_sampler, 
			r_sampler, 
			portion, 
			scaling, 
			A_diag, 
			max_samples, 
			rng)
		if ratio >= ratio_bounds[1] && ratio <= ratio_bounds[2]
			counts += 1.0
		end
	end

	return counts/n_iter
end

ratio_desired_configurations(n_species::Int64, 
	n_temps::Int64, 
	state_vec_set, 
	A_dist_type::AbstractString, 
	r_dist_type::AbstractString, 
	params::Vector{Float64}, 
	n_iter::Int64, 
	ratio_bounds::Vector{Float64}, 
	max_samples::Int64) = ratio_desired_configurations(
		n_species, 
		n_temps, 
		state_vec_set, 
		A_dist_type, 
		r_dist_type, 
		params[1], 
		params[2], 
		params[3], 
		params[4], 
		params[5], 
		params[6], 
		n_iter, 
		ratio_bounds, 
		max_samples, 
		MersenneTwister(Int64(params[7])))