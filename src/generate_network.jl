#= Helper script for generating network and transitions.
=#

function make_network(n_species::Int64, 
	n_temps::Int64, 
	assemblage, 
	params::LVParamsFull; 
	include_single = true, 
	include_natural = true, 
	remove_loop = true,
	show_print_statements = true)
	#= Function that generates the transition network given an assemblage.

	Args:
		n_species: Number of species (N)
		n_temps: Number of temperature settings (T)
		assemblage: Assemblage DataFrame
		params: Parameters, which contains A, r matrices and cost values
	
	Keyword Args:
		include_single: Include all the single species addition and deletion 
				(artificial actions)
		include_natural: Include all the natural transitions (natural actions)
		remove_loop: Remove any self-moves for natural transitions
		show_print_statements: If true, shows all print statements

	Returns:
		transitions_df: DataFrame of all the transitions ("network")
	=#

	# Transition network, which is a dataframe
	transitions_df = DataFrame(
		from = String[], 
		to = String[], 
		from_idx = Int64[], 
		to_idx = Int64[], 
		type = String[], 
		weight = Float64[])

	# Labels and indices for natural transitions
	transition_from = String[]
	transition_to = String[]
	transition_from_idx = Int64[]
	transition_to_idx = Int64[]
	for i = 1:nrow(assemblage)
		species_from_ones = findall(x -> x>0, Array(assemblage[i,1:n_species]))
		species_to_ones = findall(x -> x>0, Array(assemblage[i,end-n_species+1:end]))
		from_str = join(species_from_ones, "*") * "|" * string(assemblage[i,:temp])
		to_str = join(species_to_ones, "*") * "|" * string(assemblage[i,:temp])

		push!(transition_from, from_str)
		push!(transition_to, to_str)
		push!(transition_from_idx, state_str_to_idx(n_species, n_temps, from_str))
		push!(transition_to_idx, state_str_to_idx(n_species, n_temps, to_str))
	end

	# Append all the single species addition and deletion if asked
	if include_single
		# Obtain all the single species addition and deletion first 
		# - using a for loop in Julia is fast -- no need to construct 
		# a pairwise matrix as it is done in the example script
		transitions_df = DataFrame(
			from = String[], 
			to = String[], 
			from_idx = Int64[], 
			to_idx = Int64[], 
			type = String[], 
			weight = Float64[])
		assemblage_states = Matrix(assemblage[:,1:n_species])
		assemblage_temps = Vector(assemblage[:,:temp])

		if show_print_statements
			println(
				"Generating single species connections for A* graph with N = "
				* string(n_species) * ", T = " * string(n_temps) 
				* ", including natural: " * string(include_natural)
				* "."
				)
		end
		prog_bar = Progress(nrow(assemblage); 
			showspeed = true,
			enabled = show_print_statements
			)
		
		for i in 1:nrow(assemblage)
			old_state = LVState(assemblage_states[i, :], assemblage_temps[i])

			# Sequential temperature changes
			temps = Int64[]
			if (old_state.t < n_temps) push!(temps, old_state.t + 1) end
			if (old_state.t > 1) push!(temps, old_state.t - 1) end

			for t in temps
				new_state = LVState(assemblage_states[i, :], t)
				new_state_idx = state_LV_to_idx(n_species, n_temps, new_state)

				# Forward direction
				push!(transitions_df, [
					state_idx_to_str(n_species, n_temps, i), 
					state_idx_to_str(n_species, n_temps, new_state_idx), 
					i, new_state_idx, 
					"temp", params.temp_cost])

				# Reverse direction
				push!(transitions_df, [
					state_idx_to_str(n_species, n_temps, new_state_idx), 
					state_idx_to_str(n_species, n_temps, i), 
					new_state_idx, i, 
					"temp", params.temp_cost])
			end

			# Additions and deletions
			for j in 1:n_species
				# Get new change only in the addition direction 
				# (deletion is reverse edge)
				if assemblage_states[i, j] == 0
					ds = zeros(Int64, n_species)
					ds[j] = 1
					new_state = LVState(assemblage_states[i, :] + ds, old_state.t)
					new_state_idx = state_LV_to_idx(
						n_species, n_temps, new_state)

					# Forward direction (addition)
					push!(transitions_df, [
						state_idx_to_str(n_species, n_temps, i), 
						state_idx_to_str(n_species, n_temps, new_state_idx), 
						i, new_state_idx, 
						"addition", params.add_cost])

					# Reverse direction (deletion)
					push!(transitions_df, [
						state_idx_to_str(n_species, n_temps, new_state_idx), 
						state_idx_to_str(n_species, n_temps, i), 
						new_state_idx, i, 
						"deletion", params.del_cost])
				end
			end

			next!(prog_bar)
		end
	end

	# Append all the natural transitions if asked
	if include_natural
		transitions_natural_df = DataFrame(
			from = transition_from, 
			to = transition_to, 
			from_idx = transition_from_idx, 
			to_idx = transition_to_idx, 
			type = "natural", 
			weight = params.wait_cost)
		transitions_df = vcat(transitions_df, transitions_natural_df)

		# Remove duplicate rows that have same (from, to)
		transitions_df = combine(
			sdf -> sdf[argmin(sdf.weight), :], 
			groupby(transitions_df, [:from_idx, :to_idx]))
	end

	# Remove any self-loops if asked
	if remove_loop
		transitions_df = transitions_df[
			transitions_df[!, :from] .!= transitions_df[!, :to], :]
	end

	return transitions_df
end

function make_graph(n_species::Int64, 
	n_temps::Int64, 
	transitions_mat::Matrix{Float64})
	#= Function that makes a fully connectected graph 
	# from matrix of transition info.

	Args:
		n_species: Number of species (N)
		n_temps: Number of temperature settings (T)
		transitions_mat: Matrix of all the transitions ("network")
				1st column is from_index
				2nd column is to_index
				3rd column is cost/weight

	Returns:
		AStarStruct that contains the full graph and distance matrix
	=#

	# Constructing a directed graph and distance matrix
	distance_matrix = sparse(
		Int64.(transitions_mat[:, 1]),
		Int64.(transitions_mat[:, 2]),
		transitions_mat[:, 3],
		2^(n_species)*n_temps, 2^(n_species)*n_temps, min)
		
	graph = SimpleDiGraph(distance_matrix)

	return AStarStruct(graph, distance_matrix)
end

make_graph(n_species::Int64, n_temps::Int64, 
	transitions_df::DataFrame) = make_graph(
		n_species, n_temps,
		Matrix(transitions_natural[:, [:from_idx, :to_idx, :weight]]))