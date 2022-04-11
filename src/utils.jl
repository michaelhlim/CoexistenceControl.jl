#= Utility functions and structs
=#
function Base.isequal(x::LVState, y::LVState) 
	# Base equality function override for LVState
	return (x.s == y.s) && (x.t == y.t)
end

function Base.:(==)(x::LVState, y::LVState) 
	# Base equality function override for LVState
	return (x.s == y.s) && (x.t == y.t)
end

function Base.hash(x::LVState, h::UInt)
	# Base hash function override for LVState
	hash(vcat(x.s, x.t), h)
end

function _subparams(temperature::Int64, params::LVParamsFull)
	#= Extract the subparameter information for a given temperature.
		
	Args:
		temperature: Temperature index
		params: LVParamsFull parameters set containing LV parameters
				for all temperatures

	Return:
		LVParams object
	=#
	return LVParams(
		params.A_matrices[temperature], 
		params.r_vectors[temperature], 
		params.collabels, 
		params.add_cost, 
		params.del_cost, 
		params.wait_cost)
end

function _generate_subparameters(state::Vector{Int64}, params::LVParams)
	#= Calculate the A, r subparameters according to 
	which states are present in the 'state' vector.
		
	Args:
		state: Vector of species occupancy
		params: LVParams object of full parameters

	Return:
		params: LVParams object that contains subparameters 
				(submatrix, subvector) only for species 
				that exist in the state
	=#
	# Find which species are present in that row
	species_indices_present = findall(x -> x>0, state)

	# Pick subset of parameters 
	# (assuming that the parameters don't change when subsetting)
	A_subset = [params.A_matrix[i,j] for i = species_indices_present, j = species_indices_present]
	r_subset = params.r_vector[species_indices_present]
  	
  	# Return params
	params = LVParams(
		A_subset, 
		r_subset, 
		species_indices_present, 
		params.add_cost, 
		params.del_cost, 
		params.wait_cost)

	return params
end

_generate_subparameters(state::Vector{Int64}, 
	temperature::Int64, 
	params::LVParamsFull) = _generate_subparameters(state, _subparams(temperature, params))

_generate_subparameters(assemblage::DataFrameRow, 
	temperature::Int64, 
	params::LVParamsFull) = _generate_subparameters(Array(assemblage), _subparams(temperature, params))

_generate_subparameters(assemblage::DataFrameRow, 
	params::LVParams) = _generate_subparameters(Array(assemblage), params)

function split_idx(n_species::Int64, n_temps::Int64, idx::Int64)
    #= Split the raw index into a tuple of (species, temp).
        
    Args:
        n_species: Number of species in the system
        n_temps: Number of temperature settings in the system
        idx: Raw index that varies from 0 to (2^n_species) * n_temps - 1
    
    Return:
        A tuple (idx_species, idx_temp)
        idx_species: Index for species space
        idx_temp: Index for temperature space
    =#
    idx_species = mod(idx-1, 2^n_species) + 1
    idx_temp = Int64(div(idx-1, 2^n_species)) + 1

    return (idx_species, idx_temp)
end

function state_str_to_idx(n_species::Int64, 
    str::AbstractString)
    #= Get state index from string (single temperature).
        
    Args:
        n_species: Number of species in the system
        str: Raw string of state
    
    Return:
        idx: Raw index of state
    =#
    # Empty state is index 1
    if length(str) == 0
        return 1
    end

    # Parse the string
    state_str = parse.(Int64, split(str, "*"))
    idx = 1
    for i in state_str
        idx += 2^(i-1)
    end

    return idx
end

function state_str_to_idx(n_species::Int64, 
    n_temps::Int64, 
    str::AbstractString)
    #= Get state index from string (multi temperature).
        
    Args:
        n_species: Number of species in the system
        n_temps: Number of temperature settings in the system
        str: Raw string of state
    
    Return:
        idx: Raw index of state
    =#
    str_split = split(str, "|")
    str_species = str_split[1]
    idx_temp = parse(Int64, str_split[2])

    # Get index from sub function
    idx_species = state_str_to_idx(n_species, str_species)

    # Augment by temperature
    idx = idx_species + (idx_temp-1)*2^n_species

    return idx
end

function state_idx_to_vec(n_species::Int64, idx::Int64)
    #= Get state vector from index (single temperature).
        
    Args:
        n_species: Number of species in the system
        idx: Raw index of state
    
    Return:
        s_vec: Vector of state
    =#
    # Remember that indexing starts at 1
    s = bitstring(idx-1)
    s = reverse(s[end-n_species+1:end])
    s_vec = [parse(Int64, ss) for ss in split(s, "")]

    return s_vec
end

function state_idx_to_str(n_species::Int64, idx::Int64)
    #= Get state string from index (single temperature).
        
    Args:
        n_species: Number of species in the system
        idx: Raw index of state
    
    Return:
        str: Raw string of state
    =#
    # Get vectorized state
    species_from_idx = findall(x -> x>0, state_idx_to_vec(n_species, idx))
    str = join(species_from_idx, "*")

    return str
end

function state_idx_to_str(n_species::Int64, 
    n_temps::Int64, 
    idx::Int64)
    #= Get state string from index (multi temperature).
        
    Args:
        n_species: Number of species in the system
        n_temps: Number of temperature settings in the system
        idx: Raw index of state
    
    Return:
        str: Raw string of state
    =#
    # Get the indices
    (idx_species, idx_temp) = split_idx(n_species, n_temps, idx)

    # Get string from sub function and append temperature
    str = state_idx_to_str(n_species, idx_species) * "|" * string(idx_temp)

    return str
end

function state_LV_to_idx(n_species::Int64, 
    n_temps::Int64, 
    s::LVState)
    #= Get state index from LVState (multi temperature).
        
    Args:
        n_species: Number of species in the system
        n_temps: Number of temperature settings in the system
        s: LVState variable
    
    Return:
        idx: Raw index of state
    =#
    # Parse the vector (Empty state is index 1)
    idx = 1
    for i in 1:length(s.s)
        idx += s.s[i] * 2^(i-1)
    end

    # Add in temperature indexing (temps go from 1,...,n_temps)
    idx += (s.t - 1) * 2^(n_species)

    return idx
end

function state_idx_to_LV(n_species::Int64, 
    n_temps::Int64, 
    idx::Int64)
    #= Get LVState from state index (multi temperature).
        
    Args:
        n_species: Number of species in the system
        n_temps: Number of temperature settings in the system
        idx: Raw index of state
    
    Return:
        s: LVState variable
    =#
    # Find out which temperature zone we are in (indexing starts at 1)
    (idx_species, idx_temp) = split_idx(n_species, n_temps, idx)

    # Convert the state
    s = bitstring(idx_species-1)
    s = reverse(s[end-n_species+1:end])
    s_vec = [parse(Int64, ss) for ss in split(s, "")]
    
    return LVState(s_vec, idx_temp)
end

function operation_distance(n_species::Int64, 
    idx1::Int64, 
    idx2::Int64)
    #= Get operation distance between state 1 and 2. 
    (single temperature)

    Operation distance is defined by the minimum number of 
    operations/actions one needs to perform to transition from 
    state 1 to state 2. This is equivalent to the number of
    species presence difference between state 1 and 2.
        
    Args:
        n_species: Number of species in the system
        idx1: Raw index of state 1
        idx2: Raw index of state 2
    
    Return:
        Int object of operation distance
    =#
    # Return 0 if same
    if idx1 == idx2
        return 0
    end

    # Make them into arrays of species
    state1 = state_idx_to_vec(n_species, idx1)
    state2 = state_idx_to_vec(n_species, idx2)
    state_diff = state1 - state2

    return count(i->(i!=0), state_diff)
end

function operation_distance(n_species::Int64, 
    n_temps::Int64, 
    idx1::Int64, 
    idx2::Int64)
    #= Get operation distance between state 1 and 2. 
    (multi temperature)

    Operation distance is defined by the minimum number of 
    operations/actions one needs to perform to transition from 
    state 1 to state 2. This is equivalent to the number of
    species presence difference between state 1 and 2, added with
    the number of temperature step difference.
        
    Args:
        n_species: Number of species in the system
        n_temps: Number of temperature settings in the system
        idx1: Raw index of state 1
        idx2: Raw index of state 2
    
    Return:
        Int object of operation distance
    =#

    # Get the indices
    (idx_species1, idx_temp1) = split_idx(n_species, n_temps, idx1)
    (idx_species2, idx_temp2) = split_idx(n_species, n_temps, idx2)

    # Get temp-less operation distance
    dist = operation_distance(n_species, idx_species1, idx_species2)

    # Check how much temperature distance is there
    temp_op = abs(idx_temp1 - idx_temp2)
    
    return dist + temp_op
end

function state_distance(n_species::Int64, 
    idx1::Int64, 
    idx2::Int64, 
    add_cost::Float64, 
    del_cost::Float64)
    #= Get state distance between state 1 and 2. 
    (single temperature)

    State distance is defined by the minimum cost required in order
    to transition from state 1 to state 2, assuming no natural transition.
        
    Args:
        n_species: Number of species in the system
        idx1: Raw index of state 1
        idx2: Raw index of state 2
        add_cost: Cost of addition
		del_cost: Cost of deletion
    
    Return:
        Float object of state distance
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
    del_ops = count(i->(i==1), state_diff)

    return (add_cost * add_ops + del_cost * del_ops)
end

function state_distance(n_species::Int64, 
    n_temps::Int64, 
    idx1::Int64, 
    idx2::Int64, 
    add_cost::Float64, 
    del_cost::Float64, 
    temp_cost::Float64)
    #= Get state distance between state 1 and 2. 
    (multi temperature)

    State distance is defined by the minimum cost required in order
    to transition from state 1 to state 2, assuming no natural transition.
        
    Args:
        n_species: Number of species in the system
        n_temps: Number of temperature settings in the system
        idx1: Raw index of state 1
        idx2: Raw index of state 2
        add_cost: Cost of addition
		del_cost: Cost of deletion
		temp_cost: Cost of temperature transition
    
    Return:
        Float object of state distance
    =#
    # Get the indices
    (idx_species1, idx_temp1) = split_idx(n_species, n_temps, idx1)
    (idx_species2, idx_temp2) = split_idx(n_species, n_temps, idx2)

    # Get temp-less state distance
    dist = state_distance(
        n_species, 
        idx_species1, 
        idx_species2, 
        add_cost::Float64, 
        del_cost::Float64)

    # Check how much temperature distance is there
    temp_op = abs(idx_temp1 - idx_temp2)
    
    return dist + temp_op * temp_cost
end

function net_species_gain(n_species::Int64, 
    n_temps::Int64, 
    idx1::Int64, 
    idx2::Int64)
    #= Get net species gain from state 1 and 2. 
    (multi temperature)
        
    Args:
        n_species: Number of species in the system
        n_temps: Number of temperature settings in the system
        idx1: Raw index of state 1
        idx2: Raw index of state 2
        add_cost: Cost of addition
		del_cost: Cost of deletion
		temp_cost: Cost of temperature transition
    
    Return:
        Float object of net species gain
    =#
    # Return 0 if same
    if idx1 == idx2
        return 0
    end

    # Get the indices
    (idx_species1, idx_temp1) = split_idx(n_species, n_temps, idx1)
    (idx_species2, idx_temp2) = split_idx(n_species, n_temps, idx2)

    # Make them into arrays of species
    state1 = state_idx_to_vec(n_species, idx_species1)
    state2 = state_idx_to_vec(n_species, idx_species2)
    net_gain = sum(state2) - sum(state1)
    
    return net_gain
end