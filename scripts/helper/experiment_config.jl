########################### A* Simulation ###########################
# Running experiment setup -- general
save_data = true  # Saving data to the CoexistenceControl/data folder
parallelize = false  # Parallelize tasks
procs_num = 6  # Parallel thread number

# Determine which studies we want to perform
experimental_simulation = true
synthetic_simulation = true

# Costs
add_cost = 1.0
del_cost = 3.0
wait_cost = 0.1
temp_cost = 5.0

# Experimental data setup:
# Choose among: 
# "Venturelli", "Bucci", "Maynard", "Carrara", 
# "Maynard15-19-23", "Maynard15-17-19-21-23"
experimental_data_set_names = [
	"Venturelli", "Bucci", "Maynard", "Carrara", 
	"Maynard15-19-23", "Maynard15-17-19-21-23"]

# Synthetic experiment setup
# Pairs of (N, T) to experiment on
nt_pair_sets = [(5, 3), (10, 1), (15, 1)]
max_replicates = 100  # Number of replicates for a simulation experiment 
max_candidates = 1_000  # Max number of candidate pairs (i, j) to evaluate


########################### CEM Parameter Search ###########################
# For CEM
ratio_bounds = [0.2, 0.4]  # aiming for 20-40% stable & feasible states
max_samples = 1_000  # Sample at most 1000 candidate states for larger (N, T) spaces
k_params = 90  # Number of parameter samples
n_sims = 20  # Number of evaluation simulations
m_elites = 30  # Number of elite samples
cem_iters = 30  # Number of CEM iterations
