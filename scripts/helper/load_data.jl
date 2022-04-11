########################### Data Processing ###########################
# Generate the necessary data components
experimental_data_set = DataFrame(
	n = Int64[], t = Int64[], names = String[], temp = Vector{String}[],
	rep = Int64[], candidates = Int64[], raw_names = String[])

if experimental_simulation
	for data_name in experimental_data_set_names
		if data_name == "Venturelli"
			# 12 Member Human Gut Community from Venturelli et al. 
			# https://www.embopress.org/doi/full/10.15252/msb.20178157
			# Number of species and temperature
			n_species = 12
			temperature_list = [""]
			file_name = "venturelli"

		elseif data_name == "Bucci"
			# 11 Member Mouse Gut Community from MDSINE series of papers, Bucci et al. 
			# https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003388
			# Number of species and temperature
			n_species = 11
			temperature_list = [""]
			file_name = "bucci"

		elseif data_name == "Carrara"
			# 11 Member Community from Carrara et al. 
			# https://besjournals.onlinelibrary.wiley.com/doi/full/10.1111/2041-210X.12363
			# Number of species and temperature
			n_species = 11
			temperature_list = [""]
			file_name = "carrara"

		elseif data_name == "Maynard"
			# 5 Member Community from Maynard et al. (default temperature is 17'C)
			# https://www.nature.com/articles/s41559-019-1059-z
			# Number of species and temperature
			n_species = 5
			temperature_list = ["17"]
			file_name = "maynard"

		elseif data_name == "Maynard15-19-23"
			# 5 Member Community from Maynard et al., 3 temperatures (15'C, 19'C, 23'C)
			# https://www.nature.com/articles/s41559-019-1059-z
			# Number of species and temperature
			n_species = 5
			temperature_list = ["15", "19", "23"]
			file_name = "maynard"

		elseif data_name == "Maynard15-17-19-21-23"
			# 5 Member Community from Maynard et al., 5 temperatures (15'C, 17'C, 19'C, 21'C, 23'C)
			# https://www.nature.com/articles/s41559-019-1059-z

			# Number of species and temperature
			n_species = 5
			temperature_list = ["15", "17", "19", "21", "23"]
			file_name = "maynard"

		else 
			error("Data error: "*data_name*" does not exist!")
		end

		n_temps = length(temperature_list)
		push!(experimental_data_set, [
			n_species, n_temps, file_name, temperature_list,
			1, typemax(Int64), data_name
		])
	end
end

if synthetic_simulation
	for (n, t) in nt_pair_sets
		push!(experimental_data_set, [
			n, t, "generated", [string(e) for e in 1:t],
			max_replicates, max_candidates, ""
		])
	end
end
