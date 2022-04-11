# CoexistenceControl.jl
This repository contains code for the paper "Navigation between states in ecological communities and applications to control", by Benjamin W. Blonder*, Michael H. Lim*, Zachary Sunberg, Claire Tomlin (* denotes equal contribution).
The repository is organized into three main parts, /scripts, /data, and /src.

## Scripts _(/scripts)_
Scripts folder should contain the relevant scripts that allow us to run the numerical experiments.
1. **Experiment Configs**: Before running either the A* experiments or the CEM parameter search, we need to set up the configurations for the numerical experiments. This can be done via _/scripts/helper/experiment_config.jl_. The file lets us specify saving data, parallelization, experimental vs. synthetic, costs, dataset choices, and CEM parameter search hyperparameters. 

    _Note:_ "turning on" parallelization is unfortunately rather inefficient, since the best way we found that does not run into segfault errors for large _(N, T)_ was to create the _A, r_ matrices in each of the workers, which overshadows the incredibly quick A* search time. However, the scripts run fairly quickly, without the need for much parallelization as we anticipated in the beginning.
2. __A* Pathfinding Experiment__: In order to run the A* experiments, we can run the script _/scripts/AStar_sim.jl_ by the following:

       julia --project (path_to_CoexistenceControl.jl)/scripts/AStar_sim.jl

    This will automatically save the data in _/data/results_. If the experiment was experimental data, it will save the output txt & csv files with the appropriate dataset name under _/data/results/experimental_. If the experiment was synthetic data, it will save the output txt & csv files with the appropriate dataset name according to subfolders generated from the _(N, T)_, such as _/data/results/synthetic/n5_t3_.
3. **Cross-Entropy Parameter Search**: In order to run the CEM parameter search, we can run the script _/scripts/params_CEM.jl_ by the following:

       julia --project (path_to_CoexistenceControl.jl)/scripts/params_CEM.jl
       
    This will automatically save the data in _/data/dataset/synthetic_, as it will be the parameter data we will use for the A* experiments.

## Data _(/data)_
The datasets are stored in _/data/dataset_, under different first author names. The respective experimental _A, r_ values are stored in *a_matrix.csv* and *r_vector.csv* files. The CEM parameter search will also yield values that will be stored under _/data/dataset/synthetic_. The codebase currently contains all the experimental data used for the study. It is also possible to add new dataset by following the file organization convention described above, as well as adding an entry in the _/scripts/helper/load_data.jl_ and _/scripts/helper/experiment_config.jl_ files.

## Source _(/source)_
The source folder contains the source code. The code has been formatted with function descriptions and comments to improve readability. The functions exported in _/src/CoexistenceControl.jl_ can be called from outside this folder/package as well.
