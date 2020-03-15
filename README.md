# GW-Localization-Tiling
GW localization maps on the sphere need to be tiled so CTA can determine what parts of the sky to look at.

Described here are the codes from https://arxiv.org/abs/2003.04839

The most important files are Sphere_GA.py, Flat_GA.py, and Sphere_Light.py

To install, just clone this directory to a local machine.

# Flat_GA.py and Sphere_GA.py
This is the file to use for the genetic algorithm on a flat Euclidean geometry and spherical geometry respectively. In the files, there are 2 main parameters, 1 main function along with there are 6 important helper functions (which are the same 6 described in the pseudocode of the paper). 

The two main parameters are population and generations. The population describes the initial population of the genetic algorithm. This means if you set the population to 10, there will be 10 agents created initially for the algorithm. The generations describe the max number of generations we want to run for. Note this max number is not always reached since sometimes agents can die out too quickly or converge on the solution quickly in which case the program ends. 

The main function is ga() which takes in several arguments. First, a shapely object which is the polygon you are attempting to cover. By default, a random polygon is passed in. Additionally, the radius of the circle has to be specified. Perhaps in a later version of this program, one can specify a range of radii to select from (as the algorithm doesn't really care if it's a constant radius or not). In the case of the Flat algorithm, a bounding_box must be specified. This is just the plotting bounds we are interested in. The initial length of each agent must also be specified. It is very helpful to pick a smart initial length, picking a number like 1000 for a polygon which can be covered with 10 circles leads to a lot of unnecessary computation. Finally, there are 3 plotting arguments: "plot_regions", "save_agents", "plot_crossover". These will auto-populate you directory with 3 folders: "repair_frames", "crossover_frames", and "saved_agents". In these directories, the program will create sub-directories corresponding to the generation number and agent number of each agent that is saved. In the "plot_regions" directory, for each agent 2 images will be created. One is of the agent before the BFGS repair and one is after the BFGS repair. In the "crossover_frames" directory 6 images will be created. The two parents, the two parent's Voronoi diagrams, and the 2 children. In the "saved_agents" directory the agents will be saved as .obj files which can be read later. This is mainly for debugging purposes. 

The repair_agents function takes a list of agents and returns a new list of agents that are repaired. Note it may not always be the case that the initial length of the agent list will be the final length of the agent list. This is because not all agents can be repaired. 

The init_agents function creates the number of agents requested with the radius specified.

The fitness function determines the fitness of each agent. Note that fitness is a property of the agent and is initialized at -1000 (an arbitrary value). In this fitness function, you can specify the hyperparameters alpha, beta, and gamma as described in the paper.

The selection function will delete the specified number of agents from the list. At default it is set to .8 meaning 20% of the agents will be removed.

The crossover function takes a list of agents and randomly breeds members using the Voronoi scheme described in the paper.

The mutation function takes a list of agents and mutates them in 3 different ways: It will remove the circle which self-intersects the most and removes it, it will find the circle which intersects the specified region the least and remove it, it will move a random circle. 

Additionally, there is an "Agent" class in the code. This contains basic methods for operations on the agents. 

The only real difference between the sphere_GA and Flat_GA are the methods for calculating geometry operations. In order for the sphere_GA to work, you must download the "spherical_geometry" package using ```pip install spherical-geometry```. Alternatively, you can install it manually at https://github.com/spacetelescope/spherical_geometry. 

As a final note, for the Sphere_GA code note that you must specify a fits file in the data folder. If you aren't concerned with LIGO localization maps and instead only care about spherical polygon you can directly enter the spherical_geometry polygon as the variable labeled "region".

# Sphere_Light.py
This is not a genetic algorithm but it uses BFGS optimization to cover a spherical polygon. First, you must give the algorithm a .fits file that contains your localization map. Then, you must specify a range between which the algorithm will try to cover the region. For example, if I set intial_guess to 6 and max_circles to 100, it will iterate between 6 and 100 circles and try to cover the region. As soon as it is able to do so, the program will "converge". 
