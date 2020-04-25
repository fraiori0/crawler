# Crawler simulation

This repo contains files and materials from my Master's Thesis project at EPFL's BioRobotics Lab.

###
The files in this project implement features useful for the simulation of a "crawling" locomotion.
They rely on the open-source physics engine PyBullet and use features from the libraries Pinocchio, HyperOpt and the more common NumPy, SciPy, MatPlotLib, etc.

Since the Pinocchio library is distributed through Anaconda, a Conda environment should be set-up for properly running the code (see requirements.txt for the packages needed)


## Model generation (*.xacro file)

*crawler.urdf.xacro* is the file that can be used to generate a URDF model consistent with the rest of the code.
Note that, although the number of spinal joints can be changed parametrically (see first section of the XACRO file), the structure of the model should be kept the same in terms of:

- number of shoulder joints
- orientation of shoulder joints
- orientation of spinal joints (i.e. for the same spinal "segment", the lateral joint comes before the dorsal joint)

This parameters can be changed only if the *control_indices* parameter and all the masks for joint selection are updated accordingly in the *Crawler* class.

## Model control (*crawler.py* file)
crawler.py is the file that contain all the method for controlling the model during a simulation.

When an object of this class is instantiated, a "crawler" (generated from the specified URDF file) is added to the current running PyBullet simulation. 
The internal methods of this class can be called to keep track of the state of the model, compute inverse kinematics, add or remove constraints, compute the Computed Torque Control required for a specified trajectory, applying torques to the joints, and more.

**WARNING:** the numerical parameters (number of spinal segments, body length, etc.) at the start of the *Crawler.\__init__()* method should be matched manually to the one used in the XACRO to generate the URDF model. Other parameters are automatically set from these.


## Simulation run and control (*sim_\*.py* files)
The purpose of each one of this files is mainly to run a simulation and to sow the data.

They are all build around the *run_simulation()* function, which take as an input the desired settings for the simulation (see example), run the simulation, and, if prompted, save a video and print graphs representing joints' state and COM's trajectory of the model during the simulation.

The are built following this structure:

1. Start a PyBullet's simulation server, import the floor and instantiate a crawler object
2. Set-up the simulation;
   * Friction parameters: floor's friction values are set to 1, since the actual friction coefficient between two objects is the product of their associated friction parameters;
   * Set compliance of crawler's dorsal joints and model starting position
   * Set parameters to be used by the controller (e.g. gains) and initialize filters/integrators
   * Define parameters for the loss function
   * Initialize joints trajectories and other arrays for logging the data during the simulation. If the desired joints trajectories are already pre-determined (as for traveling wave motion) they can already be set, otherwise this arrays will be filled with the desired joint trajectories generated reactively at run time (e.g. when trying to keep a null COM's y-velocity)
   * Set the desired constraints
3. Run the simulation. Once per time-step
   * Update model's internal state (*WARNING:* must be done once and only once per time-step);
   * Run control routine (e.g. generate trajectories, compute torques to follow those trajectories, apply torques);
   * (optional) save video frame;
   * Update the loss function;
   * Update arrays for data logging
   * Step the simulation
   * (optional) check whether instabilities or unrealistic behaviour have arisen; this can be used when running optimizations to discard unrealistical beahaviours that cannot be predicted from the set-up

*run_simulation()* include an option for running the simulation in DIRECT mode (not showing the GUI, it runs faster), meant to be used along with optimization.

Additionally, they can contain functions for the set-up and run of optmizations; in the current code there are instances of this when using HyperOpt.

---

### Note
Refer to docstrings for detailed documentation on functions, especially for *Crawler.py*
