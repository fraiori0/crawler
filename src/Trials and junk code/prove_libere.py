import pybullet as p
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import time
import pybullet_data
import crawler as crw
from math import *
import pinocchio as pin
import pyparsing


matplotlib.use('TkAgg')

urdf_path="/home/fra/Uni/Tesi/crawler"
urdf_filename=urdf_path+"/crawler.urdf"

#########################################
####### SIMULATION SET-UP ###############
#########################################
### time-step used for simulation and total time passed variable
dt = 1./240.
#
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0,0,-9.81)
#p.setGravity(0.0,0.0,0.0)
#planeID = p.loadURDF("plane100.urdf", [0,0,0])
model = crw.Crawler(urdf_path=urdf_path, dt_simulation=dt, base_position=[0,0,0.5], base_orientation=[0,0,0,1])
np.random.seed()

t_array = np.linspace(0,5,51)
i_array = np.arange(4)

A = ((1,2,3,4,5),(6,6),(7,7))
f1 = ((1,0.5,1,3,1),(1,1),(1,1))
n = ((2,1,2,4,2),(0,2),(1,2))
t2_off = ((0,1,0,1,1),(0.3,1),(1,0.7))
delta_off = ((1,2,0,1,1),(0.2,1),(0.3,1))
bias = ((0,0,0,0,0),(1,1),(-1,-1))

tau_array_fun = model.get_torques_profile_fun(A,f1,n,t2_off,delta_off,bias)

duration = 0.6
tau_time_array = model.generate_torques_time_array(tau_array_fun,duration)
tau_time_array_actuated = tau_time_array[:,(model.mask_act +model.mask_right_girdle+model.mask_left_girdle)]
fig,axs = plt.subplots(3,3)
for i in range(0,tau_time_array_actuated.shape[1]):
    axs[i//3,i%3].plot(tau_time_array_actuated[:,i], color="xkcd:dark teal", label="yoh")
handles, labels = axs[0,0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center')
plt.show()
print(model.traveling_wave_lateral_trajectory_t(0,0.5,1,3,0))
