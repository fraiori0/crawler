import pybullet as p
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import time
import pybullet_data
import crawler as crw
from math import *
import subprocess as sp
import imageio

matplotlib.use('TkAgg')

#########################################
####### SIMULATION SET-UP ###############
#########################################
### time-step used for simulation and total time passed variable
dt = 1./1200.
#
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0,0,-9.81)
p.setTimeStep(dt)
#p.setGravity(0,0,0)
planeID = p.loadURDF("plane100.urdf", [0,0,0])
z_rotation = pi/4 #radians
model = crw.Crawler(
    urdf_path="/home/fra/Uni/Tesi/crawler",
    dt_simulation=dt,
    base_position=[0,0,0.5],
    base_orientation=[0,0,sin(z_rotation/2),cos(z_rotation/2)],
    mass_distribution=True,
    scale=1
    )
np.random.seed()
##########################################
####### MODEL SET-UP #####################
##########################################
# Remove motion damping ("air friction")
for index in range(-1,model.num_joints):
    p.changeDynamics(
        model.Id,
        index,
        linearDamping=0.01,
        angularDamping=0.01
    )
# Girdle link dynamic properties
p.changeDynamics(model.Id,
    linkIndex = -1,
    lateralFriction = 0.01,
    spinningFriction = 0.01,
    rollingFriction = 0.0,
    restitution = 0.1,
    #contactStiffness = 0,
    #contactDamping = 0
    )
# Body dynamic properties
for i in range(0,model.num_joints-4):
    p.changeDynamics(model.Id,
        linkIndex = i,
        lateralFriction = 0.01,
        spinningFriction = 0.01,
        rollingFriction = 0.01,
        restitution = 0.1,
        #contactStiffness = 0,
        #contactDamping = 0
        )
# # Last link dynamic properties
p.changeDynamics(model.Id,
    linkIndex = (model.control_indices[0][-1]+1),
    lateralFriction = 0.01,
    spinningFriction = 0.01,
    rollingFriction = 0.01,
    restitution = 0.1,
    #contactStiffness = 0,
    #contactDamping = 0
    )
# Right Leg dynamic properties
for i in range(model.num_joints-4,model.num_joints-2):
    p.changeDynamics(model.Id,
        linkIndex = i,
        lateralFriction = 0.01,
        spinningFriction = 0.01,
        rollingFriction = 0.01,
        restitution = 0.1,
        #contactStiffness = 0,
        #contactDamping = 0
        )
# Left Leg dynamic properties
for i in range(model.num_joints-2,model.num_joints):
    p.changeDynamics(model.Id,
        linkIndex = i,
        lateralFriction = 0.0001,
        spinningFriction = 0.0001,
        rollingFriction = 0.0001,
        restitution = 0.1,
        #contactStiffness = 0,
        #contactDamping = 0
        )
model.turn_off_crawler()



p.resetDebugVisualizerCamera(cameraDistance=0.6, cameraYaw = 50, cameraPitch=-60, cameraTargetPosition=[0,0,0])
theta_g0 = pi/4
theta_gf = -pi/3
# p.resetJointState(model.Id, model.control_indices[1][0],theta_g0)
# p.resetJointState(model.Id, model.control_indices[2][0],-theta_gf)
model.set_bent_position(theta_rg=theta_g0, theta_lg=-theta_gf,A_lat=-pi/3.5,theta_lat_0=0)
print(model.mass)
for i in range(1200):
    p.stepSimulation()
    time.sleep(dt)
print(p.getJointState(model.Id, model.control_indices[1][1]))
