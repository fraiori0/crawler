import pybullet as p
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import time
import pybullet_data
import crawler as crw
from math import *

matplotlib.use('TkAgg')

physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0,0,-9.81)
planeID = p.loadURDF("plane100.urdf", [0,0,0])
model = crw.Crawler(spine_segments=8,base_position=[0,0,0.1], base_orientation=[0,0,0,1])

np.random.seed()
#time-step used for simulation and total time passed variable
dt = 1./240.
t = 0.0
###
model.turn_off_crawler()
###
low_pass = crw.Discrete_Low_Pass(dt,tc=0.5,K=1)
eu0 = np.array((0,0,0))
eu1 = np.array((0,0,0))
eud_listx = list()
eud_listx_filtered = list()
#let the model fall
for i in range (50):
    p.stepSimulation()
    time.sleep(dt)


### fix right foot
model.fix_right_foot()
for i in range (100):
    ###
    # f=[(0*0.5-np.random.random_sample()), 0.01*(0.5-np.random.random_sample()),
    #     0*(0.5-np.random.random_sample())]
    #print(f)
    # p.applyExternalForce(model.Id, 
    #     -1, f,
    #     [0, 0, 0], flags=p.LINK_FRAME)
    COM_prev = model.COM_position_world()
    #eu0 = eu1
    ###
    ###
    p.stepSimulation()
    t+=dt
    ###
    ###
    ### right leg stance control
    t_stance = 0.5
    thetaR0 = p.getJointState(model.Id, model.control_indices[1])[0]
    thetaRf = -pi/4
    thetaR = (thetaR0+thetaRf)/2 + (thetaR0-thetaRf)*cos(pi*t/t_stance)/2
    p.setJointMotorControl2(model.Id, 
        model.control_indices[1],
        p.POSITION_CONTROL,
        targetPosition = thetaR,
        force = 1,
        positionGain=1,
        velocityGain=0.5)
    # ### control spine's lateral joints
    e = model.control_spine_lateral(K=1,fmax=0.1)
    print("e = ", np.round(e,5))
    ### show COM trajectory
    COM_curr = model.COM_position_world()
    p.addUserDebugLine(COM_prev.tolist(), COM_curr.tolist(), lineColorRGB=[sin(4*pi*t),sin(4*pi*(t+0.33)),sin(4*pi*(t+0.67))],lineWidth=3, lifeTime=2)
    ###
    time.sleep(dt)

# print(model.solve_null_COM_y_speed())

# print("link %d" %6)
# model.test_link_COM_jacobians(6)
# print("\n")

# for i in range(0,20):
#     print("link %d" %i)
#     print("COM position base world: ", np.asarray(model.links_state_array[i]["world_com_trn"]
#         -np.asarray(p.getBasePositionAndOrientation(model.Id)[0])))
#     model.test_link_COM_jacobians(i)
#     print("\n")

p.disconnect()
