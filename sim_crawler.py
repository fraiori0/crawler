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
model = crw.Crawler(spine_segments=8,base_position=[0,0,1])
#time-step used for simulation and total time passed variable
dt = 1./240.
t = 0.0
###
model.turn_off_crawler()
low_pass = crw.Discretel_Low_Pass(dt,tc=0.5,K=1)
eu0 = np.array((0,0,0))
eu1 = np.array((0,0,0))
eud_listx = list()
eud_listx_filtered = list()
for i in range (100):
    ###
    # p.applyExternalForce(model.Id, 
    #     14, [0, 10*np.random.random_sample(),
    #     np.random.random_sample()],
    #     [0, 0, 0], flags=p.LINK_FRAME)
    COM_prev = model.COM_position_world()
    #eu0 = eu1
    ###
    p.stepSimulation()
    ### 
    COM_curr = model.COM_position_world()
    p.addUserDebugLine(COM_prev.tolist(), COM_curr.tolist(), lineColorRGB=[sin(4*pi*t),sin(4*pi*(t+0.33)),sin(4*pi*(t+0.67))],lineWidth=3, lifeTime=2)
    # eu1 = np.asarray(model.get_base_Eulers())
    # eud = (eu0-eu1)/dt
    # eud_f = np.asarray(
    #     (low_pass.filter((eu0[0]-eu1[0])/dt),
    #     low_pass.filter((eu0[1]-eu1[1])/dt),
    #     low_pass.filter((eu0[2]-eu1[2])/dt)
    #     )
    # )
    # eud_listx.append(eud[1])
    # eud_listx_filtered.append(eud_f[1])
    ###
    time.sleep(dt)
    t+=dt
qdot = np.asarray(p.getBaseVelocity(model.Id)[1] + p.getBaseVelocity(model.Id)[0] + model.get_joints_speeds_tuple())
j_COM_speed = np.dot(model.COM_trn_jacobian(),qdot)
print("True COM linear speed:  ", model.COM_velocity_world(),"\n","Linear speed computed w/ Jacobian:  ",j_COM_speed)
print(model.COM_trn_jacobian())

p.disconnect()
