import pybullet as p
import numpy as np
import time
import pybullet_data
import crawler
from math import *

physicsClient = p.connect(p.GUI)

p.setAdditionalSearchPath(pybullet_data.getDataPath())

p.setGravity(0,0,-9.81)

planeID = p.loadURDF("plane100.urdf", [0,0,0])

model = crawler.Crawler(spine_segments=8)

#for i in range (500):
#  p.stepSimulation()
#  p.applyExternalForce(crawlerID, -1, [np.random.random_sample(), 3*np.random.random_sample(), np.random.random_sample()], [0, 0, 0], flags=p.WORLD_FRAME)
#  time.sleep(1./240.)


#time-step used for simulation and total time passed variable
dt = 1./240.
t = 0.0
model.turn_off_crawler()
# model.set_feet_constraints(RL=(True,False))
# model.set_feet_constraints(RL=(True,True))
for i in range (9000):
    #before stepping the simulation
    COM_prev = model.COM_position_world()

    p.stepSimulation()
    p.applyExternalForce(model.Id, 14, [0, np.random.random_sample(), np.random.random_sample()], [0, 0, 0], flags=p.LINK_FRAME)
    #crawler_control(crawlerID,control_indices)
    #after stepping the simulation
    COM_curr = model.COM_position_world()
    p.addUserDebugLine(COM_prev.tolist(), COM_curr.tolist(), lineColorRGB=[sin(4*pi*t),sin(4*pi*(t+0.33)),sin(4*pi*(t+0.67))],lineWidth=3, lifeTime=2)
    time.sleep(dt)
    t+=dt
model.set_feet_constraints(RL=(True,False))
p.disconnect()
