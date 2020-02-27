import pybullet as p
import numpy as np
import time
import pybullet_data

physicsClient = p.connect(p.GUI)

p.setAdditionalSearchPath(pybullet_data.getDataPath())

p.setGravity(0,0,-9.81)

planeId = p.loadURDF("plane100.urdf", [0,0,-1])

p.setAdditionalSearchPath("/home/fra/Uni/Tesi/crawler")
crawlerId = p.loadURDF("crawler.urdf", [0,0,0.2])

#for i in range (500):
#  p.stepSimulation()
#  p.applyExternalForce(crawlerId, -1, [np.random.random_sample(), 3*np.random.random_sample(), np.random.random_sample()], [0, 0, 0], flags=p.WORLD_FRAME)
#  time.sleep(1./240.)

for i in range (9000):
  p.stepSimulation()
  time.sleep(1./240.)

p.disconnect()
