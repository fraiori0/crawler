import pybullet as p
import numpy as np
import time
import pybullet_data
from math import *

physicsClient = p.connect(p.GUI)

p.setAdditionalSearchPath(pybullet_data.getDataPath())

p.setGravity(0,0,-9.81)

planeId = p.loadURDF("plane100.urdf", [0,0,0])

p.setAdditionalSearchPath("/home/fra/Uni/Tesi/crawler")
crawlerId = p.loadURDF("crawler.urdf", [0,0,0.2], flags=p.URDF_USE_INERTIA_FROM_FILE)

def COM_position_world(bodyUniqueId):
    #return the position of the center of mass in the world coordinates, as a NUMPY ARRAY
    COM = np.asarray(p.getBasePositionAndOrientation(bodyUniqueId)[0])
    total_mass = p.getDynamicsInfo(bodyUniqueId,-1)[0]
    for i in range(0,p.getNumJoints(bodyUniqueId)):
        link_COM_pos = np.asarray(p.getLinkState(bodyUniqueId, i)[0])
        link_mass = p.getDynamicsInfo(bodyUniqueId, i)[0]
        COM += link_COM_pos*link_mass
        total_mass += link_mass
    COM = (COM/total_mass)
    return COM
def COM_velocity_world(bodyUniqueId):
    #return the linear velocity of the center of mass in the world coordinates, as a NUMPY ARRAY
    COMv = np.asarray(p.getBaseVelocity(bodyUniqueId)[0])
    total_mass = p.getDynamicsInfo(bodyUniqueId,-1)[0]
    for i in range(0,p.getNumJoints(bodyUniqueId)):
        link_COM_vel = np.asarray(p.getLinkState(bodyUniqueId, i, computeLinkVelocity=1)[6])
        link_mass = p.getDynamicsInfo(bodyUniqueId, i)[0]
        COMv += link_COM_vel*link_mass
        total_mass += link_mass
    COMv = (COMv/total_mass)
    return COMv
def turn_off_joint(bodyUniqueId, joint_index):
    p.setJointMotorControl2(bodyUniqueId, joint_index, controlMode=p.VELOCITY_CONTROL, force=0)
    return
def turn_off_robot(bodyUniqueId):
    for i in range(0,p.getNumJoints(bodyUniqueId)):
        p.setJointMotorControl2(bodyUniqueId, i, controlMode=p.VELOCITY_CONTROL, force=0)
    return
def get_crawler_control_index(crawlerID):
    #this function relies on knowledge of the order of the joints in the crawler model
    #if the URDF is modified in ways different than just adding more segments to the spine this function should be updated properly
    lat_joints_i = tuple(range(1,(p.getNumJoints(crawlerID)-4),2))
    r_girdle_flex_i = p.getNumJoints(crawlerID)-4
    r_girdle_abd_i = p.getNumJoints(crawlerID)-3
    l_girdle_flex_i = p.getNumJoints(crawlerID)-2
    l_girdle_abd_i = p.getNumJoints(crawlerID)-1
    return (lat_joints_i,r_girdle_flex_i,r_girdle_abd_i,l_girdle_flex_i,l_girdle_abd_i)
def crawler_control(crawlerID, control_i):
    #the index of the joints to be controlled are considered to be the tuple generated through get_crawler_control_index
    #control_i = ([indices of lateral spine joint], right girdle flexion, right girdle abduction, left girdle flexion, left girdle abduction)
    for i in control_i[0]:
        p.setJointMotorControl2(crawlerID,i,p.POSITION_CONTROL, targetPosition=0,force=20)
    p.setJointMotorControl2(crawlerID,control_i[1],p.POSITION_CONTROL,targetPosition=0,force=20)
    p.setJointMotorControl2(crawlerID,control_i[2],p.POSITION_CONTROL,targetPosition=0,force=20)
    p.setJointMotorControl2(crawlerID,control_i[3],p.POSITION_CONTROL,targetPosition=0,force=20)
    p.setJointMotorControl2(crawlerID,control_i[4],p.POSITION_CONTROL,targetPosition=0,force=20)

    #might be nice to add computed torque control with an adaptive part for estimating the torques required to counteract friction, but must be formulated properly
    return

#for i in range (500):
#  p.stepSimulation()
#  p.applyExternalForce(crawlerId, -1, [np.random.random_sample(), 3*np.random.random_sample(), np.random.random_sample()], [0, 0, 0], flags=p.WORLD_FRAME)
#  time.sleep(1./240.)

turn_off_robot(crawlerId)
control_indices=get_crawler_control_index(crawlerId)
#time-step used for simulation and total time passed variable
dt = 1./240.
t = 0.0
for i in range (9000):
    #before stepping the simulation
    COM_prev = COM_position_world(crawlerId)

    p.stepSimulation()
    p.applyExternalForce(crawlerId, -1, [0, np.random.random_sample(), 2*np.random.random_sample()], [0, 0, 0], flags=p.LINK_FRAME)
    crawler_control(crawlerId,control_indices)
    #after stepping the simulation
    COM_curr = COM_position_world(crawlerId)
    p.addUserDebugLine(COM_prev.tolist(), COM_curr.tolist(), lineColorRGB=[sin(4*pi*t),sin(4*pi*(t+0.33)),sin(4*pi*(t+0.67))],lineWidth=1.5, lifeTime=2)
    time.sleep(dt)
    t+=dt

p.disconnect()
