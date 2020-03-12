import pybullet as p
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import time
import pybullet_data
import crawler as crw
from math import *

matplotlib.use('TkAgg')

#########################################
####### SIMULATION SET-UP ###############
#########################################
### time-step used for simulation and total time passed variable
dt = 1./240.
#
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0,0,-9.81)
planeID = p.loadURDF("plane100.urdf", [0,0,0])
model = crw.Crawler(urdf_path="/home/iori/Documents/Francesco_Iori_Thesis/crawler", dt_simulation=dt, base_position=[0,0,0.05], base_orientation=[0,0,0,1])
np.random.seed()
### plane set-up
p.changeDynamics(planeID,
        linkIndex = -1,
        lateralFriction = 1,
        spinningFriction = 1,
        rollingFriction = 1,
        #restitution = 0.5,
        #contactStiffness = 0,
        #contactDamping = 0
        )
### model set-up
# body
for i in range(-1,model.num_joints-4):
    p.changeDynamics(model.Id,
        linkIndex = i,
        lateralFriction = 0.2,
        spinningFriction = 0.2,
        rollingFriction = 0.02,
        restitution = 0.1,
        #contactStiffness = 0,
        #contactDamping = 0
        )
# legs
for i in range(model.num_joints-4,model.num_joints):
    p.changeDynamics(model.Id,
        linkIndex = i,
        lateralFriction = 0.1,
        spinningFriction = 0.2,
        rollingFriction = 0.02,
        restitution = 0.1,
        #contactStiffness = 0,
        #contactDamping = 0
        )
# dorsal joints (by setting them to try to keep a null speed with a limited torque they act as passive damped joints)
model.turn_off_crawler()
for i in range(0,model.num_joints,2):
    p.setJointMotorControl2(model.Id, i, 
        controlMode=p.VELOCITY_CONTROL, 
        targetVelocity=0, 
        force=0.03
        )
# Control terms
fmax_leg = 40
pGain_leg = 3
vGain_leg = 0.05
fmax_lateral = model.generate_fmax_array_lateral(100)
K_lateral = 180
k0_lateral = 100
vGain_lateral = 1
### joints' starting position and limit for the stance phase
theta_g0 = pi/5
theta_gf = -pi/5
for i in range(0,model.num_joints):
    p.resetJointState(model.Id, i, 0.0, 0.0)
p.resetJointState(model.Id, model.control_indices[1], theta_g0,0.0)
p.resetJointState(model.Id, model.control_indices[3], -theta_gf,0.0)
### let the model touch the ground plane
for i in range (10):
    p.stepSimulation()
    time.sleep(dt)


##########################################
####### WALKING SIMULATION ###############
##########################################
model.integrator_lateral.reset(np.array([0]*len(model.control_indices[0])))
model.set_low_pass_lateral(fc=100)
t_stance = 1
### Right stance
steps = int(t_stance/dt)
t = 0.0
model.fix_right_foot()
model.set_COM_y_0()
qda3=list()
qdaf3=list()
e = np.array([0])
### DEBUGGING
joints_pos = [0]*len(model.control_indices[0])
joints_speeds = [0]*len(model.control_indices[0])
joints_torques = [0]*len(model.control_indices[0])
###
for tau in range (steps):
    COM_prev = model.COM_position_world()
    ### right leg stance control
    model.control_leg_abduction(
        RL=0,
        theta0=theta_g0,
        thetaf=theta_gf,
        ti=t,
        t_stance=t_stance,
        force=fmax_leg,
        positionGain=pGain_leg,
        velocityGain=vGain_leg
        )
    ### left leg swing control
    model.control_leg_abduction(
        RL=1,
        theta0=-theta_gf,
        thetaf=-theta_g0,
        ti=t,
        t_stance=t_stance,
        force=fmax_leg,
        positionGain=pGain_leg,
        velocityGain=vGain_leg
        )
    ### control spine's lateral joints
    output_lateral = model.controlV_spine_lateral(
        K=K_lateral,
        k0=k0_lateral,
        fmax=fmax_lateral,
        #positionGain=1,
        velocityGain=vGain_lateral,
        filtered=True)
    print("e = ", np.round(output_lateral[2],5))
    # qda3.append(output_lateral[0][3])
    # print("qda = ", output_lateral[0])
    # print("qdaf = ", output_lateral[1])
    # qdaf3.append(output_lateral[1][3])
    e=np.append(e,output_lateral[2])
    ###
    ###
    p.stepSimulation()
    t += dt
    ###
    # ### DEBUGGING
    # for i in range(len(model.control_indices[0])):
    #     joints_pos[i] = round(p.getJointState(model.Id, model.control_indices[0][i])[0],4)
    #     joints_speeds[i] = round(p.getJointState(model.Id, model.control_indices[0][i])[1],4)
    #     joints_torques[i] = round(p.getJointState(model.Id, model.control_indices[0][i])[3],4)
    # print("\n")
    # print("JOINT STATES AFTER SIMULATION STEPS, TORQUES HAS BEEN APPLIED DURING THE STEP")
    # print("position", joints_pos)
    # print("speeds  ", joints_speeds)
    # print("torques ", joints_torques)
    ### show COM trajectory
    COM_curr = model.COM_position_world()
    p.addUserDebugLine(COM_prev.tolist(), COM_curr.tolist(), lineColorRGB=[sin(4*pi*t),sin(4*pi*(t+0.33)),sin(4*pi*(t+0.67))],lineWidth=3, lifeTime=2)
    #
    time.sleep(dt)

def left_step(t,e):
    ti = t
    ### constraints switch
    model.free_right_foot()
    model.fix_left_foot()
    # model.turn_off_crawler()
    # for i in range(0,model.num_joints,2):
    #     p.setJointMotorControl2(model.Id, i, 
    #         controlMode=p.VELOCITY_CONTROL, 
    #         targetVelocity=0, 
    #         force=0.03
    #         )
    p.stepSimulation()
    time.sleep(dt)
    p.stepSimulation()
    time.sleep(dt)
    model.set_COM_y_0()
    for tau in range (steps):
        # p.stepSimulation()
        # t += dt
        ### DEBUGGING
        COM_prev = model.COM_position_world()
        ### right leg swing control
        model.control_leg_abduction(
            RL=0,
            theta0=theta_g0,
            thetaf=theta_gf,
            ti=ti,
            t_stance=t_stance,
            force=fmax_leg,
            positionGain=pGain_leg,
            velocityGain=vGain_leg
            )
        ### left leg stance control
        model.control_leg_abduction(
            RL=1,
            theta0=-theta_gf,
            thetaf=-theta_g0,
            ti=ti,
            t_stance=t_stance,
            force=fmax_leg,
            positionGain=pGain_leg,
            velocityGain=vGain_leg
            )
        ### control spine's lateral joints
        output_lateral = model.controlV_spine_lateral(
            K=K_lateral,
            k0=k0_lateral,
            fmax=fmax_lateral,
            #positionGain=1,
            velocityGain=vGain_lateral,
            filtered=True)
        print("e = ", np.round(output_lateral[2],5))
        e=np.append(e,output_lateral[2])
        ###
        ###
        p.stepSimulation()
        ti += dt
        ###
        ### show COM trajectory
        COM_curr = model.COM_position_world()
        p.addUserDebugLine(COM_prev.tolist(), COM_curr.tolist(), lineColorRGB=[sin(4*pi*t),sin(4*pi*(t+0.33)),sin(4*pi*(t+0.67))],lineWidth=3, lifeTime=2)
        #
        time.sleep(dt)
    return ti, e

def right_step(t,e):
    ti = t
    ### constraints switch
    model.free_left_foot()
    model.fix_right_foot()
    # model.turn_off_crawler()
    # for i in range(0,model.num_joints,2):
    #     p.setJointMotorControl2(model.Id, i, 
    #         controlMode=p.VELOCITY_CONTROL, 
    #         targetVelocity=0, 
    #         force=0.03
    #         )
    p.stepSimulation()
    time.sleep(dt)
    p.stepSimulation()
    time.sleep(dt)
    model.set_COM_y_0()
    for tau in range (steps):
        # p.stepSimulation()
        # t += dt
        ### DEBUGGING
        COM_prev = model.COM_position_world()
        ### right leg swing control
        model.control_leg_abduction(
            RL=0,
            theta0=theta_g0,
            thetaf=theta_gf,
            ti=ti,
            t_stance=t_stance,
            force=fmax_leg,
            positionGain=pGain_leg,
            velocityGain=vGain_leg
            )
        ### left leg stance control
        model.control_leg_abduction(
            RL=1,
            theta0=-theta_gf,
            thetaf=-theta_g0,
            ti=ti,
            t_stance=t_stance,
            force=fmax_leg,
            positionGain=pGain_leg,
            velocityGain=vGain_leg
            )
        ### control spine's lateral joints
        output_lateral = model.controlV_spine_lateral(
            K=K_lateral,
            k0=k0_lateral,
            fmax=fmax_lateral,
            #positionGain=1,
            velocityGain=vGain_lateral,
            filtered=True)
        print("e = ", np.round(output_lateral[2],5))
        e=np.append(e,output_lateral[2])
        ###
        ###
        p.stepSimulation()
        ti += dt
        ###
        ### show COM trajectory
        COM_curr = model.COM_position_world()
        p.addUserDebugLine(COM_prev.tolist(), COM_curr.tolist(), lineColorRGB=[sin(4*pi*t),sin(4*pi*(t+0.33)),sin(4*pi*(t+0.67))],lineWidth=3, lifeTime=2)
        #
        time.sleep(dt)
    return ti, e

t, e = left_step(t,e)
t, e = right_step(t,e)
t, e = left_step(t,e)


# print(model.solve_null_COM_y_speed())

# print("link %d" %6)
# model.test_link_COM_jacobians(6)
# print("\n")

# for i in range(0,20):for i in range (model.num_joints):
#    print(p.getJointInfo(model.Id, i))

#     print("link %d" %i)
#     print("COM position base world: ", np.asarray(model.links_state_array[i]["world_com_trn"]
#         -np.asarray(p.getBasePositionAndOrientation(model.Id)[0])))
#     model.test_link_COM_jacobians(i)
#     print("\n")


fig, ax = plt.subplots()
plt.plot(e,"b")
#plt.plot(qdaf3,"r")
plt.show()

p.disconnect()
