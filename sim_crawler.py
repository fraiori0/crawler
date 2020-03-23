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
#p.setGravity(0,0,0)
planeID = p.loadURDF("plane100.urdf", [0,0,0])
model = crw.Crawler(urdf_path="/home/fra/Uni/Tesi/crawler", dt_simulation=dt, base_position=[0,0,0.05], base_orientation=[0,0,0,1])
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
# remove damping of motion ("air friction")
for index in range(-1,model.num_joints):
    p.changeDynamics(
        model.Id,
        index,
        linearDamping=0.0,
        angularDamping=0.0
    )
# body
for i in range(-1,model.num_joints-4):
    p.changeDynamics(model.Id,
        linkIndex = i,
        lateralFriction = 0.002,
        spinningFriction = 0.002,
        rollingFriction = 0.002,
        restitution = 0.1,
        #contactStiffness = 0,
        #contactDamping = 0
        )
# legs
for i in range(model.num_joints-4,model.num_joints):
    p.changeDynamics(model.Id,
        linkIndex = i,
        lateralFriction = 0.001,
        spinningFriction = 0.002,
        rollingFriction = 0.002,
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
### joints' starting position and limit for the stance phase
theta_g0 = pi/6
theta_gf = -pi/6
for i in range(0,model.num_joints):
    p.resetJointState(model.Id, i, 0.0, 0.0)
p.resetJointState(model.Id, model.control_indices[1], theta_g0,0.0)
p.resetJointState(model.Id, model.control_indices[3], -theta_gf,0.0)
### let the model touch the ground plane
model.turn_off_crawler()
for i in range (120):
    p.stepSimulation()
    time.sleep(dt)
# dorsal joints (by setting them to try to keep a null speed with a limited torque they act as passive damped joints)
for i in range(0,model.num_joints,2):
    p.setJointMotorControl2(model.Id, i, 
        controlMode=p.VELOCITY_CONTROL, 
        targetVelocity=0, 
        force=0.03
        )

# Control and filtering terms
fmax_leg_abd = 40
fmax_leg_flex = 40
pGain_leg = 1
vGain_leg = 0.5
fmax_lateral = model.generate_fmax_array_lateral(100)
K_lateral = 30 # is K_lateral useless with torque control?
k0_lateral = 30
vGain_lateral = 0.8
model.set_low_pass_lateral(fc=1000)
model.set_low_pass_qd(fc=100)
model.set_low_pass_tau_lateral(fc=100)
Kp_lateral = np.diag(([1000]*len(model.control_indices[0])))
Kv_lateral = np.diag(([100]*len(model.control_indices[0])))

##########################################
####### WALKING SIMULATION ###############
##########################################
model.integrator_lateral.reset(np.array([0]*len(model.control_indices[0])))
t_stance = 1
### Initialization
steps = int(t_stance/dt)
t = 0.0
model.fix_right_foot()
model.set_COM_y_0()
#
eCOM_history = np.array([0])
eq_history = np.array([0]*len(model.mask_act))
des_history=list()
tau_history=list()
# ### DEBUGGING
# joints_pos = [0]*len(model.control_indices[0])
# joints_speeds = [0]*len(model.control_indices[0])
# joints_torques = [0]*len(model.control_indices[0])
def right_step_torque_control(t,eq_history, eCOM_history, des_history, tau_history):
    ti = t
    ### constraints switch
    model.free_left_foot()
    model.fix_right_foot()
    ### "turn off" swing leg flexion
    model.turn_off_joint(model.control_indices[4])
    model.turn_off_joint(model.control_indices[2])
    for i in range(2):
        p.stepSimulation()
        time.sleep(dt)
    model.set_COM_y_0()
    qda_des_prev=np.array(([0]*len(model.control_indices[0])))
    qda_des_history=qda_des_prev.copy()
    for tau in range (steps):
        print("\n\n\n\n##### STEP %d #####" %tau)
        # p.stepSimulation()
        # t += dt
        ### operations before stepping the simulation
        COM_prev = model.COM_position_world()
        qd_prev = model.get_joints_speeds_tuple()
        ###
        ###
        p.stepSimulation()
        model.set_velocities()
        ti += dt
        ###
        ### right leg stance control
        model.control_leg_abduction(
            RL=0,
            theta0=theta_g0,
            thetaf=theta_gf,
            ti=ti,
            t_stance=t_stance,
            fmax=fmax_leg_abd,
            positionGain=pGain_leg,
            velocityGain=vGain_leg
            )
        ### right leg flexion control, to fixed "neutral contact" angle
        # model.control_leg_flexion(
        #     RL=1,
        #     fmax=fmax_leg_flex,
        #     positionGain=pGain_leg,
        #     velocityGain=vGain_leg,)
        ### left leg swing control
        model.control_leg_abduction(
            RL=1,
            theta0=-theta_gf,
            thetaf=-theta_g0,
            ti=ti,
            t_stance=t_stance,
            fmax=fmax_leg_abd,
            positionGain=pGain_leg,
            velocityGain=vGain_leg
            )
        ### control spine's lateral joints
        output_lateral = model.controlT_spine_lateral(
            qda_des_prev=qda_des_prev,
            K= K_lateral,
            Kp = Kp_lateral,
            Kv = Kv_lateral,
            filtered_des=True,
            filtered_real=False,
            filtered_tau=True
        )
        qda_des_prev = output_lateral[0][1]
        des_history.append(output_lateral[0])
        eq_history = np.vstack((eq_history, output_lateral[1]))
        eCOM_history = np.append(eCOM_history,output_lateral[2])
        tau_history.append(output_lateral[3])
        ### show COM trajectory
        COM_curr = model.COM_position_world()
        p.addUserDebugLine(COM_prev.tolist(), COM_curr.tolist(), lineColorRGB=[sin(4*pi*t),sin(4*pi*(t+0.33)),sin(4*pi*(t+0.67))],lineWidth=3, lifeTime=2)
        #
        time.sleep(dt)
    return ti, eq_history, eCOM_history, des_history, tau_history

### FUNCTION FOR SINGLE STEP
def left_step(t,e):
    ti = t
    ### constraints switch
    model.free_right_foot()
    model.fix_left_foot()
    ### "turn off" swing leg flexion
    model.turn_off_joint(model.control_indices[2])
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
            fmax=fmax_leg_abd,
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
            fmax=fmax_leg_abd,
            positionGain=pGain_leg,
            velocityGain=vGain_leg
            )
        ### left leg flexion control, to fixed "neutral contact" angle
        model.control_leg_flexion(
            RL=1,
            fmax=fmax_leg_flex,
            positionGain=pGain_leg,
            velocityGain=vGain_leg,)
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
    ### "turn off" swing leg flexion
    model.turn_off_joint(model.control_indices[4])
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
            fmax=fmax_leg_abd,
            positionGain=pGain_leg,
            velocityGain=vGain_leg
            )
        ### right leg flexion control, to fixed "neutral contact" angle
        model.control_leg_flexion(
            RL=0,
            fmax=fmax_leg_flex,
            positionGain=pGain_leg,
            velocityGain=vGain_leg,)
        ### left leg stance control
        model.control_leg_abduction(
            RL=1,
            theta0=-theta_gf,
            thetaf=-theta_g0,
            ti=ti,
            t_stance=t_stance,
            fmax=fmax_leg_abd,
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

# t, e = right_step(t,e)
# t, e = left_step(t,e)
# t, e = right_step(t,e)
# t, e = left_step(t,e)
t,eq_history, eCOM_history, des_history, tau_history = right_step_torque_control(t,eq_history, eCOM_history,des_history,tau_history)
#
qa_des_history=np.array(des_history[0][0])
qda_des_history=np.array(des_history[0][1])
qdda_des_history=np.array(des_history[0][2])
for i in range(1,len(des_history)):
    qa_des_history=np.vstack((qa_des_history,np.array(des_history[i][0])))
    qda_des_history=np.vstack((qda_des_history,np.array(des_history[i][1])))
    qdda_des_history=np.vstack((qdda_des_history,np.array(des_history[i][2])))
#
tau_lateral_history=np.array(tau_history[0][0])
tau_lateral_f_history=np.array(tau_history[0][1])
for i in range(len(des_history)):
    tau_lateral_history=np.vstack((tau_lateral_history,np.array(tau_history[i][0])))
    tau_lateral_f_history=np.vstack((tau_lateral_f_history,np.array(tau_history[i][1])))
#
fig, axs = plt.subplots(3,3)
#plt.plot(e,"b")
# for i in range(eq_history.shape[1]):
#     axs[0,0].plot(eq_history[:,i])
# axs[0,0].set_title("eq\nLateral joints positional error")
axs[0,0].plot(eCOM_history)
axs[0,0].set_title("COM y positional error")
for i in range(1,1+qda_des_history.shape[1]):
    axs[i//3,i%3].plot(qa_des_history[:,i-1],"r")
    axs[i//3,i%3].plot(qda_des_history[:,i-1],"g")
    axs[i//3,i%3].plot(qdda_des_history[:,i-1],"b")
    axs[i//3,i%3].plot(eq_history[:,i-1], color="tab:pink")
    axs[i//3,i%3].set_title("Joint %d" %i)
#plt.show()
# fig_tau, axs_tau = plt.subplots(3,3)
# for i in range(1,1+qda_des_history.shape[1]):
#     axs_tau[i//3,i%3].plot(tau_lateral_history[:,i-1],color="tab:orange")
#     axs_tau[i//3,i%3].plot(tau_lateral_f_history[:,i-1],color="tab:brown")
#     #axs[i//3,i%3].plot(eq_history[:,i-1], color="tab:pink")
#     axs_tau[i//3,i%3].set_title("Tau %d" %i)

plt.show()
p.disconnect()


# for i in range (1200):
#     p.stepSimulation()
#     time.sleep(dt)
###
# for tau in range (steps):
#     COM_prev = model.COM_position_world()
#     ### right leg stance control
#     model.control_leg_abduction(
#         RL=0,
#         theta0=theta_g0,
#         thetaf=theta_gf,
#         ti=t,
#         t_stance=t_stance,
#         fmax=fmax_leg_abd,
#         positionGain=pGain_leg,
#         velocityGain=vGain_leg
#         )
#     ### right leg flexion control, to fixed "neutral contact" angle
#     model.control_leg_flexion(
#         RL=0,
#         fmax=fmax_leg_flex,
#         positionGain=pGain_leg,
#         velocityGain=vGain_leg,)
#     ### left leg swing control
#     model.control_leg_abduction(
#         RL=1,
#         theta0=-theta_gf,
#         thetaf=-theta_g0,
#         ti=t,
#         t_stance=t_stance,
#         fmax=fmax_leg_abd,
#         positionGain=pGain_leg,
#         velocityGain=vGain_leg
#         )
#     ### control spine's lateral joints
#     output_lateral = model.controlV_spine_lateral(
#         K=K_lateral,
#         k0=k0_lateral,
#         fmax=fmax_lateral,
#         #positionGain=1,
#         velocityGain=vGain_lateral,
#         filtered=True)
#     print("e = ", np.round(output_lateral[2],5))
#     # qda3.append(output_lateral[0][3])
#     # print("qda = ", output_lateral[0])
#     # print("qdaf = ", output_lateral[1])
#     # qdaf3.append(output_lateral[1][3])
#     e=np.append(e,output_lateral[2])
#     ###
#     ###
#     p.stepSimulation()
#     t += dt
#     ###
#     # ### DEBUGGING
#     # for i in range(len(model.control_indices[0])):
#     #     joints_pos[i] = round(p.getJointState(model.Id, model.control_indices[0][i])[0],4)
#     #     joints_speeds[i] = round(p.getJointState(model.Id, model.control_indices[0][i])[1],4)
#     #     joints_torques[i] = round(p.getJointState(model.Id, model.control_indices[0][i])[3],4)
#     # print("\n")
#     # print("JOINT STATES AFTER SIMULATION STEPS, TORQUES HAS BEEN APPLIED DURING THE STEP")
#     # print("position", joints_pos)
#     # print("speeds  ", joints_speeds)
#     # print("torques ", joints_torques)
#     ### show COM trajectory
#     COM_curr = model.COM_position_world()
#     p.addUserDebugLine(COM_prev.tolist(), COM_curr.tolist(), lineColorRGB=[sin(4*pi*t),sin(4*pi*(t+0.33)),sin(4*pi*(t+0.67))],lineWidth=3, lifeTime=2)
#     #
#     time.sleep(dt)

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
