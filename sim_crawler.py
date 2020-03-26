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
dt = 1./300.
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

##########################################
####### MODEL SET-UP #####################
##########################################
# Remove motion damping ("air friction")
for index in range(-1,model.num_joints):
    p.changeDynamics(
        model.Id,
        index,
        linearDamping=0.0,
        angularDamping=0.0
    )
# Body dynamic properties
for i in range(-1,model.num_joints-4):
    p.changeDynamics(model.Id,
        linkIndex = i,
        lateralFriction = 0.12,
        spinningFriction = 0.12,
        rollingFriction = 0.012,
        restitution = 0.1,
        #contactStiffness = 0,
        #contactDamping = 0
        )
# Legs dynamic properties
for i in range(model.num_joints-4,model.num_joints):
    p.changeDynamics(model.Id,
        linkIndex = i,
        lateralFriction = 0.12,
        spinningFriction = 0.12,
        rollingFriction = 0.012,
        restitution = 0.1,
        #contactStiffness = 0,
        #contactDamping = 0
        )
# Joint properties
# Setting the dorsal joints to try to keep a null speed with a limited torque
    # they act similarly to damped passive joints)
model.turn_off_crawler()
for i in range(0,model.num_joints,2):
    p.setJointMotorControl2(model.Id, i, 
        controlMode=p.VELOCITY_CONTROL, 
        targetVelocity=0, 
        force=0.03
        )
##########################################
####### WALKING SIMULATION ###############
##########################################
K_lateral = 100
k0_lateral = 200
###
Kp = model.generate_Kp(Kp_lat=4000, Kp_abd=20000, Kp_flex=300)
Kv = model.generate_Kv(Kv_lat=800, Kv_abd=6000, Kv_flex=120)
###
#model.set_low_pass_qd(fc=10)
model.set_low_pass_lateral_qa(fc=10)
model.set_low_pass_tau(fc=50)
model.integrator_lateral_qa.reset(np.array([0]*len(model.control_indices[0])))

##########################################
####### WALKING SIMULATION ###############
##########################################
t_stance = 1
### Initialization
steps = int(t_stance/dt)
t = 0.0
#
eCOM_history = np.array([0])
eq_history = np.array([0]*model.state.nqd)
des_history=list()
tau_history=list()
# Starting position and legs abduction range
theta_g0 = pi/8
theta_gf = -pi/8
p.resetJointState(model.Id, model.control_indices[1][0],theta_g0)
p.resetJointState(model.Id, model.control_indices[2][0],-theta_gf)
# Let the model touch the ground plane
for i in range (50):
    p.stepSimulation()
    time.sleep(dt)

def single_step_torque_control(t,eq_history, eCOM_history, des_history, tau_history, steps):
    ### Stepping
    model.set_COM_y_ref()
    qda_des_history=np.array(([0]*len(model.control_indices[0])))
    for tau in range (steps):
        #print("\n\n\n\n##### STEP %d #####" %tau)
        model.state.update()
        COM_prev = model.COM_position_world()
        qa_des, qda_des, qdda_des = model.solve_null_COM_y_speed_optimization_qdda(
            qda_prev = qda_des_history[-1],
            K = K_lateral,
            filtered = True)
        qda_des_history = np.vstack((qda_des_history, qda_des))
        q_des, qd_des, qdd_des = model.generate_joints_trajectory(
            theta0=theta_g0,
            thetaf=theta_gf,
            ti=t,
            t_stance=t_stance,
            qa_des=qa_des,
            qda_des=qda_des,
            qdda_des=qdda_des
        )
        des_history.append([q_des,qd_des,qdd_des])
        tau, eq = model.solve_computed_torque_control(
            q_des=q_des,
            qd_des=qd_des,
            qdd_des=qdd_des,
            Kp=Kp,
            Kv=Kv,
            verbose=False
        )
        model.apply_torques(tau_des=tau, filtered=True)
        tau_history.append(tau)
        eq_history=np.vstack((eq_history,eq))
        eCOM_history = np.vstack((eCOM_history,model.COM_position_world()[1]))
        ###
        p.stepSimulation()
        t += dt
        ###
        ### show COM trajectory
        COM_curr = model.COM_position_world()
        p.addUserDebugLine(COM_prev.tolist(), COM_curr.tolist(), lineColorRGB=[sin(4*pi*t),sin(4*pi*(t+0.33)),sin(4*pi*(t+0.67))],lineWidth=3, lifeTime=2)
        #
        time.sleep(dt)
    return t, eq_history, eCOM_history, des_history, tau_history

# 

###################
t=0.0
##############
### RIGHT STEP
model.fix_right_foot()
model.free_left_foot()
# ### Let the model adapt to the new constraints
# for i in range(5):
#     p.stepSimulation()
#     time.sleep(dt)
t,eq_history, eCOM_history, des_history, tau_history = single_step_torque_control(t,eq_history, eCOM_history,des_history,tau_history, steps=steps)
print("\n\n\n")
model.fix_left_foot()
model.free_right_foot()
#############
### LEFT STEP
model.fix_left_foot()
model.free_right_foot()

t,eq_history, eCOM_history, des_history, tau_history = single_step_torque_control(
    t,
    eq_history,
    eCOM_history,
    des_history,tau_history, 
    steps = steps
    )
##########################################
####### PRINT RESULTS ####################
##########################################

tau_history = np.array(tau_history)
#print("Shape des_history: ", des_history.shape)
print("Shape tau_history: ", tau_history.shape)
print("Shape eq: ", eq_history.shape)

qa_des_history = (np.array(des_history[0][0]))[model.mask_act_shifted]
qda_des_history = np.array(des_history[0][1])[model.mask_act_shifted]
qdda_des_history = np.array(des_history[0][2])[model.mask_act_shifted]
qrg_des_history = (np.array(des_history[0][0][23]))
qdrg_des_history = (np.array(des_history[0][1][22]))
qddrg_des_history = (np.array(des_history[0][2][22]))
qlg_des_history = (np.array(des_history[0][0][25]))
qdlg_des_history = (np.array(des_history[0][1][24]))
qddlg_des_history = (np.array(des_history[0][2][24]))
for i in range(1,len(des_history)):
    qa_des_history=np.vstack((qa_des_history,np.array(des_history[i][0])[model.mask_act_shifted]))
    qda_des_history=np.vstack((qda_des_history,np.array(des_history[i][1])[model.mask_act_shifted]))
    qdda_des_history=np.vstack((qdda_des_history,np.array(des_history[i][2])[model.mask_act_shifted]))
    qrg_des_history = np.vstack((qrg_des_history, np.array(des_history[i][0][23])))
    qdrg_des_history = np.vstack((qdrg_des_history, np.array(des_history[i][1][22])))
    qddrg_des_history = np.vstack((qddrg_des_history, np.array(des_history[i][2][22])))
    qlg_des_history = np.vstack((qlg_des_history, np.array(des_history[i][0][25])))
    qdlg_des_history = np.vstack((qdlg_des_history, np.array(des_history[i][1][24])))
    qddlg_des_history = np.vstack((qddlg_des_history, np.array(des_history[i][2][24])))
###
eqa_history = eq_history[0][model.mask_act]
eqrg_history = eq_history[0][22]
eqlg_history = eq_history[0][24]
for i in range(1,eq_history.shape[0]):
    eqa_history = np.vstack((eqa_history,eq_history[i][model.mask_act]))
    eqrg_history = np.vstack((eqrg_history,eq_history[i][22]))
    eqlg_history = np.vstack((eqlg_history,eq_history[i][24]))
###
tau_lateral_history=np.array(tau_history[0])[model.mask_act]
tau_rg_history=np.array(tau_history[0][22])
tau_lg_history=np.array(tau_history[0][24])
for i in range(1,tau_history.shape[0]):
    tau_lateral_history=np.vstack((tau_lateral_history,np.array(tau_history[i])[model.mask_act]))
    tau_rg_history=np.vstack((tau_rg_history, tau_history[i][22]))
    tau_lg_history=np.vstack((tau_lg_history, tau_history[i][24]))
###
fig, axs = plt.subplots(3,3)
#plt.plot(e,"b")
# for i in range(eq_history.shape[1]):
#     axs[0,0].plot(eq_history[:,i])
# axs[0,0].set_title("eq\nLateral joints positional error")
for i in range(1,1+qda_des_history.shape[1]):
    axs[i//3,i%3].plot(qa_des_history[:,i-1], color="xkcd:dark teal")
    axs[i//3,i%3].plot(qda_des_history[:,i-1], color="xkcd:teal")
    axs[i//3,i%3].plot(qdda_des_history[:,i-1], color="xkcd:light teal")
    #axs[i//3,i%3].plot(tau_lateral_history[:,i-1], color="xkcd:salmon")
    axs[i//3,i%3].plot(eqa_history[:,i-1], color="xkcd:orange")
    axs[i//3,i%3].set_title("Lateral joint %d" %i)
fig_g, axs_g = plt.subplots(1,2)
axs_g[1].plot(qlg_des_history, color="xkcd:dark teal")
axs_g[1].plot(qdlg_des_history, color="xkcd:teal")
axs_g[1].plot(qddlg_des_history, color="xkcd:light teal")
#axs_g[1].plot(tau_lg_history, color="xkcd:salmon")
axs_g[1].plot(eqlg_history, color="xkcd:orange")
axs_g[1].set_title("Left abduction")
axs_g[0].plot(qrg_des_history, color="xkcd:dark teal")
axs_g[0].plot(qdrg_des_history, color="xkcd:teal")
axs_g[0].plot(qddrg_des_history, color="xkcd:light teal")
#axs_g[0].plot(tau_rg_history, color="xkcd:salmon")
axs_g[0].plot(eqrg_history, color="xkcd:orange")
axs_g[0].set_title("Right abduction")
# fig_tau, axs_tau = plt.subplots(3,3)
# for i in range(1,1+qda_des_history.shape[1]):
#     axs_tau[i//3,i%3].plot(tau_lateral_history[:,i-1],color="tab:orange")
#     axs_tau[i//3,i%3].plot(tau_lateral_f_history[:,i-1],color="tab:brown")
#     #axs[i//3,i%3].plot(eq_history[:,i-1], color="tab:pink")
#     axs_tau[i//3,i%3].set_title("Tau %d" %i)
plt.show()

p.disconnect()


#def right_step_torque_control(t,eq_history, eCOM_history, des_history, tau_history):
#     ti = 0.0
#     ### constraints switch
#     model.free_left_foot()
#     model.fix_right_foot()
#     ### "turn off" swing leg flexion
#     model.turn_off_joint(model.control_indices[2][1])
#     ### Let the model adapt to the new constraints
#     for i in range(5):
#         p.stepSimulation()
#         time.sleep(dt)
#     ### Stepping
#     model.set_COM_y_ref()
#     qda_des_history=np.array(([0]*len(model.control_indices[0])))
#     for tau in range (steps):
#         #print("\n\n\n\n##### STEP %d #####" %tau)
#         model.state.update()
#         COM_prev = model.COM_position_world()
#         qa_des, qda_des, qdda_des = model.solve_null_COM_y_speed_optimization_qdda(
#             qda_prev = qda_des_history[-1],
#             K = K_lateral,
#             filtered = True)
#         qda_des_history = np.vstack((qda_des_history, qda_des))
#         q_des, qd_des, qdd_des = model.generate_joints_trajectory(
#             theta0=theta_g0,
#             thetaf=theta_gf,
#             ti=ti,
#             t_stance=t_stance,
#             RL=0,
#             qa_des=qa_des,
#             qda_des=qda_des,
#             qdda_des=qdda_des
#         )
#         des_history.append([q_des,qd_des,qdd_des])
#         tau, eq = model.solve_computed_torque_control(
#             q_des=q_des,
#             qd_des=qd_des,
#             qdd_des=qdd_des,
#             Kp=Kp,
#             Kv=Kv,
#             verbose=False
#         )
#         model.apply_torques(tau_des=tau, filtered=True)
#         tau_history.append(tau)
#         eq_history=np.vstack((eq_history,eq))
#         eCOM_history = np.vstack((eCOM_history,model.COM_position_world()[1]))
#         ###
#         p.stepSimulation()
#         ti += dt
#         t += dt
#         ###
#         ### show COM trajectory
#         COM_curr = model.COM_position_world()
#         p.addUserDebugLine(COM_prev.tolist(), COM_curr.tolist(), lineColorRGB=[sin(4*pi*t),sin(4*pi*(t+0.33)),sin(4*pi*(t+0.67))],lineWidth=3, lifeTime=2)
#         #
#         time.sleep(dt)
#     return ti, eq_history, eCOM_history, des_history, tau_history


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

### FUNCTION FOR SINGLE STEP
# def left_step(t,e):
#     ti = t
#     ### constraints switch
#     model.free_right_foot()
#     model.fix_left_foot()
#     ### "turn off" swing leg flexion
#     model.turn_off_joint(model.control_indices[1][1])
#     p.stepSimulation()
#     time.sleep(dt)
#     p.stepSimulation()
#     time.sleep(dt)
#     model.set_COM_y_0()
#     for tau in range (steps):
#         # p.stepSimulation()
#         # t += dt
#         ### DEBUGGING
#         COM_prev = model.COM_position_world()
#         ### right leg swing control
#         model.control_leg_abduction(
#             RL=0,
#             theta0=theta_g0,
#             thetaf=theta_gf,
#             ti=ti,
#             t_stance=t_stance,
#             fmax=fmax_leg_abd,
#             positionGain=pGain_leg,
#             velocityGain=vGain_leg
#             )
#         ### left leg stance control
#         model.control_leg_abduction(
#             RL=1,
#             theta0=-theta_gf,
#             thetaf=-theta_g0,
#             ti=ti,
#             t_stance=t_stance,
#             fmax=fmax_leg_abd,
#             positionGain=pGain_leg,
#             velocityGain=vGain_leg
#             )
#         ### left leg flexion control, to fixed "neutral contact" angle
#         model.control_leg_flexion(
#             RL=1,
#             fmax=fmax_leg_flex,
#             positionGain=pGain_leg,
#             velocityGain=vGain_leg,)
#         ### control spine's lateral joints
#         output_lateral = model.controlV_spine_lateral(
#             K=K_lateral,
#             k0=k0_lateral,
#             fmax=fmax_lateral,
#             #positionGain=1,
#             velocityGain=vGain_lateral,
#             filtered=True)
#         print("e = ", np.round(output_lateral[2],5))
#         e=np.append(e,output_lateral[2])
#         ###
#         ###
#         p.stepSimulation()
#         ti += dt
#         ###
#         ### show COM trajectory
#         COM_curr = model.COM_position_world()
#         p.addUserDebugLine(COM_prev.tolist(), COM_curr.tolist(), lineColorRGB=[sin(4*pi*t),sin(4*pi*(t+0.33)),sin(4*pi*(t+0.67))],lineWidth=3, lifeTime=2)
#         #
#         time.sleep(dt)
#     return ti, e

# def right_step(t,e):
#     ti = t
#     ### constraints switch
#     model.free_left_foot()
#     model.fix_right_foot()
#     ### "turn off" swing leg flexion
#     model.turn_off_joint(model.control_indices[2][1])
#     p.stepSimulation()
#     time.sleep(dt)
#     p.stepSimulation()
#     time.sleep(dt)
#     model.set_COM_y_0()
#     for tau in range (steps):
#         # p.stepSimulation()
#         # t += dt
#         ### DEBUGGING
#         COM_prev = model.COM_position_world()
#         ### right leg swing control
#         model.control_leg_abduction(
#             RL=0,
#             theta0=theta_g0,
#             thetaf=theta_gf,
#             ti=ti,
#             t_stance=t_stance,
#             fmax=fmax_leg_abd,
#             positionGain=pGain_leg,
#             velocityGain=vGain_leg
#             )
#         ### right leg flexion control, to fixed "neutral contact" angle
#         model.control_leg_flexion(
#             RL=0,
#             fmax=fmax_leg_flex,
#             positionGain=pGain_leg,
#             velocityGain=vGain_leg,)
#         ### left leg stance control
#         model.control_leg_abduction(
#             RL=1,
#             theta0=-theta_gf,
#             thetaf=-theta_g0,
#             ti=ti,
#             t_stance=t_stance,
#             fmax=fmax_leg_abd,
#             positionGain=pGain_leg,
#             velocityGain=vGain_leg
#             )
#         ### control spine's lateral joints
#         output_lateral = model.controlV_spine_lateral(
#             K=K_lateral,
#             k0=k0_lateral,
#             fmax=fmax_lateral,
#             #positionGain=1,
#             velocityGain=vGain_lateral,
#             filtered=True)
#         print("e = ", np.round(output_lateral[2],5))
#         e=np.append(e,output_lateral[2])
#         ###
#         ###
#         p.stepSimulation()
#         ti += dt
#         ###
#         ### show COM trajectory
#         COM_curr = model.COM_position_world()
#         p.addUserDebugLine(COM_prev.tolist(), COM_curr.tolist(), lineColorRGB=[sin(4*pi*t),sin(4*pi*(t+0.33)),sin(4*pi*(t+0.67))],lineWidth=3, lifeTime=2)
#         #
#         time.sleep(dt)
#     return ti, e

# t, e = right_step(t,e)
# t, e = left_step(t,e)
# t, e = right_step(t,e)
# t, e = left_step(t,e)
