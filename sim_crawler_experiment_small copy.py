import pybullet as p
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import time
import pybullet_data
import crawler_small_mass as crw
from math import *
import subprocess as sp
import imageio

matplotlib.use('TkAgg')

#########################################
####### SIMULATION SET-UP ###############
#########################################
### time-step used for simulation and total time passed variable
dt = 1./2400.
#
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0,0,-9.81)
p.setTimeStep(dt)
#p.setGravity(0,0,0)
planeID = p.loadURDF("plane100.urdf", [0,0,0])
z_rotation = pi/6 #radians
model = crw.Crawler(
    urdf_path="/home/fra/Uni/Tesi/crawler",
    dt_simulation=dt,
    base_position=[0,0,0.005],
    base_orientation=[0,0,sin(z_rotation/2),cos(z_rotation/2)],
    mass_distribution=True
    )
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
        linearDamping=0.01,
        angularDamping=0.01
    )
# Body dynamic properties
for i in range(-1,model.num_joints-4):
    p.changeDynamics(model.Id,
        linkIndex = i,
        lateralFriction = 0.001,
        spinningFriction = 0.0001,
        rollingFriction = 0.0001,
        restitution = 0.1,
        #contactStiffness = 0,
        #contactDamping = 0
        )
# # Last link dynamic properties
p.changeDynamics(model.Id,
    linkIndex = (model.control_indices[0][-1]+1),
    lateralFriction = 0.00001,
    spinningFriction = 0.00001,
    rollingFriction = 0.0,
    restitution = 0.1,
    #contactStiffness = 0,
    #contactDamping = 0
    )
# Right Leg dynamic properties
for i in range(model.num_joints-4,model.num_joints-2):
    p.changeDynamics(model.Id,
        linkIndex = i,
        lateralFriction = 0.00001,
        spinningFriction = 0.00001,
        rollingFriction = 0.00001,
        restitution = 0.1,
        #contactStiffness = 0,
        #contactDamping = 0
        )
# Left Leg dynamic properties
for i in range(model.num_joints-2,model.num_joints):
    p.changeDynamics(model.Id,
        linkIndex = i,
        lateralFriction = 0.00001,
        spinningFriction = 0.00001,
        rollingFriction = 0.00001,
        restitution = 0.1,
        #contactStiffness = 0,
        #contactDamping = 0
        )
# Joint properties
# Setting the dorsal joints to try to keep a null speed with a limited torque
    # they act similarly to damped passive joints)
#model.turn_off_crawler()
# for i in range(0,model.num_joints,2):
#     p.setJointMotorControl2(model.Id, i, 
#         controlMode=p.VELOCITY_CONTROL, 
#         targetVelocity=0, 
#         force=0.03
#         )
##########################################
####### CONTROL PARAMETERS ###############
##########################################
K_lateral = 0. #1000.
k0_lateral = 0.
###
# Kp = model.generate_Kp(Kp_lat=0., Kp_r_abd=0., Kp_l_abd=0., Kp_flex=0.)
# Kv = model.generate_Kv(Kv_lat=0., Kv_r_abd=0., Kv_l_abd=0., Kv_flex=0.)
# Kp[6+model.num_joints-4,6+model.num_joints-4]=10.
# Kv[6+model.num_joints-4,6+model.num_joints-4]=1.
# Kp = model.generate_Kp(Kp_lat=20000., Kp_r_abd=20000., Kp_l_abd=20000., Kp_flex=100.)
# Kv = model.generate_Kv(Kv_lat=500, Kv_r_abd=500, Kv_l_abd=500, Kv_flex=50)
Kp = model.generate_Kp(Kp_lat=1e7, Kp_r_abd=1e8, Kp_l_abd=1e8, Kp_flex=0.)
Kv = model.generate_Kv(Kv_lat=5e6, Kv_r_abd=5e7, Kv_l_abd=5e7, Kv_flex=0.)
# Gain for more joints
###
#model.set_low_pass_qd(fc=10)
model.set_low_pass_lateral_qa(fc=60)
model.set_low_pass_tau(fc=90)
### Sliding mode --> rho = upper bound on the norm of the disturbance
rho = 0.5
### PD & P controllers parameters
fmax_list = model.generate_fmax_list(fmax_lat=0.01, fmax_r_abd=0.01, fmax_l_abd=0.01, fmax_flex=0.01)
pGain_list = model.generate_pGain_list(pGain_lat=0.0,  pGain_r_abd=0.0,   pGain_l_abd=0.0,   pGain_flex=0.0)
vGain_list = model.generate_vGain_list(vGain_lat=0.0, vGain_r_abd=0.0,   vGain_l_abd=0.0,   vGain_flex=0.0)
pGain_list_vel = model.generate_pGain_list(pGain_lat=1,  pGain_r_abd=1,   pGain_l_abd=1,   pGain_flex=1)
vGain_lat_mixed = 0.09
##########################################
####### SINGLE-STEP FUNCTION #############
##########################################

def single_step_torque_control(t,eq_history, eCOM_history, des_history, tau_history, state_history, steps, log_data=True, log_video=False, stop_motion=False):
    model.state.update()
    model.set_COM_y_ref()
    qda_des_prev=np.array(([0]*len(model.control_indices[0])))
    p_COM_time_array = np.zeros((steps,3))
    for i in range (steps):
        #print("\n\n\n\n##### STEP %d #####" %i)
        model.state.update(filtered_acc=True)
        COM_prev = model.COM_position_world()
        p_COM_time_array[i]=model.COM_position_world()
        qa_des, qda_des, qdda_des = model.solve_null_COM_y_speed_optimization_qdda(
            qda_prev = qda_des_prev,
            K = K_lateral,
            filtered = True)
        qda_des_prev=qda_des
        if stop_motion:
            q_des, qd_des, qdd_des, eq = model.generate_joints_trajectory_stop(
                theta0=theta_g0,
                thetaf=theta_gf,
                ti=t,
                t_stance=t_stance,
                qa_des=qa_des,
                qda_des=qda_des,
                qdda_des=qdda_des
            )
        else:
            q_des, qd_des, qdd_des, eq = model.generate_joints_trajectory(
                theta0=theta_g0,
                thetaf=theta_gf,
                ti=t,
                t_stance=t_stance,
                qa_des=qa_des,
                qda_des=qda_des,
                qdda_des=qdda_des,
                cos_abd=True
            )
        tau_des, eq = model.solve_computed_torque_control(
            q_des=q_des,
            qd_des=qd_des,
            qdd_des=qdd_des,
            Kp=Kp,
            Kv=Kv,
            #rho=rho,
            verbose=False
        )
        # for i in range(1,model.num_joints-4,2):
        #     p.setJointMotorControl2(
        #         bodyUniqueId=model.Id,
        #         jointIndex=i,
        #         controlMode=p.TORQUE_CONTROL,
        #         force = 1e-20
        #         )
        tau_applied = model.apply_torques(
            tau_des=tau_des,
            filtered=True
        )
        ### STEP
        p.stepSimulation()
        t += dt
        ###
        ### show COM trajectory
        COM_curr = model.COM_position_world()
        p.addUserDebugLine(COM_prev.tolist(), COM_curr.tolist(), lineColorRGB=[sin(4*pi*t),sin(4*pi*(t+0.33)),sin(4*pi*(t+0.67))],lineWidth=3, lifeTime=10)
        # log to video
        if log_video:
            img=p.getCameraImage(800, 640, renderer=p.ER_BULLET_HARDWARE_OPENGL)[2]
            writer.append_data(img)
        # log data
        if log_data:
            state_history.append([model.state.q.copy(), model.state.qd.copy(), model.state.qdd.copy()])
            tau_history.append(tau_applied)
            eq_history=np.vstack((eq_history,eq))
            eCOM_history = np.vstack((eCOM_history,model.COM_position_world()[1]))
            des_history.append([q_des,qd_des,qdd_des])
        if not log_video:
            time.sleep(dt)
    return t, eq_history, eCOM_history, des_history, tau_history, state_history, p_COM_time_array
################################################### 
def single_step_mixed_torqueP_control(t,eq_history, eCOM_history, des_history, tau_history, state_history, steps, log_data=True, log_video=False):
    ### Stepping
    model.set_COM_y_ref()
    qda_des_prev=np.array(([0]*len(model.control_indices[0])))
    for i in range (steps):
        #print("\n\n\n\n##### STEP %d #####" %i)
        model.state.update(filtered_acc=True)
        COM_prev = model.COM_position_world()
        qa_des, qda_des, qdda_des = model.solve_null_COM_y_speed_optimization_qdda(
            qda_prev = qda_des_prev,
            K = K_lateral,
            filtered = False)
        qda_des_prev=qda_des
        q_des, qd_des, qdd_des, eq = model.generate_joints_trajectory(
            theta0=theta_g0,
            thetaf=theta_gf,
            ti=t,
            t_stance=t_stance,
            qa_des=qa_des,
            qda_des=qda_des,
            qdda_des=qdda_des
        )
        tau_des, eq = model.solve_computed_torque_control(
            q_des=q_des,
            qd_des=qd_des,
            qdd_des=qdd_des,
            Kp=Kp,
            Kv=Kv,
            #rho=rho,
            verbose=False
        )
        tau_applied = model.apply_torques(
            tau_des=tau_des,
            filtered=True
        )
        model.controlV_spine_lateral(K=K_lateral, fmax=fmax_list,k0=k0_lateral,velocityGain=vGain_lat_mixed,filtered=True)
        tau_applied = model.state.tau.copy()
        ### STEP
        p.stepSimulation()
        t += dt
        ###
        ### show COM trajectory
        COM_curr = model.COM_position_world()
        p.addUserDebugLine(COM_prev.tolist(), COM_curr.tolist(), lineColorRGB=[sin(4*pi*t),sin(4*pi*(t+0.33)),sin(4*pi*(t+0.67))],lineWidth=3, lifeTime=10)
        # log to video
        if log_video:
            img=p.getCameraImage(800, 600, renderer=p.ER_BULLET_HARDWARE_OPENGL)[2]
            writer.append_data(img)
        # log data
        if log_data:
            state_history.append([model.state.q.copy(), model.state.qd.copy(), model.state.qdd.copy()])
            tau_history.append(tau_applied)
            eq_history=np.vstack((eq_history,eq))
            eCOM_history = np.vstack((eCOM_history,model.COM_position_world()[1]))
            des_history.append([q_des,qd_des,qdd_des])
        time.sleep(dt)
    return t, eq_history, eCOM_history, des_history, tau_history, state_history
################################################### 
def single_step_PD_control(t,eq_history, eCOM_history, des_history, tau_history, state_history, steps, log_data=True, log_video=False):
    ### Stepping
    model.set_COM_y_ref()
    qda_des_prev=np.array(([0]*len(model.control_indices[0])))
    for i in range (steps):
        #print("\n\n\n\n##### STEP %d #####" %i)
        ###
        model.state.update(filtered_acc=False)
        ###
        COM_prev = model.COM_position_world()
        ###
        qa_des, qda_des, qdda_des = model.solve_null_COM_y_speed_optimization_qdda(
            qda_prev = qda_des_prev,
            K = K_lateral,
            filtered = True)
        qda_des_prev=qda_des
        q_des, qd_des, qdd_des, eq = model.generate_joints_trajectory(
            theta0=theta_g0,
            thetaf=theta_gf,
            ti=t,
            t_stance=t_stance,
            qa_des=qa_des,
            qda_des=qda_des,
            qdda_des=qdda_des
        )
        model.PD_control(q_des=q_des, fmax_list=fmax_list, pGain_list=pGain_list, vGain_list=vGain_list)
        tau_applied = model.state.tau.copy()
        ###
        p.stepSimulation()
        t += dt
        ### show COM trajectory
        COM_curr = model.COM_position_world()
        p.addUserDebugLine(COM_prev.tolist(), COM_curr.tolist(), lineColorRGB=[sin(4*pi*t),sin(4*pi*(t+0.33)),sin(4*pi*(t+0.67))],lineWidth=3, lifeTime=2)
        # log to video
        if log_video:
            img=p.getCameraImage(800, 600, renderer=p.ER_BULLET_HARDWARE_OPENGL)[2]
            writer.append_data(img)
        # log data
        if log_data:
            state_history.append([model.state.q.copy(), model.state.qd.copy(), model.state.qdd.copy()])
            tau_history.append(tau_applied)
            eq_history=np.vstack((eq_history,eq))
            eCOM_history = np.vstack((eCOM_history,model.COM_position_world()[1]))
            des_history.append([q_des,qd_des,qdd_des])
        #
        if not log_video:
            time.sleep(dt)
    #state_history=[q_hist,qd_hist,qdd_hist]
    return t, eq_history, eCOM_history, des_history, tau_history, state_history
###################
def single_step_P_control_vel(t,eq_history, eCOM_history, des_history, tau_history, state_history, steps, log_data=True, log_video=False):
    ### Stepping
    model.set_COM_y_ref()
    qda_des_prev=np.array(([0]*len(model.control_indices[0])))
    for i in range (steps):
        #print("\n\n\n\n##### STEP %d #####" %i)
        ###
        model.state.update(filtered_acc=False)
        #
        COM_prev = model.COM_position_world()
        #
        qa_des, qda_des, qdda_des = model.solve_null_COM_y_speed_optimization_qdda(
            qda_prev = qda_des_prev,
            K = K_lateral,
            filtered = True)
        qda_des_prev=qda_des
        q_des, qd_des, qdd_des, eq = model.generate_joints_trajectory(
            theta0=theta_g0,
            thetaf=theta_gf,
            ti=t,
            t_stance=t_stance,
            qa_des=qa_des,
            qda_des=qda_des,
            qdda_des=qdda_des
        )
        model.P_control_vel(qd_des=qd_des, fmax_list=fmax_list, pGain_list=pGain_list_vel)
        tau_applied = model.state.tau.copy()
        ###
        p.stepSimulation()
        t += dt
        ### show COM trajectory
        COM_curr = model.COM_position_world()
        p.addUserDebugLine(COM_prev.tolist(), COM_curr.tolist(), lineColorRGB=[sin(4*pi*t),sin(4*pi*(t+0.33)),sin(4*pi*(t+0.67))],lineWidth=3, lifeTime=2)
        # log to video
        if log_video:
            img=p.getCameraImage(800, 600, renderer=p.ER_BULLET_HARDWARE_OPENGL)[2]
            writer.append_data(img)
        # log data
        if log_data:
            state_history.append([model.state.q.copy(), model.state.qd.copy(), model.state.qdd.copy()])
            tau_history.append(tau_applied)
            eq_history=np.vstack((eq_history,eq))
            eCOM_history = np.vstack((eCOM_history,model.COM_position_world()[1]))
            des_history.append([q_des,qd_des,qdd_des])
        #
        time.sleep(dt)
    #state_history=[q_hist,qd_hist,qdd_hist]
    return t, eq_history, eCOM_history, des_history, tau_history, state_history
###################
def single_step_mixed_PDP_control(t,eq_history, eCOM_history, des_history, tau_history, state_history, steps, log_data=True, log_video=False):
    ### Stepping
    model.set_COM_y_ref()
    qda_des_prev=np.array(([0]*len(model.control_indices[0])))
    for i in range (steps):
        #print("\n\n\n\n##### STEP %d #####" %i)
        ###
        model.state.update(filtered_acc=False)
        ###
        COM_prev = model.COM_position_world()
        ###
        qa_des, qda_des, qdda_des = model.solve_null_COM_y_speed_optimization_qdda(
            qda_prev = qda_des_prev,
            K = K_lateral,
            filtered = False)
        q_des, qd_des, qdd_des, eq = model.generate_joints_trajectory(
            theta0=theta_g0,
            thetaf=theta_gf,
            ti=t,
            t_stance=t_stance,
            qa_des=qa_des,
            qda_des=qda_des,
            qdda_des=qdda_des
        )
        model.PD_control(q_des=q_des, fmax_list=fmax_list, pGain_list=pGain_list, vGain_list=vGain_list)
        model.controlV_spine_lateral(K=K_lateral, fmax=fmax_list,k0=k0_lateral,velocityGain=vGain_lat_mixed,filtered=True)
        tau_applied = model.state.tau.copy()
        ###
        p.stepSimulation()
        t += dt
        ### show COM trajectory
        COM_curr = model.COM_position_world()
        p.addUserDebugLine(COM_prev.tolist(), COM_curr.tolist(), lineColorRGB=[sin(4*pi*t),sin(4*pi*(t+0.33)),sin(4*pi*(t+0.67))],lineWidth=3, lifeTime=2)
        # log to video
        if log_video:
            img=p.getCameraImage(800, 600, renderer=p.ER_BULLET_HARDWARE_OPENGL)[2]
            writer.append_data(img)
        # log data
        if log_data:
            state_history.append([model.state.q.copy(), model.state.qd.copy(), model.state.qdd.copy()])
            tau_history.append(tau_applied)
            eq_history=np.vstack((eq_history,eq))
            eCOM_history = np.vstack((eCOM_history,model.COM_position_world()[1]))
            des_history.append([q_des,qd_des,qdd_des])
        #
        time.sleep(dt)
    #state_history=[q_hist,qd_hist,qdd_hist]
    return t, eq_history, eCOM_history, des_history, tau_history, state_history
###################
def single_step_torque_control_traveling_wave(t,eq_history, eCOM_history, des_history, tau_history, state_history, steps, log_data=True, log_video=False, stop_motion=False):
    model.state.update()
    model.set_COM_y_ref()
    qda_des_prev=np.array(([0]*len(model.control_indices[0])))
    p_COM_time_array = np.zeros((steps,3))
    for i in range (steps):
        #print("\n\n\n\n##### STEP %d #####" %i)
        model.state.update(filtered_acc=True)
        COM_prev = model.COM_position_world()
        p_COM_time_array[i]=model.COM_position_world()
        qa_des, qda_des, qdda_des = model.solve_null_COM_y_speed_optimization_qdda(
            qda_prev = qda_des_prev,
            K = K_lateral,
            filtered = True)
        qda_des_prev=qda_des
        if stop_motion:
            q_des, qd_des, qdd_des, eq = model.generate_joints_trajectory_stop(
                theta0=theta_g0,
                thetaf=theta_gf,
                ti=t,
                t_stance=t_stance,
                qa_des=qa_des,
                qda_des=qda_des,
                qdda_des=qdda_des
            )
        else:
            q_des, qd_des, qdd_des, eq = model.generate_joints_trajectory(
                theta0=theta_g0,
                thetaf=theta_gf,
                ti=t,
                t_stance=t_stance,
                qa_des=qa_des,
                qda_des=qda_des,
                qdda_des=qdda_des,
                cos_abd=True
            )
        tau_des, eq = model.solve_computed_torque_control(
            q_des=q_des,
            qd_des=qd_des,
            qdd_des=qdd_des,
            Kp=Kp,
            Kv=Kv,
            #rho=rho,
            verbose=False
        )
        tau_applied = model.apply_torques(
            tau_des=tau_des,
            filtered=True
        )
        ### STEP
        p.stepSimulation()
        t += dt
        ###
        ### show COM trajectory
        COM_curr = model.COM_position_world()
        p.addUserDebugLine(COM_prev.tolist(), COM_curr.tolist(), lineColorRGB=[sin(4*pi*t),sin(4*pi*(t+0.33)),sin(4*pi*(t+0.67))],lineWidth=3, lifeTime=10)
        # log to video
        if log_video:
            img=p.getCameraImage(800, 640, renderer=p.ER_BULLET_HARDWARE_OPENGL)[2]
            writer.append_data(img)
        # log data
        if log_data:
            state_history.append([model.state.q.copy(), model.state.qd.copy(), model.state.qdd.copy()])
            tau_history.append(tau_applied)
            eq_history=np.vstack((eq_history,eq))
            eCOM_history = np.vstack((eCOM_history,model.COM_position_world()[1]))
            des_history.append([q_des,qd_des,qdd_des])
        if not log_video:
            time.sleep(dt)
    return t, eq_history, eCOM_history, des_history, tau_history, state_history, p_COM_time_array
###################
##########################################
####### WALKING SIMULATION ###############
##########################################
t_stance = 0.2
### Initialization
steps = int(t_stance/dt)
t = 0.0
# Video logger settings
log_video = False
if log_video:
    video_name = "./video/small model/trial4.mp4"
    fps = int(1/dt)
    if fps>360:
        fps=360
    writer = imageio.get_writer(video_name, fps=fps)
p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
p.resetDebugVisualizerCamera(cameraDistance=0.1, cameraYaw = 50, cameraPitch=-60, cameraTargetPosition=[0,0,0])
# Data logger settings
log_data = True
eCOM_history = np.array([0])
eq_history = np.array([0]*model.state.nqd)
des_history=list()
tau_history=list()
state_history=list()
# Starting position and legs abduction range
theta_g0 = pi/6
theta_gf = -pi/2
# p.resetJointState(model.Id, model.control_indices[1][0],theta_g0)
# p.resetJointState(model.Id, model.control_indices[2][0],-theta_gf)
model.set_bent_position(theta_rg=theta_g0, theta_lg=-theta_gf,A_lat=-pi/3.5,theta_lat_0=0)
for i in range(1,model.num_joints-4,2):
    p.setJointMotorControl2(
        bodyUniqueId=model.Id,
        jointIndex=i,
        controlMode=p.TORQUE_CONTROL,
        force = -1e-20
        )
# Let the model touch the ground plane
for i in range (30):
    p.stepSimulation()
    time.sleep(dt)
##############
### RIGHT STEP
model.fix_right_foot()
model.free_left_foot()
model.fix_tail()
#model.set_right_erp(0.01)
# ### Let the model adapt to the new constraints
for i in range(30):
    p.stepSimulation()
    time.sleep(dt)
### Start the simulation
model.state.update()
model.integrator_lateral_qa.reset(model.state.q[model.mask_act_shifted])
t,eq_history, eCOM_history, des_history, tau_history, state_history, p_COM_time_array = single_step_torque_control(
    t,
    eq_history,
    eCOM_history,
    des_history,
    tau_history,
    state_history,
    steps=steps,
    log_data=log_data,
    log_video=log_video
    )
# t,eq_history, eCOM_history, des_history, tau_history, state_history = single_step_PD_control(
#     t,
#     eq_history,
#     eCOM_history,
#     des_history,
#     tau_history,
#     state_history,
#     steps=steps
#     )


if  log_video:
    writer.close()
p.disconnect()


##########################################
####### SHOW LOGGED DATA #################
##########################################
if log_data:
    tau_history = np.array(tau_history)
    #print("Shape des_history: ", des_history.shape)
    qa_history = np.array(state_history[0][0])[model.mask_act_shifted]
    qda_history = np.array(state_history[0][1])[model.mask_act]
    qdda_history = np.array(state_history[0][2])[model.mask_act]
    qrg_history = (np.array(state_history[0][0][model.state.nq-4]))
    qdrg_history = (np.array(state_history[0][1][model.state.nqd-4]))
    qddrg_history = (np.array(state_history[0][2][model.state.nqd-4]))
    qlg_history = (np.array(state_history[0][0][model.state.nq-2]))
    qdlg_history = (np.array(state_history[0][1][model.state.nqd-2]))
    qddlg_history = (np.array(state_history[0][2][model.state.nqd-2]))
    for i in range(1,len(state_history)):
        qa_history = np.vstack((qa_history, np.array(state_history[i][0])[model.mask_act_shifted]))
        qda_history = np.vstack((qda_history, np.array(state_history[i][1])[model.mask_act]))
        qdda_history = np.vstack((qdda_history, np.array(state_history[i][2])[model.mask_act]))
        qrg_history = np.vstack((qrg_history, np.array(state_history[i][0][model.state.nq-4])))
        qdrg_history = np.vstack((qdrg_history, np.array(state_history[i][1][model.state.nqd-4])))
        qddrg_history = np.vstack((qddrg_history, np.array(state_history[i][2][model.state.nqd-4])))
        qlg_history = np.vstack((qlg_history, np.array(state_history[i][0][model.state.nq-2])))
        qdlg_history = np.vstack((qdlg_history, np.array(state_history[i][1][model.state.nqd-2])))
        qddlg_history = np.vstack((qddlg_history, np.array(state_history[i][2][model.state.nqd-2])))
    ###
    qa_des_history = (np.array(des_history[0][0]))[model.mask_act_shifted]
    qda_des_history = np.array(des_history[0][1])[model.mask_act]
    qdda_des_history = np.array(des_history[0][2])[model.mask_act]
    qrg_des_history = (np.array(des_history[0][0][model.state.nq-4]))
    qdrg_des_history = (np.array(des_history[0][1][model.state.nqd-4]))
    qddrg_des_history = (np.array(des_history[0][2][model.state.nqd-4]))
    qlg_des_history = (np.array(des_history[0][0][model.state.nq-2]))
    qdlg_des_history = (np.array(des_history[0][1][model.state.nqd-2]))
    qddlg_des_history = (np.array(des_history[0][2][model.state.nqd-2]))
    for i in range(1,len(des_history)):
        qa_des_history=np.vstack((qa_des_history,np.array(des_history[i][0])[model.mask_act_shifted]))
        qda_des_history=np.vstack((qda_des_history,np.array(des_history[i][1])[model.mask_act]))
        qdda_des_history=np.vstack((qdda_des_history,np.array(des_history[i][2])[model.mask_act]))
        qrg_des_history = np.vstack((qrg_des_history, np.array(des_history[i][0][model.state.nq-4])))
        qdrg_des_history = np.vstack((qdrg_des_history, np.array(des_history[i][1][model.state.nqd-4])))
        qddrg_des_history = np.vstack((qddrg_des_history, np.array(des_history[i][2][model.state.nqd-4])))
        qlg_des_history = np.vstack((qlg_des_history, np.array(des_history[i][0][model.state.nq-2])))
        qdlg_des_history = np.vstack((qdlg_des_history, np.array(des_history[i][1][model.state.nqd-2])))
        qddlg_des_history = np.vstack((qddlg_des_history, np.array(des_history[i][2][model.state.nqd-2])))
    #qa_des_history = qa_des_history+qa_history[0]
    ###
    eqa_history = eq_history[0][model.mask_act]
    eqrg_history = eq_history[0][model.state.nqd-4]
    eqlg_history = eq_history[0][model.state.nqd-2]
    for i in range(1,eq_history.shape[0]):
        eqa_history = np.vstack((eqa_history,eq_history[i][model.mask_act]))
        eqrg_history = np.vstack((eqrg_history,eq_history[i][model.state.nqd-4]))
        eqlg_history = np.vstack((eqlg_history,eq_history[i][model.state.nqd-2]))
    ###
    tau_lateral_history=np.array(tau_history[0])[model.mask_act]
    tau_rg_history=np.array(tau_history[0][model.state.nqd-4])
    tau_lg_history=np.array(tau_history[0][model.state.nqd-2])
    for i in range(1,tau_history.shape[0]):
        tau_lateral_history=np.vstack((tau_lateral_history,np.array(tau_history[i])[model.mask_act]))
        tau_rg_history=np.vstack((tau_rg_history, tau_history[i][model.state.nqd-4]))
        tau_lg_history=np.vstack((tau_lg_history, tau_history[i][model.state.nqd-2]))
    ###
    q_history = state_history[0]
    ###
    fig, axs = plt.subplots(3,3)
    #plt.plot(e,"b")
    # for i in range(eq_history.shape[1]):
    #     axs[0,0].plot(eq_history[:,i])
    # axs[0,0].set_title("eq\nLateral joints positional error")
    axs[0,0].plot(eCOM_history, color="xkcd:red")
    axs[0,0].set_title("eCOM")
    for i in range(1,1+qda_des_history.shape[1]):
        #axs[i//3,i%3].plot(qda_history[:,i-1], color="xkcd:green")
        #axs[i//3,i%3].plot(qdda_history[:,i-1], color="xkcd:light green")
        #axs[i//3,i%3].plot(qda_des_history[:,i-1], color="xkcd:teal")
        #axs[i//3,i%3].plot(qdda_des_history[:,i-1], color="xkcd:light teal")
        axs[i//3,i%3].plot(tau_lateral_history[:,i-1], color="xkcd:salmon")
        axs[i//3,i%3].plot(eqa_history[:,i-1], color="xkcd:red")
        axs[i//3,i%3].plot(qa_des_history[:,i-1], color="xkcd:dark teal")
        axs[i//3,i%3].plot(qa_history[:,i-1], color="xkcd:yellow")
        axs[i//3,i%3].set_title("Lateral joint %d" %i)
    #
    fig_g, axs_g = plt.subplots(1,2)
    #
    #axs_g[1].plot(qdlg_history, color="xkcd:green")
    #axs_g[1].plot(qddlg_history, color="xkcd:light green")
    #axs_g[1].plot(qdlg_des_history, color="xkcd:teal")
    #axs_g[1].plot(qddlg_des_history, color="xkcd:light teal")
    axs_g[1].plot(tau_lg_history, color="xkcd:salmon")
    axs_g[1].plot(eqlg_history, color="xkcd:red")
    axs_g[1].plot(qlg_des_history, color="xkcd:dark teal")
    axs_g[1].plot(qlg_history, color="xkcd:yellow")
    axs_g[1].set_title("Left abduction")
    #
    #axs_g[0].plot(qdrg_history, color="xkcd:green")
    #axs_g[0].plot(qddrg_history, color="xkcd:light green")
    #axs_g[0].plot(qdrg_des_history, color="xkcd:teal")
    #axs_g[0].plot(qddrg_des_history, color="xkcd:light teal")
    axs_g[0].plot(tau_rg_history, color="xkcd:salmon")
    axs_g[0].plot(eqrg_history, color="xkcd:red")
    axs_g[0].plot(qrg_des_history, color="xkcd:dark teal")
    axs_g[0].plot(qrg_history, color="xkcd:yellow")
    axs_g[0].set_title("Right abduction")
    # fig_tau, axs_tau = plt.subplots(3,3)
    # for i in range(1,1+qda_des_history.shape[1]):
    #     axs_tau[i//3,i%3].plot(tau_lateral_history[:,i-1],color="tab:orange")
    #     axs_tau[i//3,i%3].plot(tau_lateral_f_history[:,i-1],color="tab:brown")
    #     #axs[i//3,i%3].plot(eq_history[:,i-1], color="tab:pink")
    #     axs_tau[i//3,i%3].set_title("Tau %d" %i)
    # fig_COM= plt.figure()
    # axs_COM = fig_COM.add_subplot(111)
    # axs_COM.plot(p_COM_time_array[:,0], p_COM_time_array[:,1],color="xkcd:teal")
    # axs_COM.set_title("COM x-y trajectory")
    # axs_COM.set_aspect('equal')

    plt.show()







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
