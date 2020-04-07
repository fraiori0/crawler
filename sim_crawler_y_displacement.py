import pybullet as p
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import time
import pybullet_data
import crawler as crw
from math import *
import subprocess as sp
import imageio
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, pyll
import pandas as pd

matplotlib.use('TkAgg')


def run_simulation(dt, t_stance, duration,
    f, A_lat,th0_lat, th0_abd, thf_abd,
    z_rotation,
    girdle_friction, body_friction, last_link_friction, leg_friction,
    K_lateral, k0_lateral,
    Kp_lat, Kp_r_abd, Kp_l_abd, Kp_flex,
    Kv_lat, Kv_r_abd, Kv_l_abd, Kv_flex,
    keep_in_check=True,
    graphic_mode=False, plot_graph_joint=False, plot_graph_COM=False,
    video_logging=False, video_name="./video/dump/dump.mp4"):
    # dt, t_stance, duration = time parameter, dt MUST be the same used to create the model
    # A,f1,n,t2_off,delta_off,bias = parameters for the generation of the torques
    # theta_rg0, theta_lg0, A_lat_0, theta_lat_0= parameters for setting the starting position
    # girdle_friction, body_friction, leg_friction = lateral friction values
    ##########################################
    ####### PYBULLET SET-UP ##################
    if graphic_mode:
        physicsClient = p.connect(p.GUI)
    else:
        physicsClient = p.connect(p.DIRECT)
    p.setTimeStep(dt)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0,0,-9.81)
    ### plane set-up
    planeID = p.loadURDF("plane100.urdf", [0,0,0])
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
    model = crw.Crawler(
        urdf_path="/home/fra/Uni/Tesi/crawler", 
        dt_simulation=dt,
        base_position=[0,0,0.05],
        base_orientation=[0,0,sin(z_rotation/2),cos(z_rotation/2)],
        mass_distribution=True
        )
    # for i,elem in enumerate(model.links_state_array):
    #     print("Link %d  "%i, elem["world_com_trn"])
    # print(model.links_state_array)
    # Set motion damping ("air friction")
    for index in range(-1,model.num_joints):
        p.changeDynamics(
            model.Id,
            index,
            linearDamping=0.0001,
            angularDamping=0.0001
        )
    # Girdle link dynamic properties
    p.changeDynamics(model.Id,
        linkIndex = -1,
        lateralFriction = girdle_friction,
        spinningFriction = girdle_friction/100,
        rollingFriction = girdle_friction/100,
        restitution = 0.1,
        #contactStiffness = 0,
        #contactDamping = 0
        )
    # Body dynamic properties
    for i in range(1,model.num_joints-4,2):
        p.changeDynamics(model.Id,
            linkIndex = i,
            lateralFriction = body_friction,
            spinningFriction = body_friction/100,
            rollingFriction = body_friction/100,
            restitution = 0.1,
            #contactStiffness = 0,
            #contactDamping = 0
            )
    # # Last link dynamic properties
    p.changeDynamics(model.Id,
        linkIndex = (model.control_indices[0][-1]+1),
        lateralFriction = last_link_friction,
        spinningFriction = last_link_friction/100000,
        rollingFriction = last_link_friction/100000,
        restitution = 0.1,
        #contactStiffness = 0,
        #contactDamping = 0
        )
    # Legs dynamic properties
    for i in range(model.num_joints-3,model.num_joints,2):
        p.changeDynamics(model.Id,
            linkIndex = i,
            lateralFriction = leg_friction,
            spinningFriction = leg_friction/100,
            rollingFriction = leg_friction/100,
            restitution = 0.1,
            #contactStiffness = 0,
            #contactDamping = 0
            )
    # Dorsal joints "compliance"
    model.turn_off_crawler()
    for i in range(1,model.control_indices[0][-1]+1,2):
        p.setJointMotorControl2(model.Id, i, 
            controlMode=p.VELOCITY_CONTROL, 
            targetVelocity=0, 
            force=0.05
            )
    # Starting position
    model.set_bent_position(theta_rg=th0_abd, theta_lg=-thf_abd,A_lat=A_lat,theta_lat_0=th0_lat)
    ##########################################
    ####### CONTROLLER PARAMETERS ############
    Kp = model.generate_Kp(Kp_lat, Kp_r_abd, Kp_l_abd, Kp_flex)
    Kv = model.generate_Kv(Kv_lat, Kv_r_abd, Kv_l_abd, Kv_flex)
    model.set_low_pass_lateral_qa(fc=10)
    model.set_low_pass_tau(fc=90)
    ##########################################
    ####### MISCELLANEOUS ####################
    if graphic_mode:
        p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
        p.resetDebugVisualizerCamera(cameraDistance=0.8, cameraYaw = 50, cameraPitch=-60, cameraTargetPosition=[0,0,0])
    if (video_logging and graphic_mode):
        video_writer = imageio.get_writer(video_name, fps=int(1/dt))
    ##########################################
    ####### PERFORMANCE EVALUATION ###########
    loss = 0.0
    max_loss = 10000000.0 # value to set when something goes wrong, if constraints are "hard"
    max_COM_height = 4*model.body_sphere_radius
    max_COM_speed = 10 #TODO: check this value
    n_mean = 6
    ##########################################
    ####### TIME SERIES CREATION #############
    steps = int(duration/dt)
    steps_stance = int(t_stance/dt)
    num_lat = len(model.control_indices[0])
    f_walk = 1/(2*t_stance)
    # Data array initialization
    p_COM_time_array = np.zeros((steps,3))
    v_COM_time_array = np.zeros((steps,3))
    tau_time_array = np.zeros((steps,model.state.nqd))
    joint_power_time_array = np.zeros((steps,model.state.nqd))
    q_time_array = np.zeros((steps,model.state.nq))
    qd_time_array = np.zeros((steps,model.state.nqd))
    qdd_time_array = np.zeros((steps,model.state.nqd))
    eq_time_array = np.zeros((steps,model.state.nq))
    q_des_time_array = np.zeros((steps,model.state.nq))
    qd_des_time_array = np.zeros((steps,model.state.nqd))
    qdd_des_time_array = np.zeros((steps,model.state.nqd))
    # Integrator for the work done by each joint
    joint_power_integrator = crw.Integrator_Forward_Euler(dt,np.zeros(model.state.nqd))
    joint_energy_time_array = np.zeros((steps,model.state.nqd))
    ##########################################
    ####### RUN SIMULATION ###################
    t = 0.0
    # let the legs fall and leave the time to check the starting position
    for i in range(100):
        p.stepSimulation()
        if graphic_mode:
            time.sleep(dt)
    # Initial Condition
    model.state.update()
    model.integrator_lateral_qa.reset(model.state.q[model.mask_act_shifted])
    model.free_right_foot()
    model.fix_left_foot()
    model.fix_tail()
    model.set_COM_y_ref()
    qda_des_prev=np.array(([0]*len(model.control_indices[0])))
    # walk and record data
    for i in range(steps):
        # UPDATE CONSTRAINTS
        if not (i%steps_stance):
            model.invert_feet()
            print("step")
            model.turn_off_crawler()
            for tmp in range(1):
                p.stepSimulation()
        # UPDATE STATE
        model.state.update()
        p_COM_curr = np.array(model.COM_position_world())
        v_COM_curr = np.array(model.COM_velocity_world())
        # ADD FRAME TO VIDEO
        if (graphic_mode and video_logging):
            img=p.getCameraImage(800, 640, renderer=p.ER_BULLET_HARDWARE_OPENGL)[2]
            video_writer.append_data(img)
        # ACTUATE MODEL AND STEP SIMULATION
        qa_des, qda_des, qdda_des = model.solve_null_COM_y_speed_optimization_qdda(
            qda_prev = qda_des_prev,
            K = K_lateral,
            k0=k0_lateral,
            filtered = True)
        qda_des_prev=qda_des
        q_des, qd_des, qdd_des, eq = model.generate_joints_trajectory(
                theta0=th0_abd,
                thetaf=thf_abd,
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
            filtered=True,
            passive_body=False
        )
        # UPDATE TIME-ARRAYS
        if plot_graph_joint or plot_graph_COM:
            q_time_array[i] = model.state.q.copy()
            qd_time_array[i] = model.state.qd.copy()
            qdd_time_array[i] = model.state.qdd.copy()
            q_des_time_array[i] = q_des.copy()
            qd_des_time_array[i] = qd_des.copy()
            qdd_des_time_array[i] = qdd_des.copy()
            p_COM_time_array[i]=p_COM_curr
            v_COM_time_array[i]=v_COM_curr
            eq_time_array[i] = q_des_time_array[i] - q_time_array[i]
            tau_time_array[i] = tau_applied
            for j, (v_j, tau_j) in enumerate(zip(model.state.qd,tau_applied)):
                joint_power_time_array[i,j] = abs(v_j*tau_j)
            joint_energy = joint_power_integrator.integrate(joint_power_time_array[i])
            joint_energy_time_array[i] = joint_energy
        # print("eq:      ", np.round(eq[model.mask_act],2), np.round(eq[model.mask_both_legs],2))
        # print("tau:     ", np.round(tau_applied[model.mask_act],2), np.round(tau_applied[model.mask_both_legs],2))
        # print("qdd_des: ", np.round(qdd_des_time_array[i][model.mask_act],2), np.round(qdd_des_time_array[i][model.mask_both_legs],2))
        #UPDATE LOSS - VARIATIONAL VERSION -d(x-displacement) + d(total energy expenditure)
        # loss += (
        #     -100*(p_COM_time_array[i][0]-p_COM_time_array[i-1][0]) + 
        #     0.1*(joint_power_time_array[i]).dot(joint_power_time_array[i])
        #     )
        # STEP SIMULATION
        p.stepSimulation()
        if graphic_mode:
            time.sleep(dt) 
        t += dt
        # CHECK SIMULATION to avoid that the model get schwifty
        if keep_in_check:
            # check COM height, model shouldn't fly or go through the ground
            if (p_COM_curr[-1] > (max_COM_height)) or (p_COM_curr[-1] < 0.0):
                print("\nLook at me I'm jumping\n")
                loss += max_loss
                break
            if (np.linalg.norm(np.mean(v_COM_time_array[i-n_mean:i+1], 0)) > max_COM_speed):
                print("\nFaster than light. WOOOOOOOSH\n")
                loss += max_loss
                break
    p.disconnect()
    ####### END SIMULATION ###################
    if (video_logging and graphic_mode):
        video_writer.close()
    ##########################################
    ####### PLOT DATA ########################
    if plot_graph_joint:
        fig_lat, axs_lat = plt.subplots(2,3)
        for i in range(0,num_lat):
            #axs_lat[i//3,i%3].plot(qd_time_array[:,model.mask_act][:,i], color="xkcd:dark yellow", label="qd")
            #axs_lat[i//3,i%3].plot(qdd_time_array[:,model.mask_act][:,i], color="xkcd:pale yellow", label="qdd")
            #axs_lat[i//3,i%3].plot(joint_power_time_array[:,model.mask_act][:,i], color="xkcd:light salmon", label="Power")
            #axs_lat[i//3,i%3].plot(joint_energy_time_array[:,model.mask_act][:,i], color="xkcd:dark salmon", label="Energy")
            axs_lat[i//3,i%3].plot(tau_time_array[:,model.mask_act][:,i], color="xkcd:salmon", label="Torque")
            axs_lat[i//3,i%3].plot(q_time_array[:,model.mask_act_shifted][:,i], color="xkcd:yellow", label="q")
            axs_lat[i//3,i%3].plot(q_des_time_array[:,model.mask_act_shifted][:,i], color="xkcd:dark teal", label="q_des")
            #axs_lat[i//3,i%3].plot(eq_time_array[:,model.mask_act_shifted][:,i], color="xkcd:red", label="error")
            axs_lat[i//3,i%3].set_title("Lateral joint %d" %i)
        handles_lat, labels_lat = axs_lat[0,0].get_legend_handles_labels()
        fig_lat.legend(handles_lat, labels_lat, loc='center right')
        #
        fig_girdle, axs_girdle = plt.subplots(2,2)
        for i in range(0,len(model.mask_both_legs)):
            #axs_girdle[i//2,i%2].plot(qd_time_array[:,model.mask_both_legs][:,i], color="xkcd:dark yellow", label="qd")
            #axs_girdle[i//2,i%2].plot(qdd_time_array[:,model.mask_both_legs][:,i], color="xkcd:pale yellow", label="qdd")
            #axs_girdle[i//2,i%2].plot(joint_power_time_array[:,model.mask_both_legs][:,i], color="xkcd:light salmon", label="Power")
            #axs_girdle[i//2,i%2].plot(joint_energy_time_array[:,model.mask_both_legs][:,i], color="xkcd:dark salmon", label="Energy")
            axs_girdle[i//2,i%2].plot(tau_time_array[:,model.mask_both_legs][:,i], color="xkcd:salmon", label="Torque")
            axs_girdle[i//2,i%2].plot(q_time_array[:,model.mask_both_legs_shifted][:,i], color="xkcd:yellow", label="q")
            axs_girdle[i//2,i%2].plot(q_des_time_array[:,model.mask_both_legs_shifted][:,i], color="xkcd:dark teal", label="q_des")
            axs_girdle[i//2,i%2].plot(eq_time_array[:,model.mask_both_legs_shifted][:,i], color="xkcd:red", label="error")
        axs_girdle[0,0].set_title("Right abduction")
        axs_girdle[0,1].set_title("Right flexion")
        axs_girdle[1,0].set_title("Left abduction")
        axs_girdle[1,1].set_title("Left flexion")
        handles_girdle, labels_girdle = axs_girdle[0,0].get_legend_handles_labels()
        fig_girdle.legend(handles_girdle, labels_girdle, loc='center right')
        #
        plt.show()
        #
    if plot_graph_COM:
        fig_COM = plt.figure()
        axs_COM = fig_COM.add_subplot(111)
        axs_COM.plot(p_COM_time_array[:,0], p_COM_time_array[:,1],color="xkcd:teal", linewidth=4)
        axs_COM.set_title("COM x-y trajectory")
        axs_COM.set_xlim(-0.22,0.01)
        axs_COM.set_ylim(-0.16,0.01)
        axs_COM.set_aspect('equal')
        # fig_COM_3D = plt.figure()
        # axs_COM_3D = fig_COM_3D.add_subplot(111, projection='3d')
        # axs_COM_3D.plot(p_COM_time_array[:,0], p_COM_time_array[:,1], p_COM_time_array[:,2],color="xkcd:teal")
        # axs_COM_3D.set_title("COM 3D trajectory")
        #
        plt.show()
    return loss

t_stance = 0.65
run_simulation(
    dt=1./360., t_stance=t_stance, duration=t_stance,
    f=1/(2*t_stance), A_lat=-pi/3.5, th0_lat=0.0, th0_abd=pi/8, thf_abd=-pi/2, z_rotation=pi/3,
    girdle_friction=0.2, body_friction=0.1, last_link_friction=0.1, leg_friction=0.0001,
    # Kp_lat=0, Kp_r_abd=0, Kp_l_abd=0, Kp_flex=0,
    # Kv_lat=0, Kv_r_abd=0, Kv_l_abd=0, Kv_flex=0,
    K_lateral=3000, k0_lateral=1,
    Kp_lat=70e3, Kp_r_abd=70e3, Kp_l_abd=70e3, Kp_flex=20e3,
    Kv_lat=20e3, Kv_r_abd=50e3, Kv_l_abd=50e3, Kv_flex=10e3,
    keep_in_check=False, graphic_mode=True, plot_graph_joint=True, plot_graph_COM=True,
    video_logging=False, video_name="./video/scaled model ydisp/hinged_tail_pi82_A35_z3.mp4"
)

# 70p 50v abd pi/8 -pi/2 abd -pi/3.5 A_lat