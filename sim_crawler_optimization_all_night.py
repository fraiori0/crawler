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

n_iter = 1500

def run_simulation1(dt, t_stance, duration,
    A, f1, n, t1_off, delta_off, bias,
    theta_rg0, theta_lg0, A_lat_0, theta_lat_0,
    girdle_friction, body_friction, leg_friction,
    keep_in_check=True, soft_constraints=False,
    graphic_mode=False, plot_graph=False, video_logging=False, video_name="./video/dump/dump"):
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
    z_rotation = 0#pi/3.5 #radians
    model = crw.Crawler(urdf_path="/home/fra/Uni/Tesi/crawler", dt_simulation=dt, base_position=[0,0,0.05], base_orientation=[0,0,sin(z_rotation/2),cos(z_rotation/2)])
    # for i,elem in enumerate(model.links_state_array):
    #     print("Link %d  "%i, elem["world_com_trn"])
    # print(model.links_state_array)
    # Set motion damping ("air friction")
    for index in range(-1,model.num_joints):
        p.changeDynamics(
            model.Id,
            index,
            linearDamping=0.01,
            angularDamping=0.01
        )
    # Girdle link dynamic properties
    p.changeDynamics(model.Id,
        linkIndex = -1,
        lateralFriction = girdle_friction,
        spinningFriction = girdle_friction/10,
        rollingFriction = girdle_friction/20,
        restitution = 0.1,
        #contactStiffness = 0,
        #contactDamping = 0
        )
    # Body dynamic properties
    for i in range(0,model.num_joints-4):
        p.changeDynamics(model.Id,
            linkIndex = i,
            lateralFriction = body_friction,
            spinningFriction = body_friction/10,
            rollingFriction = body_friction/20,
            restitution = 0.1,
            #contactStiffness = 0,
            #contactDamping = 0
            )
    # Legs dynamic properties
    for i in range(model.num_joints-4,model.num_joints):
        p.changeDynamics(model.Id,
            linkIndex = i,
            lateralFriction = leg_friction,
            spinningFriction = leg_friction/10,
            rollingFriction = leg_friction/20,
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
    #model.set_bent_position(theta_rg=theta_rg0, theta_lg=theta_lg0,A_lat=A_lat_0,theta_lat_0=0.0)
    ##########################################
    ####### MISCELLANEOUS ####################
    if graphic_mode:
        p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
        p.resetDebugVisualizerCamera(cameraDistance=0.5, cameraYaw = 50, cameraPitch=-60, cameraTargetPosition=[0,0,0])
    if (video_logging and graphic_mode):
        video_writer = imageio.get_writer(video_name, fps=int(1/dt))
    ##########################################
    ####### PERFORMANCE EVALUATION ###########
    loss = 0.0
    max_loss = 10000000.0 # value to set when something goes wrong, if constraints are "hard"
    max_COM_height = 2*model.body_sphere_radius
    max_COM_speed = 0.8 #TODO: check this value
    n_mean = 3
    ##########################################
    ####### TIME SERIES CREATION #############
    steps = int(duration/dt)
    num_lat = len(model.control_indices[0])
    # Torques evolution
    tau_array_fun = model.get_torques_profile_fun(A,f1,n,t1_off,delta_off,bias)
    tau_time_array = model.generate_torques_time_array(tau_array_fun,duration,t0=0.0,include_base=True)
    # Data array initialization
    p_COM_time_array = np.zeros((steps,3))
    v_COM_time_array = np.zeros((steps,3))
    joint_power_time_array = np.zeros((steps,model.state.nqd))
    q_time_array = np.zeros((steps,model.state.nq))
    qd_time_array = np.zeros((steps,model.state.nqd))
    qdd_time_array = np.zeros((steps,model.state.nqd))
    # Integrator for the work done by each joint
    joint_power_integrator = crw.Integrator_Forward_Euler(dt,np.zeros(model.state.nqd))
    joint_energy_time_array = np.zeros((steps,model.state.nqd))
    ##########################################
    ####### RUN SIMULATION ###################
    t = 0.0
    # let the leg fall
    for i in range(30):
        p.stepSimulation()
        if graphic_mode:
            time.sleep(dt)
    # walk and record data
    for i in range(steps):
        model.state.update()
        q_time_array[i] = model.state.q.copy()
        qd_time_array[i] = model.state.qd.copy()
        p_COM_curr = np.array(model.COM_position_world())
        v_COM_curr = np.array(model.COM_velocity_world())
        qdd_time_array[i] = model.state.qdd.copy()
        p_COM_time_array[i]=p_COM_curr
        v_COM_time_array[i]=v_COM_curr
        for j, (v_j, tau_j) in enumerate(zip(model.state.qd,tau_time_array[i])):
            joint_power_time_array[i,j] = abs(tau_j)
        joint_energy = joint_power_integrator.integrate(joint_power_time_array[i])
        joint_energy_time_array[i] = joint_energy
        ### UPDATE LOSS
        # -(x-displacement) + (total energy expenditure)
        #loss = -(p_COM_time_array[i][0]-p_COM_time_array[0][0]) + (joint_energy_time_array[-1]).dot(joint_energy_time_array[-1])
        # VARIATIONAL VERSION -d(x-displacement) + d(total energy expenditure)
        loss += (
            -100*(p_COM_time_array[i][0]-p_COM_time_array[i-1][0]) + 
            0.1*(joint_power_time_array[i]).dot(joint_power_time_array[i])
            )
        # if soft_constraints:
        #     loss+= (np.exp((p_COM_curr[-1]/(max_COM_height))**2)-1.0 + 
        #         np.exp(((np.linalg.norm(np.mean(v_COM_time_array[i-n_mean:i+1], 0)))/max_COM_speed)**2)-1.0)
        ###
        if (graphic_mode and video_logging):
            img=p.getCameraImage(800, 640, renderer=p.ER_BULLET_HARDWARE_OPENGL)[2]
            video_writer.append_data(img)
        ###
        model.apply_torques(tau_time_array[i], filtered=False)
        p.stepSimulation()
        if graphic_mode:
            time.sleep(dt) 
        t += dt
        ### check simulation to avoid that the model get schwifty
        # can be substituted by soft constraints for the violation of the two conditions
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
    if plot_graph:
        fig_lat, axs_lat = plt.subplots(2,3)
        for i in range(0,num_lat):
            #axs_lat[i//3,i%3].plot(qd_time_array[:,model.mask_act][:,i], color="xkcd:teal", label="qd")
            #axs_lat[i//3,i%3].plot(qdd_time_array[:,model.mask_act][:,i], color="xkcd:light teal", label="qdd")
            axs_lat[i//3,i%3].plot(joint_power_time_array[:,model.mask_act][:,i], color="xkcd:salmon", label="Power")
            axs_lat[i//3,i%3].plot(joint_energy_time_array[:,model.mask_act][:,i], color="xkcd:light salmon", label="Energy")
            axs_lat[i//3,i%3].plot(tau_time_array[:,model.mask_act][:,i], color="xkcd:dark salmon", label="Torque")
            axs_lat[i//3,i%3].plot(q_time_array[:,model.mask_act_shifted][:,i], color="xkcd:dark teal", label="q")
            axs_lat[i//3,i%3].set_title("Lateral joint %d" %i)
        handles_lat, labels_lat = axs_lat[0,0].get_legend_handles_labels()
        #
        fig_lat.legend(handles_lat, labels_lat, loc='center left')
        fig_girdle, axs_girdle = plt.subplots(2,2)
        for i in range(0,len(model.mask_both_legs)):
            #axs_girdle[i//2,i%2].plot(qd_time_array[:,model.mask_both_legs][:,i], color="xkcd:teal", label="qd")
            #axs_girdle[i//2,i%2].plot(qdd_time_array[:,model.mask_both_legs][:,i], color="xkcd:light teal", label="qdd")
            axs_girdle[i//2,i%2].plot(joint_power_time_array[:,model.mask_both_legs][:,i], color="xkcd:salmon", label="Power")
            axs_girdle[i//2,i%2].plot(joint_energy_time_array[:,model.mask_both_legs][:,i], color="xkcd:light salmon", label="Energy")
            axs_girdle[i//2,i%2].plot(tau_time_array[:,model.mask_both_legs][:,i], color="xkcd:dark salmon", label="Torque yeah")
            axs_girdle[i//2,i%2].plot(q_time_array[:,model.mask_both_legs_shifted][:,i], color="xkcd:dark teal", label="q")
        axs_girdle[0,0].set_title("Right abduction")
        axs_girdle[0,1].set_title("Right flexion")
        axs_girdle[1,0].set_title("Left abduction")
        axs_girdle[1,1].set_title("Left flexion")
        handles_girdle, labels_girdle = axs_girdle[0,0].get_legend_handles_labels()
        fig_girdle.legend(handles_girdle, labels_girdle, loc='center left')
        #
        plt.show()
        #
        fig_COM = plt.figure()
        axs_COM = fig_COM.add_subplot(111)
        axs_COM.plot(p_COM_time_array[:,0], p_COM_time_array[:,1],color="xkcd:teal")
        axs_COM.set_title("COM x-y trajectory")
        fig_COM_3D = plt.figure()
        axs_COM_3D = fig_COM_3D.add_subplot(111, projection='3d')
        axs_COM_3D.plot(p_COM_time_array[:,0], p_COM_time_array[:,1], p_COM_time_array[:,2],color="xkcd:teal")
        axs_COM_3D.set_title("COM 3D trajectory")
        #
        plt.show()
    return loss

def generate_parameters_array1(t_stance,
                            A_lat, k_lat, A_abd, A_flex,
                            n_lat, n_abd,
                            t_lat, t_lat_delay, t_abd, t_flex,
                            delta_lat_1, delta_lat_2, delta_lat_3, delta_abd,
                            k_bias_abd, k_bias_flex):
    ### TIME VARIABLES
    dt = 1/240
    duration = 10
    f_walk = 1/(2*t_stance)
    ### TORQUES EVOLUTION VARIABLEs
    # A_lat*(2*k_lat) < 1, 1Nm is a lot already (tested)
    A = (
        (A_lat,
        A_lat*(1+k_lat),
        A_lat*(2+k_lat),
        A_lat*(1+k_lat),
        A_lat),
        (A_abd,A_flex),
        (A_abd,A_flex))
    f1 = ((f_walk,f_walk,f_walk,f_walk,f_walk),(f_walk,f_walk),(f_walk,f_walk))
    #TODO give the option to set n_lat=0 so that abduction can also be a pure cosine wave
    n = ((n_lat,n_lat,n_lat,n_lat,n_lat),(n_lat,0),(n_lat,0))
    # left leg is just as the right leg but translated by half a period
    t1_off = (
        (t_lat,
        t_lat + 1*t_lat_delay,
        t_lat + 2*t_lat_delay,
        t_lat + 3*t_lat_delay,
        t_lat + 4*t_lat_delay),
        (t_abd, t_flex),
        (t_abd+t_stance, t_flex+t_stance)
        )
    # For delta, considering symmetry on the middle spinal joint of the shape of the functions.
    # delta_abd = 0 and n_abd=0 to make the flexion a single sine wave.
    # if (int(n_lat) == 0):
    #     for delta_lat in (delta_lat_1,delta_lat_2,delta_lat_3):
    #         delta_lat = pi/2
    # if (int(n_abd) == 0):
    #     delta_abd = pi/2
    delta_off = (
        (delta_lat_1,
        delta_lat_2,
        delta_lat_3,
        delta_lat_2,
        delta_lat_1
        ),
        (delta_abd, 0),
        (delta_abd, 0))
    # for correct bias on flexion and abduction check how reference system are oriented
    bias = ((0,0,0,0,0),(-k_bias_abd*A_abd,k_bias_flex*A_flex),(k_bias_abd*A_abd,-k_bias_flex*A_flex))
    ### STARTING POSITION VARIABLES
    theta_rg0=pi/12
    theta_lg0=-pi/4
    A_lat_0=-pi/3.5
    theta_lat_0=0.0
    #
    girdle_friction = 0.3
    body_friction = 0.3
    leg_friction = 0.3
    return (dt, t_stance, duration,
    A, f1, n, t1_off, delta_off, bias,
    theta_rg0, theta_lg0, A_lat_0, theta_lat_0,
    girdle_friction, body_friction, leg_friction)

def objective_function1(params):
    #value not contained in params must be passed manually, as below
    loss = run_simulation1(
        *generate_parameters_array1(
            **params,
            t_stance = 0.5,
            n_lat=2, n_abd=2,
            k_bias_abd=0.5, k_bias_flex=0.5
        ),
        keep_in_check=True,soft_constraints=False,
        graphic_mode=False, plot_graph=False
    )
    return loss

def run_bayesian_optimization1(objective_function,param_dist,iterations,filename="trial.csv",save=True):
    trials = Trials()
    seed=int(time.time())
    best_param = fmin(
        objective_function, 
        param_dist, 
        algo=tpe.suggest, 
        max_evals=iterations, 
        trials=trials,
        rstate=np.random.RandomState(seed)
        )
    losses = [x['result']['loss'] for x in trials.trials]
    vals = [x['misc']['vals']for x in trials.trials]
    best_param_values = [val for val in best_param.values()]
    best_param_values = [x for x in best_param.values()]
    print("Best loss obtained: %f\n with parameters: %s" % (min(losses), best_param_values))
    if save:
        dict_csv={}
        dict_csv.update({'Score' : []})
        for key in vals[0]:
            dict_csv.update({key : []})
        for index,val in enumerate(vals):
            for key in val:
                dict_csv[key].append((val[key])[0])
            dict_csv['Score'].append(losses[index])
        df = pd.DataFrame.from_dict(dict_csv, orient='columns')
        df = df.sort_values("Score",axis=0,ascending=True)
        df.to_csv(path_or_buf=("./optimization results/"+filename),sep=',', index_label='Index')

param_dist1 = {
    "A_lat": hp.uniform("A_lat",0.1,0.2),
    "k_lat": hp.uniform("k_lat",0.5,1),
    "A_abd": hp.uniform("A_abd",0.1,0.3),
    "A_flex": hp.uniform("A_flex",0.1,0.3),
    "t_lat": hp.uniform("t_lat",0.0,0.5),
    "t_lat_delay": hp.uniform("t_lat_delay",0.0,0.25),
    "t_abd": hp.uniform("t_abd",0.0,0.5),
    "t_flex": hp.uniform("t_flex",0.0,0.5),
    "delta_lat_1": hp.uniform("delta_lat_1",0.0,pi),
    "delta_lat_2": hp.uniform("delta_lat_2",0.0,pi),
    "delta_lat_3": hp.uniform("delta_lat_3",0.0,pi),
    "delta_abd": hp.uniform("delta_abd",0.0,pi)
}
csv_name1="trial_weak_03friction.csv"
#run_bayesian_optimization1(objective_function=objective_function1,param_dist=param_dist1,iterations=n_iter, filename=csv_name1)

if False:
    trial_params_dict = {
        'A_abd': 0.5160641349761493,
        'A_flex': 0.5699905310095197,
        'A_lat': 0.37343856127784986,
        'delta_abd': 3.0972197864513813,
        'delta_lat_1': 1.1918035584969797,
        'delta_lat_2': 1.296461810132655,
        'delta_lat_3': 2.690594077074573,
        'k_lat': 0.6364078952413598,
        't_abd': 0.01087000771822172,
        't_flex': 0.16850242486762096,
        't_lat': 0.368409259583114,
        't_lat_delay': 0.07711040413095935}

    trial_params_tuple = (0.11755002981847623,0.13747994087504625,0.17621970228297168,0.46071392468134925,0.8121952620219266,2.530824762115656,0.20501255772523033,0.6968850597414533,0.0009446896081053966,0.06659510463168704,0.023131850116563442,0.20277601134705836)
    for key,val in zip(trial_params_dict,trial_params_tuple):
        trial_params_dict[key] = val
    # print(trial_params_dict)

    loss = run_simulation1(
            *generate_parameters_array1(
                **trial_params_dict,
                t_stance = 0.5,
                n_lat=2, n_abd=2,
                k_bias_abd=0.5, k_bias_flex=0.5
            ),
            keep_in_check=True,soft_constraints=False,
            graphic_mode=True, plot_graph=True
        )
    print("loss: ",loss)

#######################################################################################
def run_simulation2(dt, t_stance, duration,
    A, f1, n, t1_off, delta_off, bias,
    theta_rg0, theta_lg0, A_lat_0, theta_lat_0,
    girdle_friction, body_friction, leg_friction,
    keep_in_check=True, soft_constraints=False,
    graphic_mode=False, plot_graph=False, video_logging=False, video_name="./video/dump/dump"):
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
    z_rotation = 0#pi/3.5 #radians
    model = crw.Crawler(urdf_path="/home/fra/Uni/Tesi/crawler", dt_simulation=dt, base_position=[0,0,0.05], base_orientation=[0,0,sin(z_rotation/2),cos(z_rotation/2)])
    # for i,elem in enumerate(model.links_state_array):
    #     print("Link %d  "%i, elem["world_com_trn"])
    # print(model.links_state_array)
    # Set motion damping ("air friction")
    for index in range(-1,model.num_joints):
        p.changeDynamics(
            model.Id,
            index,
            linearDamping=0.01,
            angularDamping=0.01
        )
    # Girdle link dynamic properties
    p.changeDynamics(model.Id,
        linkIndex = -1,
        lateralFriction = girdle_friction,
        spinningFriction = girdle_friction/10,
        rollingFriction = girdle_friction/20,
        restitution = 0.1,
        #contactStiffness = 0,
        #contactDamping = 0
        )
    # Body dynamic properties
    for i in range(0,model.num_joints-4):
        p.changeDynamics(model.Id,
            linkIndex = i,
            lateralFriction = body_friction,
            spinningFriction = body_friction/10,
            rollingFriction = body_friction/20,
            restitution = 0.1,
            #contactStiffness = 0,
            #contactDamping = 0
            )
    # Legs dynamic properties
    for i in range(model.num_joints-4,model.num_joints):
        p.changeDynamics(model.Id,
            linkIndex = i,
            lateralFriction = leg_friction,
            spinningFriction = leg_friction/10,
            rollingFriction = leg_friction/20,
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
    #model.set_bent_position(theta_rg=theta_rg0, theta_lg=theta_lg0,A_lat=A_lat_0,theta_lat_0=0.0)
    ##########################################
    ####### MISCELLANEOUS ####################
    if graphic_mode:
        p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
        p.resetDebugVisualizerCamera(cameraDistance=0.5, cameraYaw = 50, cameraPitch=-60, cameraTargetPosition=[0,0,0])
    if (video_logging and graphic_mode):
        video_writer = imageio.get_writer(video_name, fps=int(1/dt))
    ##########################################
    ####### PERFORMANCE EVALUATION ###########
    loss = 0.0
    max_loss = 10000000.0 # value to set when something goes wrong, if constraints are "hard"
    max_COM_height = 2*model.body_sphere_radius
    max_COM_speed = 0.8 #TODO: check this value
    n_mean = 3
    ##########################################
    ####### TIME SERIES CREATION #############
    steps = int(duration/dt)
    num_lat = len(model.control_indices[0])
    # Torques evolution
    tau_array_fun = model.get_torques_profile_fun(A,f1,n,t1_off,delta_off,bias)
    tau_time_array = model.generate_torques_time_array(tau_array_fun,duration,t0=0.0,include_base=True)
    # Data array initialization
    p_COM_time_array = np.zeros((steps,3))
    v_COM_time_array = np.zeros((steps,3))
    joint_power_time_array = np.zeros((steps,model.state.nqd))
    q_time_array = np.zeros((steps,model.state.nq))
    qd_time_array = np.zeros((steps,model.state.nqd))
    qdd_time_array = np.zeros((steps,model.state.nqd))
    # Integrator for the work done by each joint
    joint_power_integrator = crw.Integrator_Forward_Euler(dt,np.zeros(model.state.nqd))
    joint_energy_time_array = np.zeros((steps,model.state.nqd))
    ##########################################
    ####### RUN SIMULATION ###################
    t = 0.0
    # let the leg fall
    for i in range(30):
        p.stepSimulation()
        if graphic_mode:
            time.sleep(dt)
    # walk and record data
    for i in range(steps):
        model.state.update()
        q_time_array[i] = model.state.q.copy()
        qd_time_array[i] = model.state.qd.copy()
        p_COM_curr = np.array(model.COM_position_world())
        v_COM_curr = np.array(model.COM_velocity_world())
        qdd_time_array[i] = model.state.qdd.copy()
        p_COM_time_array[i]=p_COM_curr
        v_COM_time_array[i]=v_COM_curr
        for j, (v_j, tau_j) in enumerate(zip(model.state.qd,tau_time_array[i])):
            joint_power_time_array[i,j] = abs(tau_j)
        joint_energy = joint_power_integrator.integrate(joint_power_time_array[i])
        joint_energy_time_array[i] = joint_energy
        ### UPDATE LOSS
        # -(x-displacement) + (total energy expenditure)
        #loss = -(p_COM_time_array[i][0]-p_COM_time_array[0][0]) + (joint_energy_time_array[-1]).dot(joint_energy_time_array[-1])
        # VARIATIONAL VERSION -d(x-displacement) + d(total energy expenditure)
        loss += (
            -60*(p_COM_time_array[i][0]-p_COM_time_array[i-1][0]) + 
            (joint_power_time_array[i]).dot(joint_power_time_array[i])
            )
        # if soft_constraints:
        #     loss+= (np.exp((p_COM_curr[-1]/(max_COM_height))**2)-1.0 + 
        #         np.exp(((np.linalg.norm(np.mean(v_COM_time_array[i-n_mean:i+1], 0)))/max_COM_speed)**2)-1.0)
        ###
        if (graphic_mode and video_logging):
            img=p.getCameraImage(800, 640, renderer=p.ER_BULLET_HARDWARE_OPENGL)[2]
            video_writer.append_data(img)
        ###
        model.apply_torques(tau_time_array[i], filtered=False)
        p.stepSimulation()
        if graphic_mode:
            time.sleep(dt) 
        t += dt
        ### check simulation to avoid that the model get schwifty
        # can be substituted by soft constraints for the violation of the two conditions
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
    if plot_graph:
        fig_lat, axs_lat = plt.subplots(2,3)
        for i in range(0,num_lat):
            #axs_lat[i//3,i%3].plot(qd_time_array[:,model.mask_act][:,i], color="xkcd:teal", label="qd")
            #axs_lat[i//3,i%3].plot(qdd_time_array[:,model.mask_act][:,i], color="xkcd:light teal", label="qdd")
            axs_lat[i//3,i%3].plot(joint_power_time_array[:,model.mask_act][:,i], color="xkcd:salmon", label="Power")
            axs_lat[i//3,i%3].plot(joint_energy_time_array[:,model.mask_act][:,i], color="xkcd:light salmon", label="Energy")
            axs_lat[i//3,i%3].plot(tau_time_array[:,model.mask_act][:,i], color="xkcd:dark salmon", label="Torque")
            axs_lat[i//3,i%3].plot(q_time_array[:,model.mask_act_shifted][:,i], color="xkcd:dark teal", label="q")
            axs_lat[i//3,i%3].set_title("Lateral joint %d" %i)
        handles_lat, labels_lat = axs_lat[0,0].get_legend_handles_labels()
        #
        fig_lat.legend(handles_lat, labels_lat, loc='center left')
        fig_girdle, axs_girdle = plt.subplots(2,2)
        for i in range(0,len(model.mask_both_legs)):
            #axs_girdle[i//2,i%2].plot(qd_time_array[:,model.mask_both_legs][:,i], color="xkcd:teal", label="qd")
            #axs_girdle[i//2,i%2].plot(qdd_time_array[:,model.mask_both_legs][:,i], color="xkcd:light teal", label="qdd")
            axs_girdle[i//2,i%2].plot(joint_power_time_array[:,model.mask_both_legs][:,i], color="xkcd:salmon", label="Power")
            axs_girdle[i//2,i%2].plot(joint_energy_time_array[:,model.mask_both_legs][:,i], color="xkcd:light salmon", label="Energy")
            axs_girdle[i//2,i%2].plot(tau_time_array[:,model.mask_both_legs][:,i], color="xkcd:dark salmon", label="Torque yeah")
            axs_girdle[i//2,i%2].plot(q_time_array[:,model.mask_both_legs_shifted][:,i], color="xkcd:dark teal", label="q")
        axs_girdle[0,0].set_title("Right abduction")
        axs_girdle[0,1].set_title("Right flexion")
        axs_girdle[1,0].set_title("Left abduction")
        axs_girdle[1,1].set_title("Left flexion")
        handles_girdle, labels_girdle = axs_girdle[0,0].get_legend_handles_labels()
        fig_girdle.legend(handles_girdle, labels_girdle, loc='center left')
        #
        plt.show()
        #
        fig_COM = plt.figure()
        axs_COM = fig_COM.add_subplot(111)
        axs_COM.plot(p_COM_time_array[:,0], p_COM_time_array[:,1],color="xkcd:teal")
        axs_COM.set_title("COM x-y trajectory")
        fig_COM_3D = plt.figure()
        axs_COM_3D = fig_COM_3D.add_subplot(111, projection='3d')
        axs_COM_3D.plot(p_COM_time_array[:,0], p_COM_time_array[:,1], p_COM_time_array[:,2],color="xkcd:teal")
        axs_COM_3D.set_title("COM 3D trajectory")
        #
        plt.show()
    return loss

def generate_parameters_array2(t_stance,
                            A_lat, k_lat, A_abd, A_flex,
                            n_lat, n_abd,
                            t_lat, t_lat_delay, t_abd, t_flex,
                            delta_lat_1, delta_lat_2, delta_lat_3, delta_abd,
                            k_bias_abd, k_bias_flex):
    ### TIME VARIABLES
    dt = 1/240
    duration = 10
    f_walk = 1/(2*t_stance)
    ### TORQUES EVOLUTION VARIABLEs
    # A_lat*(2*k_lat) < 1, 1Nm is a lot already (tested)
    A = (
        (A_lat,
        A_lat*(1+k_lat),
        A_lat*(2+k_lat),
        A_lat*(1+k_lat),
        A_lat),
        (A_abd,A_flex),
        (A_abd,A_flex))
    f1 = ((f_walk,f_walk,f_walk,f_walk,f_walk),(f_walk,f_walk),(f_walk,f_walk))
    #TODO give the option to set n_lat=0 so that abduction can also be a pure cosine wave
    n = ((n_lat,n_lat,n_lat,n_lat,n_lat),(n_lat,0),(n_lat,0))
    # left leg is just as the right leg but translated by half a period
    t1_off = (
        (t_lat,
        t_lat + 1*t_lat_delay,
        t_lat + 2*t_lat_delay,
        t_lat + 3*t_lat_delay,
        t_lat + 4*t_lat_delay),
        (t_abd, t_flex),
        (t_abd+t_stance, t_flex+t_stance)
        )
    # For delta, considering symmetry on the middle spinal joint of the shape of the functions.
    # delta_abd = 0 and n_abd=0 to make the flexion a single sine wave.
    # if (int(n_lat) == 0):
    #     for delta_lat in (delta_lat_1,delta_lat_2,delta_lat_3):
    #         delta_lat = pi/2
    # if (int(n_abd) == 0):
    #     delta_abd = pi/2
    delta_off = (
        (delta_lat_1,
        delta_lat_2,
        delta_lat_3,
        delta_lat_2,
        delta_lat_1
        ),
        (delta_abd, 0),
        (delta_abd, 0))
    # for correct bias on flexion and abduction check how reference system are oriented
    bias = ((0,0,0,0,0),(-k_bias_abd*A_abd,k_bias_flex*A_flex),(k_bias_abd*A_abd,-k_bias_flex*A_flex))
    ### STARTING POSITION VARIABLES
    theta_rg0=pi/12
    theta_lg0=-pi/4
    A_lat_0=-pi/3.5
    theta_lat_0=0.0
    #
    girdle_friction = 0.2
    body_friction = 0.2
    leg_friction = 0.2
    return (dt, t_stance, duration,
    A, f1, n, t1_off, delta_off, bias,
    theta_rg0, theta_lg0, A_lat_0, theta_lat_0,
    girdle_friction, body_friction, leg_friction)

def objective_function2(params):
    #value not contained in params must be passed manually, as below
    loss = run_simulation2(
        *generate_parameters_array2(
            **params,
            t_stance = 0.5,
            n_lat=2, n_abd=2,
            k_bias_abd=0.7, k_bias_flex=0.7
        ),
        keep_in_check=True,soft_constraints=False,
        graphic_mode=False, plot_graph=False
    )
    return loss

def run_bayesian_optimization2(objective_function,param_dist,iterations,filename="trial.csv",save=True):
    trials = Trials()
    seed=int(time.time())
    best_param = fmin(
        objective_function, 
        param_dist, 
        algo=tpe.suggest, 
        max_evals=iterations, 
        trials=trials,
        rstate=np.random.RandomState(seed)
        )
    losses = [x['result']['loss'] for x in trials.trials]
    vals = [x['misc']['vals']for x in trials.trials]
    best_param_values = [val for val in best_param.values()]
    best_param_values = [x for x in best_param.values()]
    print("Best loss obtained: %f\n with parameters: %s" % (min(losses), best_param_values))
    if save:
        dict_csv={}
        dict_csv.update({'Score' : []})
        for key in vals[0]:
            dict_csv.update({key : []})
        for index,val in enumerate(vals):
            for key in val:
                dict_csv[key].append((val[key])[0])
            dict_csv['Score'].append(losses[index])
        df = pd.DataFrame.from_dict(dict_csv, orient='columns')
        df = df.sort_values("Score",axis=0,ascending=True)
        df.to_csv(path_or_buf=("./optimization results/"+filename),sep=',', index_label='Index')

param_dist2 = {
    "A_lat": hp.uniform("A_lat",0.1,0.2),
    "k_lat": hp.uniform("k_lat",0.5,1),
    "A_abd": hp.uniform("A_abd",0.1,0.3),
    "A_flex": hp.uniform("A_flex",0.1,0.3),
    "t_lat": hp.uniform("t_lat",0.0,0.5),
    "t_lat_delay": hp.uniform("t_lat_delay",0.0,0.25),
    "t_abd": hp.uniform("t_abd",0.0,0.5),
    "t_flex": hp.uniform("t_flex",0.0,0.5),
    "delta_lat_1": hp.uniform("delta_lat_1",0.0,pi),
    "delta_lat_2": hp.uniform("delta_lat_2",0.0,pi),
    "delta_lat_3": hp.uniform("delta_lat_3",0.0,pi),
    "delta_abd": hp.uniform("delta_abd",0.0,pi)
}
csv_name2="weak_nowoosh_02friction_07bias_.csv"
#run_bayesian_optimization2(objective_function=objective_function2,param_dist=param_dist2,iterations=n_iter, filename=csv_name2)
if False:
    trial_params_dict = {
        'A_abd': 0.5160641349761493,
        'A_flex': 0.5699905310095197,
        'A_lat': 0.37343856127784986,
        'delta_abd': 3.0972197864513813,
        'delta_lat_1': 1.1918035584969797,
        'delta_lat_2': 1.296461810132655,
        'delta_lat_3': 2.690594077074573,
        'k_lat': 0.6364078952413598,
        't_abd': 0.01087000771822172,
        't_flex': 0.16850242486762096,
        't_lat': 0.368409259583114,
        't_lat_delay': 0.07711040413095935}

    trial_params_tuple = (0.11020485251379682,0.10302738528403088,0.10012184566618046,0.384329648795947,1.9947470549570216,0.7679059501306296,2.0799663672151056,0.5275772672689625,0.41887083297047223,0.39866585544264355,0.10110978697940828,0.20319390439165222)
    for key,val in zip(trial_params_dict,trial_params_tuple):
        trial_params_dict[key] = val
    # print(trial_params_dict)

    loss = run_simulation2(
            *generate_parameters_array2(
                **trial_params_dict,
                t_stance = 0.5,
                n_lat=2, n_abd=2,
                k_bias_abd=0.5, k_bias_flex=0.5
            ),
            keep_in_check=True,soft_constraints=False,
            graphic_mode=True, plot_graph=False
        )
    print("loss: ",loss)
###########################################################
def run_simulation3(dt, t_stance, duration,
    A, f1, n, t1_off, delta_off, bias,
    theta_rg0, theta_lg0, A_lat_0, theta_lat_0,
    girdle_friction, body_friction, leg_friction,
    keep_in_check=True, soft_constraints=False,
    graphic_mode=False, plot_graph=False, video_logging=False, video_name="./video/dump/dump"):
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
    z_rotation = 0#pi/3.5 #radians
    model = crw.Crawler(urdf_path="/home/fra/Uni/Tesi/crawler", dt_simulation=dt, base_position=[0,0,0.05], base_orientation=[0,0,sin(z_rotation/2),cos(z_rotation/2)])
    # for i,elem in enumerate(model.links_state_array):
    #     print("Link %d  "%i, elem["world_com_trn"])
    # print(model.links_state_array)
    # Set motion damping ("air friction")
    for index in range(-1,model.num_joints):
        p.changeDynamics(
            model.Id,
            index,
            linearDamping=0.01,
            angularDamping=0.01
        )
    # Girdle link dynamic properties
    p.changeDynamics(model.Id,
        linkIndex = -1,
        lateralFriction = girdle_friction,
        spinningFriction = girdle_friction/10,
        rollingFriction = girdle_friction/20,
        restitution = 0.1,
        #contactStiffness = 0,
        #contactDamping = 0
        )
    # Body dynamic properties
    for i in range(0,model.num_joints-4):
        p.changeDynamics(model.Id,
            linkIndex = i,
            lateralFriction = body_friction,
            spinningFriction = body_friction/10,
            rollingFriction = body_friction/20,
            restitution = 0.1,
            #contactStiffness = 0,
            #contactDamping = 0
            )
    # Legs dynamic properties
    for i in range(model.num_joints-4,model.num_joints):
        p.changeDynamics(model.Id,
            linkIndex = i,
            lateralFriction = leg_friction,
            spinningFriction = leg_friction/10,
            rollingFriction = leg_friction/20,
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
    #model.set_bent_position(theta_rg=theta_rg0, theta_lg=theta_lg0,A_lat=A_lat_0,theta_lat_0=0.0)
    ##########################################
    ####### MISCELLANEOUS ####################
    if graphic_mode:
        p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
        p.resetDebugVisualizerCamera(cameraDistance=0.5, cameraYaw = 50, cameraPitch=-60, cameraTargetPosition=[0,0,0])
    if (video_logging and graphic_mode):
        video_writer = imageio.get_writer(video_name, fps=int(1/dt))
    ##########################################
    ####### PERFORMANCE EVALUATION ###########
    loss = 0.0
    max_loss = 10000000.0 # value to set when something goes wrong, if constraints are "hard"
    max_COM_height = 2*model.body_sphere_radius
    max_COM_speed = 0.8 #TODO: check this value
    n_mean = 3
    ##########################################
    ####### TIME SERIES CREATION #############
    steps = int(duration/dt)
    num_lat = len(model.control_indices[0])
    # Torques evolution
    tau_array_fun = model.get_torques_profile_fun(A,f1,n,t1_off,delta_off,bias)
    tau_time_array = model.generate_torques_time_array(tau_array_fun,duration,t0=0.0,include_base=True)
    # Data array initialization
    p_COM_time_array = np.zeros((steps,3))
    v_COM_time_array = np.zeros((steps,3))
    joint_power_time_array = np.zeros((steps,model.state.nqd))
    q_time_array = np.zeros((steps,model.state.nq))
    qd_time_array = np.zeros((steps,model.state.nqd))
    qdd_time_array = np.zeros((steps,model.state.nqd))
    # Integrator for the work done by each joint
    joint_power_integrator = crw.Integrator_Forward_Euler(dt,np.zeros(model.state.nqd))
    joint_energy_time_array = np.zeros((steps,model.state.nqd))
    ##########################################
    ####### RUN SIMULATION ###################
    t = 0.0
    # let the leg fall
    for i in range(30):
        p.stepSimulation()
        if graphic_mode:
            time.sleep(dt)
    # walk and record data
    for i in range(steps):
        model.state.update()
        q_time_array[i] = model.state.q.copy()
        qd_time_array[i] = model.state.qd.copy()
        p_COM_curr = np.array(model.COM_position_world())
        v_COM_curr = np.array(model.COM_velocity_world())
        qdd_time_array[i] = model.state.qdd.copy()
        p_COM_time_array[i]=p_COM_curr
        v_COM_time_array[i]=v_COM_curr
        for j, (v_j, tau_j) in enumerate(zip(model.state.qd,tau_time_array[i])):
            joint_power_time_array[i,j] = abs((1+v_j)*tau_j)
        joint_energy = joint_power_integrator.integrate(joint_power_time_array[i])
        joint_energy_time_array[i] = joint_energy
        ### UPDATE LOSS
        # -(x-displacement) + (total energy expenditure)
        loss += (
            -100*(p_COM_time_array[i][0]-p_COM_time_array[i-1][0]) + 
            (joint_power_time_array[i]).dot(joint_power_time_array[i]) +
            50*(abs(p_COM_time_array[i][1]-p_COM_time_array[i-1][1]))
            )
        # VARIATIONAL VERSION -d(x-displacement) + d(total energy expenditure)
        #loss += -(p_COM_time_array[i][0]-p_COM_time_array[i-1][0]) + (joint_power_time_array[i]).dot(joint_power_time_array[i])
        # if soft_constraints:
        #     loss+= (np.exp((p_COM_curr[-1]/(max_COM_height))**2)-1.0 + 
        #         np.exp(((np.linalg.norm(np.mean(v_COM_time_array[i-n_mean:i+1], 0)))/max_COM_speed)**2)-1.0)
        ###
        if (graphic_mode and video_logging):
            img=p.getCameraImage(800, 640, renderer=p.ER_BULLET_HARDWARE_OPENGL)[2]
            video_writer.append_data(img)
        ###
        model.apply_torques(tau_time_array[i], filtered=False)
        p.stepSimulation()
        if graphic_mode:
            time.sleep(dt) 
        t += dt
        ### check simulation to avoid that the model get schwifty
        # can be substituted by soft constraints for the violation of the two conditions
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
    if plot_graph:
        fig_lat, axs_lat = plt.subplots(2,3)
        for i in range(0,num_lat):
            #axs_lat[i//3,i%3].plot(qd_time_array[:,model.mask_act][:,i], color="xkcd:teal", label="qd")
            #axs_lat[i//3,i%3].plot(qdd_time_array[:,model.mask_act][:,i], color="xkcd:light teal", label="qdd")
            axs_lat[i//3,i%3].plot(joint_power_time_array[:,model.mask_act][:,i], color="xkcd:salmon", label="Power")
            axs_lat[i//3,i%3].plot(joint_energy_time_array[:,model.mask_act][:,i], color="xkcd:light salmon", label="Energy")
            axs_lat[i//3,i%3].plot(tau_time_array[:,model.mask_act][:,i], color="xkcd:dark salmon", label="Torque")
            axs_lat[i//3,i%3].plot(q_time_array[:,model.mask_act_shifted][:,i], color="xkcd:dark teal", label="q")
            axs_lat[i//3,i%3].set_title("Lateral joint %d" %i)
        handles_lat, labels_lat = axs_lat[0,0].get_legend_handles_labels()
        #
        fig_lat.legend(handles_lat, labels_lat, loc='center left')
        fig_girdle, axs_girdle = plt.subplots(2,2)
        for i in range(0,len(model.mask_both_legs)):
            #axs_girdle[i//2,i%2].plot(qd_time_array[:,model.mask_both_legs][:,i], color="xkcd:teal", label="qd")
            #axs_girdle[i//2,i%2].plot(qdd_time_array[:,model.mask_both_legs][:,i], color="xkcd:light teal", label="qdd")
            axs_girdle[i//2,i%2].plot(joint_power_time_array[:,model.mask_both_legs][:,i], color="xkcd:salmon", label="Power")
            axs_girdle[i//2,i%2].plot(joint_energy_time_array[:,model.mask_both_legs][:,i], color="xkcd:light salmon", label="Energy")
            axs_girdle[i//2,i%2].plot(tau_time_array[:,model.mask_both_legs][:,i], color="xkcd:dark salmon", label="Torque yeah")
            axs_girdle[i//2,i%2].plot(q_time_array[:,model.mask_both_legs_shifted][:,i], color="xkcd:dark teal", label="q")
        axs_girdle[0,0].set_title("Right abduction")
        axs_girdle[0,1].set_title("Right flexion")
        axs_girdle[1,0].set_title("Left abduction")
        axs_girdle[1,1].set_title("Left flexion")
        handles_girdle, labels_girdle = axs_girdle[0,0].get_legend_handles_labels()
        fig_girdle.legend(handles_girdle, labels_girdle, loc='center left')
        #
        plt.show()
        #
        fig_COM = plt.figure()
        axs_COM = fig_COM.add_subplot(111)
        axs_COM.plot(p_COM_time_array[:,0], p_COM_time_array[:,1],color="xkcd:teal")
        axs_COM.set_title("COM x-y trajectory")
        fig_COM_3D = plt.figure()
        axs_COM_3D = fig_COM_3D.add_subplot(111, projection='3d')
        axs_COM_3D.plot(p_COM_time_array[:,0], p_COM_time_array[:,1], p_COM_time_array[:,2],color="xkcd:teal")
        axs_COM_3D.set_title("COM 3D trajectory")
        #
        plt.show()
    return loss

def generate_parameters_array3(t_stance,
                            A_lat, k_lat, A_abd, A_flex,
                            n_lat, n_abd,
                            t_lat, t_lat_delay, t_abd, t_flex,
                            delta_lat_1, delta_lat_2, delta_lat_3, delta_abd,
                            k_bias_abd, k_bias_flex):
    ### TIME VARIABLES
    dt = 1/240
    duration = 10
    f_walk = 1/(2*t_stance)
    ### TORQUES EVOLUTION VARIABLEs
    # A_lat*(2*k_lat) < 1, 1Nm is a lot already (tested)
    A = (
        (A_lat,
        A_lat*(1+k_lat),
        A_lat*(2+k_lat),
        A_lat*(1+k_lat),
        A_lat),
        (A_abd,A_flex),
        (A_abd,A_flex))
    f1 = ((f_walk,f_walk,f_walk,f_walk,f_walk),(f_walk,f_walk),(f_walk,f_walk))
    #TODO give the option to set n_lat=0 so that abduction can also be a pure cosine wave
    n = ((n_lat,n_lat,n_lat,n_lat,n_lat),(n_lat,0),(n_lat,0))
    # left leg is just as the right leg but translated by half a period
    t1_off = (
        (t_lat,
        t_lat + 1*t_lat_delay,
        t_lat + 2*t_lat_delay,
        t_lat + 3*t_lat_delay,
        t_lat + 4*t_lat_delay),
        (t_abd, t_flex),
        (t_abd+t_stance, t_flex+t_stance)
        )
    # For delta, considering symmetry on the middle spinal joint of the shape of the functions.
    # delta_abd = 0 and n_abd=0 to make the flexion a single sine wave.
    # if (int(n_lat) == 0):
    #     for delta_lat in (delta_lat_1,delta_lat_2,delta_lat_3):
    #         delta_lat = pi/2
    # if (int(n_abd) == 0):
    #     delta_abd = pi/2
    delta_off = (
        (delta_lat_1,
        delta_lat_2,
        delta_lat_3,
        delta_lat_2,
        delta_lat_1
        ),
        (delta_abd, 0),
        (delta_abd, 0))
    # for correct bias on flexion and abduction check how reference system are oriented
    bias = ((0,0,0,0,0),(-k_bias_abd*A_abd,k_bias_flex*A_flex),(k_bias_abd*A_abd,-k_bias_flex*A_flex))
    ### STARTING POSITION VARIABLES
    theta_rg0=pi/12
    theta_lg0=-pi/4
    A_lat_0=-pi/3.5
    theta_lat_0=0.0
    #
    girdle_friction = 0.2
    body_friction = 0.2
    leg_friction = 0.2
    return (dt, t_stance, duration,
    A, f1, n, t1_off, delta_off, bias,
    theta_rg0, theta_lg0, A_lat_0, theta_lat_0,
    girdle_friction, body_friction, leg_friction)

def objective_function3(params):
    #value not contained in params must be passed manually, as below
    loss = run_simulation3(
        *generate_parameters_array3(
            **params,
            t_stance = 0.5,
            n_lat=2, n_abd=2,
            k_bias_abd=0.5, k_bias_flex=0.5
        ),
        keep_in_check=True,soft_constraints=False,
        graphic_mode=False, plot_graph=False
    )
    return loss

def run_bayesian_optimization3(objective_function,param_dist,iterations,filename="trial.csv",save=True):
    trials = Trials()
    seed=int(time.time())
    best_param = fmin(
        objective_function, 
        param_dist, 
        algo=tpe.suggest, 
        max_evals=iterations, 
        trials=trials,
        rstate=np.random.RandomState(seed)
        )
    losses = [x['result']['loss'] for x in trials.trials]
    vals = [x['misc']['vals']for x in trials.trials]
    best_param_values = [val for val in best_param.values()]
    best_param_values = [x for x in best_param.values()]
    print("Best loss obtained: %f\n with parameters: %s" % (min(losses), best_param_values))
    if save:
        dict_csv={}
        dict_csv.update({'Score' : []})
        for key in vals[0]:
            dict_csv.update({key : []})
        for index,val in enumerate(vals):
            for key in val:
                dict_csv[key].append((val[key])[0])
            dict_csv['Score'].append(losses[index])
        df = pd.DataFrame.from_dict(dict_csv, orient='columns')
        df = df.sort_values("Score",axis=0,ascending=True)
        df.to_csv(path_or_buf=("./optimization results/"+filename),sep=',', index_label='Index')

param_dist3 = {
    "A_lat": hp.uniform("A_lat",0.1,0.2),
    "k_lat": hp.uniform("k_lat",0.5,1),
    "A_abd": hp.uniform("A_abd",0.1,0.3),
    "A_flex": hp.uniform("A_flex",0.1,0.3),
    "t_lat": hp.uniform("t_lat",0.0,0.5),
    "t_lat_delay": hp.uniform("t_lat_delay",0.0,0.25),
    "t_abd": hp.uniform("t_abd",0.0,0.5),
    "t_flex": hp.uniform("t_flex",0.0,0.5),
    "delta_lat_1": hp.uniform("delta_lat_1",0.0,pi),
    "delta_lat_2": hp.uniform("delta_lat_2",0.0,pi),
    "delta_lat_3": hp.uniform("delta_lat_3",0.0,pi),
    "delta_abd": hp.uniform("delta_abd",0.0,pi)
}
csv_name3="weak_nowoosh_02friction_ycost.csv"
#run_bayesian_optimization3(objective_function=objective_function3,param_dist=param_dist3,iterations=n_iter, filename=csv_name3)
if False:
    trial_params_dict = {
        'A_abd': 0.5160641349761493,
        'A_flex': 0.5699905310095197,
        'A_lat': 0.37343856127784986,
        'delta_abd': 3.0972197864513813,
        'delta_lat_1': 1.1918035584969797,
        'delta_lat_2': 1.296461810132655,
        'delta_lat_3': 2.690594077074573,
        'k_lat': 0.6364078952413598,
        't_abd': 0.01087000771822172,
        't_flex': 0.16850242486762096,
        't_lat': 0.368409259583114,
        't_lat_delay': 0.07711040413095935}

    trial_params_tuple = (0.10016887504633509,0.10005585073474774,0.10202288161987762,0.22084156049283366,2.534041304958117,0.8586996683901065,1.4828843351146381,0.5299849708669773,0.39671388294469473,0.4534352993028361,0.3598483752693789,0.22239688879572433)
    for key,val in zip(trial_params_dict,trial_params_tuple):
        trial_params_dict[key] = val
    # print(trial_params_dict)

    loss = run_simulation3(
            *generate_parameters_array3(
                **trial_params_dict,
                t_stance = 0.5,
                n_lat=2, n_abd=2,
                k_bias_abd=0.5, k_bias_flex=0.5
            ),
            keep_in_check=True,soft_constraints=False,
            graphic_mode=True, plot_graph=True
        )
    print("loss: ",loss)
#################################################
def run_simulation4(dt, t_stance, duration,
    A, f1, n, t1_off, delta_off, bias,
    theta_rg0, theta_lg0, A_lat_0, theta_lat_0,
    girdle_friction, body_friction, leg_friction,
    keep_in_check=True, soft_constraints=False,
    graphic_mode=False, plot_graph=False, video_logging=False, video_name="./video/dump/dump"):
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
    z_rotation = 0#pi/3.5 #radians
    model = crw.Crawler(urdf_path="/home/fra/Uni/Tesi/crawler", dt_simulation=dt, base_position=[0,0,0.05], base_orientation=[0,0,sin(z_rotation/2),cos(z_rotation/2)])
    # for i,elem in enumerate(model.links_state_array):
    #     print("Link %d  "%i, elem["world_com_trn"])
    # print(model.links_state_array)
    # Set motion damping ("air friction")
    for index in range(-1,model.num_joints):
        p.changeDynamics(
            model.Id,
            index,
            linearDamping=0.01,
            angularDamping=0.01
        )
    # Girdle link dynamic properties
    p.changeDynamics(model.Id,
        linkIndex = -1,
        lateralFriction = girdle_friction,
        spinningFriction = girdle_friction/10,
        rollingFriction = girdle_friction/20,
        restitution = 0.1,
        #contactStiffness = 0,
        #contactDamping = 0
        )
    # Body dynamic properties
    for i in range(0,model.num_joints-4):
        p.changeDynamics(model.Id,
            linkIndex = i,
            lateralFriction = body_friction,
            spinningFriction = body_friction/10,
            rollingFriction = body_friction/20,
            restitution = 0.1,
            #contactStiffness = 0,
            #contactDamping = 0
            )
    # Legs dynamic properties
    for i in range(model.num_joints-4,model.num_joints):
        p.changeDynamics(model.Id,
            linkIndex = i,
            lateralFriction = leg_friction,
            spinningFriction = leg_friction/10,
            rollingFriction = leg_friction/20,
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
    #model.set_bent_position(theta_rg=theta_rg0, theta_lg=theta_lg0,A_lat=A_lat_0,theta_lat_0=0.0)
    ##########################################
    ####### MISCELLANEOUS ####################
    if graphic_mode:
        p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
        p.resetDebugVisualizerCamera(cameraDistance=0.5, cameraYaw = 50, cameraPitch=-60, cameraTargetPosition=[0,0,0])
    if (video_logging and graphic_mode):
        video_writer = imageio.get_writer(video_name, fps=int(1/dt))
    ##########################################
    ####### PERFORMANCE EVALUATION ###########
    loss = 0.0
    max_loss = 10000000.0 # value to set when something goes wrong, if constraints are "hard"
    max_COM_height = 2*model.body_sphere_radius
    max_COM_speed = 0.8 #TODO: check this value
    n_mean = 3
    ##########################################
    ####### TIME SERIES CREATION #############
    steps = int(duration/dt)
    num_lat = len(model.control_indices[0])
    # Torques evolution
    tau_array_fun = model.get_torques_profile_fun(A,f1,n,t1_off,delta_off,bias)
    tau_time_array = model.generate_torques_time_array(tau_array_fun,duration,t0=0.0,include_base=True)
    # Data array initialization
    p_COM_time_array = np.zeros((steps,3))
    v_COM_time_array = np.zeros((steps,3))
    joint_power_time_array = np.zeros((steps,model.state.nqd))
    q_time_array = np.zeros((steps,model.state.nq))
    qd_time_array = np.zeros((steps,model.state.nqd))
    qdd_time_array = np.zeros((steps,model.state.nqd))
    # Integrator for the work done by each joint
    joint_power_integrator = crw.Integrator_Forward_Euler(dt,np.zeros(model.state.nqd))
    joint_energy_time_array = np.zeros((steps,model.state.nqd))
    ##########################################
    ####### RUN SIMULATION ###################
    t = 0.0
    # let the leg fall
    for i in range(30):
        p.stepSimulation()
        if graphic_mode:
            time.sleep(dt)
    # walk and record data
    for i in range(steps):
        model.state.update()
        q_time_array[i] = model.state.q.copy()
        qd_time_array[i] = model.state.qd.copy()
        p_COM_curr = np.array(model.COM_position_world())
        v_COM_curr = np.array(model.COM_velocity_world())
        qdd_time_array[i] = model.state.qdd.copy()
        p_COM_time_array[i]=p_COM_curr
        v_COM_time_array[i]=v_COM_curr
        for j, (v_j, tau_j) in enumerate(zip(model.state.qd,tau_time_array[i])):
            joint_power_time_array[i,j] = abs((1+v_j)*tau_j)
        joint_energy = joint_power_integrator.integrate(joint_power_time_array[i])
        joint_energy_time_array[i] = joint_energy
        ### UPDATE LOSS
        # -(x-displacement) + (total energy expenditure)
        #loss = -(p_COM_time_array[i][0]-p_COM_time_array[0][0]) + (joint_energy_time_array[-1]).dot(joint_energy_time_array[-1])
        # VARIATIONAL VERSION -d(x-displacement) + d(total energy expenditure)
        loss += (
            -100*(p_COM_time_array[i][0]-p_COM_time_array[i-1][0]) + 
            3*(joint_power_time_array[i]).dot(joint_power_time_array[i])
            )
        # if soft_constraints:
        #     loss+= (np.exp((p_COM_curr[-1]/(max_COM_height))**2)-1.0 + 
        #         np.exp(((np.linalg.norm(np.mean(v_COM_time_array[i-n_mean:i+1], 0)))/max_COM_speed)**2)-1.0)
        ###
        if (graphic_mode and video_logging):
            img=p.getCameraImage(800, 640, renderer=p.ER_BULLET_HARDWARE_OPENGL)[2]
            video_writer.append_data(img)
        ###
        model.apply_torques(tau_time_array[i], filtered=False)
        p.stepSimulation()
        if graphic_mode:
            time.sleep(dt) 
        t += dt
        ### check simulation to avoid that the model get schwifty
        # can be substituted by soft constraints for the violation of the two conditions
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
    if plot_graph:
        fig_lat, axs_lat = plt.subplots(2,3)
        for i in range(0,num_lat):
            #axs_lat[i//3,i%3].plot(qd_time_array[:,model.mask_act][:,i], color="xkcd:teal", label="qd")
            #axs_lat[i//3,i%3].plot(qdd_time_array[:,model.mask_act][:,i], color="xkcd:light teal", label="qdd")
            axs_lat[i//3,i%3].plot(joint_power_time_array[:,model.mask_act][:,i], color="xkcd:salmon", label="Power")
            axs_lat[i//3,i%3].plot(joint_energy_time_array[:,model.mask_act][:,i], color="xkcd:light salmon", label="Energy")
            axs_lat[i//3,i%3].plot(tau_time_array[:,model.mask_act][:,i], color="xkcd:dark salmon", label="Torque")
            axs_lat[i//3,i%3].plot(q_time_array[:,model.mask_act_shifted][:,i], color="xkcd:dark teal", label="q")
            axs_lat[i//3,i%3].set_title("Lateral joint %d" %i)
        handles_lat, labels_lat = axs_lat[0,0].get_legend_handles_labels()
        #
        fig_lat.legend(handles_lat, labels_lat, loc='center left')
        fig_girdle, axs_girdle = plt.subplots(2,2)
        for i in range(0,len(model.mask_both_legs)):
            #axs_girdle[i//2,i%2].plot(qd_time_array[:,model.mask_both_legs][:,i], color="xkcd:teal", label="qd")
            #axs_girdle[i//2,i%2].plot(qdd_time_array[:,model.mask_both_legs][:,i], color="xkcd:light teal", label="qdd")
            axs_girdle[i//2,i%2].plot(joint_power_time_array[:,model.mask_both_legs][:,i], color="xkcd:salmon", label="Power")
            axs_girdle[i//2,i%2].plot(joint_energy_time_array[:,model.mask_both_legs][:,i], color="xkcd:light salmon", label="Energy")
            axs_girdle[i//2,i%2].plot(tau_time_array[:,model.mask_both_legs][:,i], color="xkcd:dark salmon", label="Torque yeah")
            axs_girdle[i//2,i%2].plot(q_time_array[:,model.mask_both_legs_shifted][:,i], color="xkcd:dark teal", label="q")
        axs_girdle[0,0].set_title("Right abduction")
        axs_girdle[0,1].set_title("Right flexion")
        axs_girdle[1,0].set_title("Left abduction")
        axs_girdle[1,1].set_title("Left flexion")
        handles_girdle, labels_girdle = axs_girdle[0,0].get_legend_handles_labels()
        fig_girdle.legend(handles_girdle, labels_girdle, loc='center left')
        #
        plt.show()
        #
        fig_COM = plt.figure()
        axs_COM = fig_COM.add_subplot(111)
        axs_COM.plot(p_COM_time_array[:,0], p_COM_time_array[:,1],color="xkcd:teal")
        axs_COM.set_title("COM x-y trajectory")
        fig_COM_3D = plt.figure()
        axs_COM_3D = fig_COM_3D.add_subplot(111, projection='3d')
        axs_COM_3D.plot(p_COM_time_array[:,0], p_COM_time_array[:,1], p_COM_time_array[:,2],color="xkcd:teal")
        axs_COM_3D.set_title("COM 3D trajectory")
        #
        plt.show()
    return loss

def generate_parameters_array4(t_stance,
                            A_lat, k_lat, A_abd, A_flex,
                            n_lat, n_abd,
                            t_lat, t_lat_delay, t_abd, t_flex,
                            delta_lat_1, delta_lat_2, delta_lat_3, delta_abd,
                            k_bias_abd, k_bias_flex):
    ### TIME VARIABLES
    dt = 1/240
    duration = 10
    f_walk = 1/(2*t_stance)
    ### TORQUES EVOLUTION VARIABLEs
    # A_lat*(2*k_lat) < 1, 1Nm is a lot already (tested)
    A = (
        (A_lat,
        A_lat*(1+k_lat),
        A_lat*(2+k_lat),
        A_lat*(1+k_lat),
        A_lat),
        (A_abd,A_flex),
        (A_abd,A_flex))
    f1 = ((f_walk,f_walk,f_walk,f_walk,f_walk),(f_walk,f_walk),(f_walk,f_walk))
    #TODO give the option to set n_lat=0 so that abduction can also be a pure cosine wave
    n = ((n_lat,n_lat,n_lat,n_lat,n_lat),(n_lat,0),(n_lat,0))
    # left leg is just as the right leg but translated by half a period
    t1_off = (
        (t_lat,
        t_lat + 1*t_lat_delay,
        t_lat + 2*t_lat_delay,
        t_lat + 3*t_lat_delay,
        t_lat + 4*t_lat_delay),
        (t_abd, t_flex),
        (t_abd+t_stance, t_flex+t_stance)
        )
    # For delta, considering symmetry on the middle spinal joint of the shape of the functions.
    # delta_abd = 0 and n_abd=0 to make the flexion a single sine wave.
    # if (int(n_lat) == 0):
    #     for delta_lat in (delta_lat_1,delta_lat_2,delta_lat_3):
    #         delta_lat = pi/2
    # if (int(n_abd) == 0):
    #     delta_abd = pi/2
    delta_off = (
        (delta_lat_1,
        delta_lat_2,
        delta_lat_3,
        delta_lat_2,
        delta_lat_1
        ),
        (delta_abd, 0),
        (delta_abd, 0))
    # for correct bias on flexion and abduction check how reference system are oriented
    bias = ((0,0,0,0,0),(-k_bias_abd*A_abd,k_bias_flex*A_flex),(k_bias_abd*A_abd,-k_bias_flex*A_flex))
    ### STARTING POSITION VARIABLES
    theta_rg0=pi/12
    theta_lg0=-pi/4
    A_lat_0=-pi/3.5
    theta_lat_0=0.0
    #
    girdle_friction = 0.2
    body_friction = 0.2
    leg_friction = 0.2
    return (dt, t_stance, duration,
    A, f1, n, t1_off, delta_off, bias,
    theta_rg0, theta_lg0, A_lat_0, theta_lat_0,
    girdle_friction, body_friction, leg_friction)

def objective_function4(params):
    #value not contained in params must be passed manually, as below
    loss = run_simulation4(
        *generate_parameters_array4(
            **params,
            t_stance = 0.5,
            n_lat=2, n_abd=2,
            k_bias_abd=0.5, k_bias_flex=0.5
        ),
        keep_in_check=True,soft_constraints=False,
        graphic_mode=False, plot_graph=False
    )
    return loss

def run_bayesian_optimization4(objective_function,param_dist,iterations,filename="trial.csv",save=True):
    trials = Trials()
    seed=int(time.time())
    best_param = fmin(
        objective_function, 
        param_dist, 
        algo=tpe.suggest, 
        max_evals=iterations, 
        trials=trials,
        rstate=np.random.RandomState(seed)
        )
    losses = [x['result']['loss'] for x in trials.trials]
    vals = [x['misc']['vals']for x in trials.trials]
    best_param_values = [val for val in best_param.values()]
    best_param_values = [x for x in best_param.values()]
    print("Best loss obtained: %f\n with parameters: %s" % (min(losses), best_param_values))
    if save:
        dict_csv={}
        dict_csv.update({'Score' : []})
        for key in vals[0]:
            dict_csv.update({key : []})
        for index,val in enumerate(vals):
            for key in val:
                dict_csv[key].append((val[key])[0])
            dict_csv['Score'].append(losses[index])
        df = pd.DataFrame.from_dict(dict_csv, orient='columns')
        df = df.sort_values("Score",axis=0,ascending=True)
        df.to_csv(path_or_buf=("./optimization results/"+filename),sep=',', index_label='Index')

param_dist4 = {
    "A_lat": hp.uniform("A_lat",0.1,0.2),
    "k_lat": hp.uniform("k_lat",0.5,1),
    "A_abd": hp.uniform("A_abd",0.1,0.3),
    "A_flex": hp.uniform("A_flex",0.1,0.3),
    "t_lat": hp.uniform("t_lat",0.0,0.5),
    "t_lat_delay": hp.uniform("t_lat_delay",0.0,0.25),
    "t_abd": hp.uniform("t_abd",0.0,0.5),
    "t_flex": hp.uniform("t_flex",0.0,0.5),
    "delta_lat_1": hp.uniform("delta_lat_1",0.0,pi),
    "delta_lat_2": hp.uniform("delta_lat_2",0.0,pi),
    "delta_lat_3": hp.uniform("delta_lat_3",0.0,pi),
    "delta_abd": hp.uniform("delta_abd",0.0,pi)
}
csv_name4="weak_02friction_3xEnergycost_nowoosh.csv"
#run_bayesian_optimization4(objective_function=objective_function4,param_dist=param_dist4,iterations=n_iter, filename=csv_name4)
if False:
    trial_params_dict4 = {
        'A_abd': 0.5160641349761493,
        'A_flex': 0.5699905310095197,
        'A_lat': 0.37343856127784986,
        'delta_abd': 3.0972197864513813,
        'delta_lat_1': 1.1918035584969797,
        'delta_lat_2': 1.296461810132655,
        'delta_lat_3': 2.690594077074573,
        'k_lat': 0.6364078952413598,
        't_abd': 0.01087000771822172,
        't_flex': 0.16850242486762096,
        't_lat': 0.368409259583114,
        't_lat_delay': 0.07711040413095935}

    trial_params_tuple4 = (0.10041232511275597,0.12375104007354679,0.10359352716386758,0.4933294193778756,0.7485566707962767,0.8232171024556536,0.5206389198603718,0.6578597817117677,0.028681556252829938,0.3297961113239051,0.3877431240350188,0.011847865073260102)
    for key,val in zip(trial_params_dict4,trial_params_tuple4):
        trial_params_dict4[key] = val
    # print(trial_params_dict)

    loss = run_simulation4(
            *generate_parameters_array4(
                **trial_params_dict4,
                t_stance = 0.5,
                n_lat=2, n_abd=2,
                k_bias_abd=0.5, k_bias_flex=0.5
            ),
            keep_in_check=True,soft_constraints=False,
            graphic_mode=True, plot_graph=False
        )
    print("loss: ",loss)
#########################################
def run_simulation5(dt, t_stance, duration,
    A, f1, n, t1_off, delta_off, bias,
    theta_rg0, theta_lg0, A_lat_0, theta_lat_0,
    girdle_friction, body_friction, leg_friction,
    keep_in_check=True, soft_constraints=False,
    graphic_mode=False, plot_graph=False, video_logging=False, video_name="./video/dump/dump"):
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
    z_rotation = 0#pi/3.5 #radians
    model = crw.Crawler(urdf_path="/home/fra/Uni/Tesi/crawler", dt_simulation=dt, base_position=[0,0,0.05], base_orientation=[0,0,sin(z_rotation/2),cos(z_rotation/2)])
    # for i,elem in enumerate(model.links_state_array):
    #     print("Link %d  "%i, elem["world_com_trn"])
    # print(model.links_state_array)
    # Set motion damping ("air friction")
    for index in range(-1,model.num_joints):
        p.changeDynamics(
            model.Id,
            index,
            linearDamping=0.01,
            angularDamping=0.01
        )
    # Girdle link dynamic properties
    p.changeDynamics(model.Id,
        linkIndex = -1,
        lateralFriction = girdle_friction,
        spinningFriction = girdle_friction/10,
        rollingFriction = girdle_friction/20,
        restitution = 0.1,
        #contactStiffness = 0,
        #contactDamping = 0
        )
    # Body dynamic properties
    for i in range(0,model.num_joints-4):
        p.changeDynamics(model.Id,
            linkIndex = i,
            lateralFriction = body_friction,
            spinningFriction = body_friction/10,
            rollingFriction = body_friction/20,
            restitution = 0.1,
            #contactStiffness = 0,
            #contactDamping = 0
            )
    # Legs dynamic properties
    for i in range(model.num_joints-4,model.num_joints):
        p.changeDynamics(model.Id,
            linkIndex = i,
            lateralFriction = leg_friction,
            spinningFriction = leg_friction/10,
            rollingFriction = leg_friction/20,
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
    #model.set_bent_position(theta_rg=theta_rg0, theta_lg=theta_lg0,A_lat=A_lat_0,theta_lat_0=0.0)
    ##########################################
    ####### MISCELLANEOUS ####################
    if graphic_mode:
        p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
        p.resetDebugVisualizerCamera(cameraDistance=0.5, cameraYaw = 50, cameraPitch=-60, cameraTargetPosition=[0,0,0])
    if (video_logging and graphic_mode):
        video_writer = imageio.get_writer(video_name, fps=int(1/dt))
    ##########################################
    ####### PERFORMANCE EVALUATION ###########
    loss = 0.0
    max_loss = 10000000.0 # value to set when something goes wrong, if constraints are "hard"
    max_COM_height = 2*model.body_sphere_radius
    max_COM_speed = 0.8 #TODO: check this value
    n_mean = 3
    ##########################################
    ####### TIME SERIES CREATION #############
    steps = int(duration/dt)
    num_lat = len(model.control_indices[0])
    # Torques evolution
    tau_array_fun = model.get_torques_profile_fun(A,f1,n,t1_off,delta_off,bias)
    tau_time_array = model.generate_torques_time_array(tau_array_fun,duration,t0=0.0,include_base=True)
    # Data array initialization
    p_COM_time_array = np.zeros((steps,3))
    v_COM_time_array = np.zeros((steps,3))
    joint_power_time_array = np.zeros((steps,model.state.nqd))
    q_time_array = np.zeros((steps,model.state.nq))
    qd_time_array = np.zeros((steps,model.state.nqd))
    qdd_time_array = np.zeros((steps,model.state.nqd))
    # Integrator for the work done by each joint
    joint_power_integrator = crw.Integrator_Forward_Euler(dt,np.zeros(model.state.nqd))
    joint_energy_time_array = np.zeros((steps,model.state.nqd))
    ##########################################
    ####### RUN SIMULATION ###################
    t = 0.0
    # let the leg fall
    for i in range(30):
        p.stepSimulation()
        if graphic_mode:
            time.sleep(dt)
    # walk and record data
    for i in range(steps):
        model.state.update()
        q_time_array[i] = model.state.q.copy()
        qd_time_array[i] = model.state.qd.copy()
        p_COM_curr = np.array(model.COM_position_world())
        v_COM_curr = np.array(model.COM_velocity_world())
        qdd_time_array[i] = model.state.qdd.copy()
        p_COM_time_array[i]=p_COM_curr
        v_COM_time_array[i]=v_COM_curr
        for j, (v_j, tau_j) in enumerate(zip(model.state.qd,tau_time_array[i])):
            joint_power_time_array[i,j] = abs((1+v_j)*tau_j)
        joint_energy = joint_power_integrator.integrate(joint_power_time_array[i])
        joint_energy_time_array[i] = joint_energy
        ### UPDATE LOSS
        # -(x-displacement) + (total energy expenditure)
        # loss += -(p_COM_time_array[i][0]-p_COM_time_array[0][0]) + (joint_energy_time_array[-1]).dot(joint_energy_time_array[-1])
        # VARIATIONAL VERSION -d(x-displacement) + d(total energy expenditure)
        loss += (
            -100*(p_COM_time_array[i][0]-p_COM_time_array[i-1][0]) + 
            0.2*(joint_power_time_array[i]).dot(joint_power_time_array[i]) +
            0*(abs(p_COM_time_array[i][1]-p_COM_time_array[i-1][1]))
            )
        # if soft_constraints:
        #     loss+= (np.exp((p_COM_curr[-1]/(max_COM_height))**2)-1.0 + 
        #         np.exp(((np.linalg.norm(np.mean(v_COM_time_array[i-n_mean:i+1], 0)))/max_COM_speed)**2)-1.0)
        ###
        if (graphic_mode and video_logging):
            img=p.getCameraImage(800, 640, renderer=p.ER_BULLET_HARDWARE_OPENGL)[2]
            video_writer.append_data(img)
        ###
        model.apply_torques(tau_time_array[i], filtered=False)
        p.stepSimulation()
        if graphic_mode:
            time.sleep(dt) 
        t += dt
        ### check simulation to avoid that the model get schwifty
        # can be substituted by soft constraints for the violation of the two conditions
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
    if plot_graph:
        fig_lat, axs_lat = plt.subplots(2,3)
        for i in range(0,num_lat):
            #axs_lat[i//3,i%3].plot(qd_time_array[:,model.mask_act][:,i], color="xkcd:teal", label="qd")
            #axs_lat[i//3,i%3].plot(qdd_time_array[:,model.mask_act][:,i], color="xkcd:light teal", label="qdd")
            axs_lat[i//3,i%3].plot(joint_power_time_array[:,model.mask_act][:,i], color="xkcd:salmon", label="Power")
            axs_lat[i//3,i%3].plot(joint_energy_time_array[:,model.mask_act][:,i], color="xkcd:light salmon", label="Energy")
            axs_lat[i//3,i%3].plot(tau_time_array[:,model.mask_act][:,i], color="xkcd:dark salmon", label="Torque")
            axs_lat[i//3,i%3].plot(q_time_array[:,model.mask_act_shifted][:,i], color="xkcd:dark teal", label="q")
            axs_lat[i//3,i%3].set_title("Lateral joint %d" %i)
        handles_lat, labels_lat = axs_lat[0,0].get_legend_handles_labels()
        #
        fig_lat.legend(handles_lat, labels_lat, loc='center left')
        fig_girdle, axs_girdle = plt.subplots(2,2)
        for i in range(0,len(model.mask_both_legs)):
            #axs_girdle[i//2,i%2].plot(qd_time_array[:,model.mask_both_legs][:,i], color="xkcd:teal", label="qd")
            #axs_girdle[i//2,i%2].plot(qdd_time_array[:,model.mask_both_legs][:,i], color="xkcd:light teal", label="qdd")
            axs_girdle[i//2,i%2].plot(joint_power_time_array[:,model.mask_both_legs][:,i], color="xkcd:salmon", label="Power")
            axs_girdle[i//2,i%2].plot(joint_energy_time_array[:,model.mask_both_legs][:,i], color="xkcd:light salmon", label="Energy")
            axs_girdle[i//2,i%2].plot(tau_time_array[:,model.mask_both_legs][:,i], color="xkcd:dark salmon", label="Torque yeah")
            axs_girdle[i//2,i%2].plot(q_time_array[:,model.mask_both_legs_shifted][:,i], color="xkcd:dark teal", label="q")
        axs_girdle[0,0].set_title("Right abduction")
        axs_girdle[0,1].set_title("Right flexion")
        axs_girdle[1,0].set_title("Left abduction")
        axs_girdle[1,1].set_title("Left flexion")
        handles_girdle, labels_girdle = axs_girdle[0,0].get_legend_handles_labels()
        fig_girdle.legend(handles_girdle, labels_girdle, loc='center left')
        #
        plt.show()
        #
        fig_COM = plt.figure()
        axs_COM = fig_COM.add_subplot(111)
        axs_COM.plot(p_COM_time_array[:,0], p_COM_time_array[:,1],color="xkcd:teal")
        axs_COM.set_title("COM x-y trajectory")
        fig_COM_3D = plt.figure()
        axs_COM_3D = fig_COM_3D.add_subplot(111, projection='3d')
        axs_COM_3D.plot(p_COM_time_array[:,0], p_COM_time_array[:,1], p_COM_time_array[:,2],color="xkcd:teal")
        axs_COM_3D.set_title("COM 3D trajectory")
        #
        plt.show()
    return loss

def generate_parameters_array5(t_stance,
                            A_lat, k_lat, A_abd, A_flex,
                            n_lat, n_abd,
                            t_lat, t_lat_delay, t_abd, t_flex,
                            delta_lat_1, delta_lat_2, delta_lat_3, delta_abd,
                            k_bias_abd, k_bias_flex):
    ### TIME VARIABLES
    dt = 1/240
    duration = 10
    f_walk = 1/(2*t_stance)
    ### TORQUES EVOLUTION VARIABLEs
    # A_lat*(2*k_lat) < 1, 1Nm is a lot already (tested)
    A = (
        (A_lat,
        A_lat*(1+k_lat),
        A_lat*(2+k_lat),
        A_lat*(1+k_lat),
        A_lat),
        (A_abd,A_flex),
        (A_abd,A_flex))
    f1 = ((f_walk,f_walk,f_walk,f_walk,f_walk),(f_walk,f_walk),(f_walk,f_walk))
    #TODO give the option to set n_lat=0 so that abduction can also be a pure cosine wave
    n = ((n_lat,n_lat,n_lat,n_lat,n_lat),(n_lat,0),(n_lat,0))
    # left leg is just as the right leg but translated by half a period
    t1_off = (
        (t_lat,
        t_lat + 1*t_lat_delay,
        t_lat + 2*t_lat_delay,
        t_lat + 3*t_lat_delay,
        t_lat + 4*t_lat_delay),
        (t_abd, t_flex),
        (t_abd+t_stance, t_flex+t_stance)
        )
    # For delta, considering symmetry on the middle spinal joint of the shape of the functions.
    # delta_abd = 0 and n_abd=0 to make the flexion a single sine wave.
    # if (int(n_lat) == 0):
    #     for delta_lat in (delta_lat_1,delta_lat_2,delta_lat_3):
    #         delta_lat = pi/2
    # if (int(n_abd) == 0):
    #     delta_abd = pi/2
    delta_off = (
        (delta_lat_1,
        delta_lat_2,
        delta_lat_3,
        delta_lat_2,
        delta_lat_1
        ),
        (delta_abd, 0),
        (delta_abd, 0))
    # for correct bias on flexion and abduction check how reference system are oriented
    bias = ((0,0,0,0,0),(-k_bias_abd*A_abd,k_bias_flex*A_flex),(k_bias_abd*A_abd,-k_bias_flex*A_flex))
    ### STARTING POSITION VARIABLES
    theta_rg0=pi/12
    theta_lg0=-pi/4
    A_lat_0=-pi/3.5
    theta_lat_0=0.0
    #
    girdle_friction = 0.2
    body_friction = 0.2
    leg_friction = 0.2
    return (dt, t_stance, duration,
    A, f1, n, t1_off, delta_off, bias,
    theta_rg0, theta_lg0, A_lat_0, theta_lat_0,
    girdle_friction, body_friction, leg_friction)

def objective_function5(params):
    #value not contained in params must be passed manually, as below
    loss = run_simulation5(
        *generate_parameters_array5(
            **params,
            t_stance = 0.5,
            n_lat=2, n_abd=2,
            k_bias_abd=0.5, k_bias_flex=0.5
        ),
        keep_in_check=True,soft_constraints=False,
        graphic_mode=False, plot_graph=False
    )
    return loss

def run_bayesian_optimization5(objective_function,param_dist,iterations,filename="trial.csv",save=True):
    trials = Trials()
    seed=int(time.time())
    best_param = fmin(
        objective_function, 
        param_dist, 
        algo=tpe.suggest, 
        max_evals=iterations, 
        trials=trials,
        rstate=np.random.RandomState(seed)
        )
    losses = [x['result']['loss'] for x in trials.trials]
    vals = [x['misc']['vals']for x in trials.trials]
    best_param_values = [val for val in best_param.values()]
    best_param_values = [x for x in best_param.values()]
    print("Best loss obtained: %f\n with parameters: %s" % (min(losses), best_param_values))
    if save:
        dict_csv={}
        dict_csv.update({'Score' : []})
        for key in vals[0]:
            dict_csv.update({key : []})
        for index,val in enumerate(vals):
            for key in val:
                dict_csv[key].append((val[key])[0])
            dict_csv['Score'].append(losses[index])
        df = pd.DataFrame.from_dict(dict_csv, orient='columns')
        df = df.sort_values("Score",axis=0,ascending=True)
        df.to_csv(path_or_buf=("./optimization results/"+filename),sep=',', index_label='Index')

param_dist5 = {
    "A_lat": hp.uniform("A_lat",0.05,0.15),
    "k_lat": hp.uniform("k_lat",0.5,1),
    "A_abd": hp.uniform("A_abd",0.05,0.2),
    "A_flex": hp.uniform("A_flex",0.05,0.1),
    "t_lat": hp.uniform("t_lat",0.0,0.5),
    "t_lat_delay": hp.uniform("t_lat_delay",0.0,0.25),
    "t_abd": hp.uniform("t_abd",0.0,0.5),
    "t_flex": hp.uniform("t_flex",0.0,0.5),
    "delta_lat_1": hp.uniform("delta_lat_1",0.0,pi),
    "delta_lat_2": hp.uniform("delta_lat_2",0.0,pi),
    "delta_lat_3": hp.uniform("delta_lat_3",0.0,pi),
    "delta_abd": hp.uniform("delta_abd",0.0,pi)
}
csv_name5="ultraultraweak_02friction_ycost.csv"
#run_bayesian_optimization5(objective_function=objective_function5,param_dist=param_dist5,iterations=n_iter, filename=csv_name5)
if False:
    trial_params_dict5 = {
        'A_abd': 0.5160641349761493,
        'A_flex': 0.5699905310095197,
        'A_lat': 0.37343856127784986,
        'delta_abd': 3.0972197864513813,
        'delta_lat_1': 1.1918035584969797,
        'delta_lat_2': 1.296461810132655,
        'delta_lat_3': 2.690594077074573,
        'k_lat': 0.6364078952413598,
        't_abd': 0.01087000771822172,
        't_flex': 0.16850242486762096,
        't_lat': 0.368409259583114,
        't_lat_delay': 0.07711040413095935}

    trial_params_tuple5 = (0.06762201275412225,0.052694574112614925,0.06063162701508158,2.535067445350104,2.371883114859409,1.9661491827330464,2.2652397362792427,0.5275395566057579,0.2082804463884652,0.4940662663261475,0.4166513108763533,0.03199103490565866)
    for key,val in zip(trial_params_dict5,trial_params_tuple5):
        trial_params_dict5[key] = val
    # print(trial_params_dict)

    loss = run_simulation5(
            *generate_parameters_array5(
                **trial_params_dict5,
                t_stance = 0.5,
                n_lat=2, n_abd=2,
                k_bias_abd=0.5, k_bias_flex=0.5
            ),
            keep_in_check=True,soft_constraints=False,
            graphic_mode=True, plot_graph=False
        )
    print("loss: ",loss)
######################################################
def run_simulation6(dt, t_stance, duration,
    A, f1, n, t1_off, delta_off, bias,
    theta_rg0, theta_lg0, A_lat_0, theta_lat_0,
    girdle_friction, body_friction, leg_friction,
    keep_in_check=True, soft_constraints=False,
    graphic_mode=False, plot_graph=False, video_logging=False, video_name="./video/dump/dump"):
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
    z_rotation = 0#pi/3.5 #radians
    model = crw.Crawler(urdf_path="/home/fra/Uni/Tesi/crawler", dt_simulation=dt, base_position=[0,0,0.05], base_orientation=[0,0,sin(z_rotation/2),cos(z_rotation/2)])
    # for i,elem in enumerate(model.links_state_array):
    #     print("Link %d  "%i, elem["world_com_trn"])
    # print(model.links_state_array)
    # Set motion damping ("air friction")
    for index in range(-1,model.num_joints):
        p.changeDynamics(
            model.Id,
            index,
            linearDamping=0.01,
            angularDamping=0.01
        )
    # Girdle link dynamic properties
    p.changeDynamics(model.Id,
        linkIndex = -1,
        lateralFriction = girdle_friction,
        spinningFriction = girdle_friction/10,
        rollingFriction = girdle_friction/20,
        restitution = 0.1,
        #contactStiffness = 0,
        #contactDamping = 0
        )
    # Body dynamic properties
    for i in range(0,model.num_joints-4):
        p.changeDynamics(model.Id,
            linkIndex = i,
            lateralFriction = body_friction,
            spinningFriction = body_friction/10,
            rollingFriction = body_friction/20,
            restitution = 0.1,
            #contactStiffness = 0,
            #contactDamping = 0
            )
    # Legs dynamic properties
    for i in range(model.num_joints-4,model.num_joints):
        p.changeDynamics(model.Id,
            linkIndex = i,
            lateralFriction = leg_friction,
            spinningFriction = leg_friction/10,
            rollingFriction = leg_friction/20,
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
    #model.set_bent_position(theta_rg=theta_rg0, theta_lg=theta_lg0,A_lat=A_lat_0,theta_lat_0=0.0)
    ##########################################
    ####### MISCELLANEOUS ####################
    if graphic_mode:
        p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
        p.resetDebugVisualizerCamera(cameraDistance=0.5, cameraYaw = 50, cameraPitch=-60, cameraTargetPosition=[0,0,0])
    if (video_logging and graphic_mode):
        video_writer = imageio.get_writer(video_name, fps=int(1/dt))
    ##########################################
    ####### PERFORMANCE EVALUATION ###########
    loss = 0.0
    max_loss = 10000000.0 # value to set when something goes wrong, if constraints are "hard"
    max_COM_height = 2*model.body_sphere_radius
    max_COM_speed = 0.8 #TODO: check this value
    n_mean = 3
    ##########################################
    ####### TIME SERIES CREATION #############
    steps = int(duration/dt)
    num_lat = len(model.control_indices[0])
    # Torques evolution
    tau_array_fun = model.get_torques_profile_fun(A,f1,n,t1_off,delta_off,bias)
    tau_time_array = model.generate_torques_time_array(tau_array_fun,duration,t0=0.0,include_base=True)
    # Data array initialization
    p_COM_time_array = np.zeros((steps,3))
    v_COM_time_array = np.zeros((steps,3))
    joint_power_time_array = np.zeros((steps,model.state.nqd))
    q_time_array = np.zeros((steps,model.state.nq))
    qd_time_array = np.zeros((steps,model.state.nqd))
    qdd_time_array = np.zeros((steps,model.state.nqd))
    # Integrator for the work done by each joint
    joint_power_integrator = crw.Integrator_Forward_Euler(dt,np.zeros(model.state.nqd))
    joint_energy_time_array = np.zeros((steps,model.state.nqd))
    ##########################################
    ####### RUN SIMULATION ###################
    t = 0.0
    # let the leg fall
    for i in range(30):
        p.stepSimulation()
        if graphic_mode:
            time.sleep(dt)
    # walk and record data
    for i in range(steps):
        model.state.update()
        q_time_array[i] = model.state.q.copy()
        qd_time_array[i] = model.state.qd.copy()
        p_COM_curr = np.array(model.COM_position_world())
        v_COM_curr = np.array(model.COM_velocity_world())
        qdd_time_array[i] = model.state.qdd.copy()
        p_COM_time_array[i]=p_COM_curr
        v_COM_time_array[i]=v_COM_curr
        for j, (v_j, tau_j) in enumerate(zip(model.state.qd,tau_time_array[i])):
            joint_power_time_array[i,j] = abs((1+v_j)*tau_j)
        joint_energy = joint_power_integrator.integrate(joint_power_time_array[i])
        joint_energy_time_array[i] = joint_energy
        ### UPDATE LOSS
        # -(x-displacement) + (total energy expenditure)
        # loss = -(p_COM_time_array[i][0]-p_COM_time_array[0][0]) + (joint_energy_time_array[-1]).dot(joint_energy_time_array[-1])
        # VARIATIONAL VERSION -d(x-displacement) + d(total energy expenditure)
        loss += (
            -100*(p_COM_time_array[i][0]-p_COM_time_array[i-1][0]) + 
            (joint_power_time_array[i]).dot(joint_power_time_array[i])
            )
        #loss += -(p_COM_time_array[i][0]-p_COM_time_array[i-1][0]) + (joint_power_time_array[i]).dot(joint_power_time_array[i])
        # if soft_constraints:
        #     loss+= (np.exp((p_COM_curr[-1]/(max_COM_height))**2)-1.0 + 
        #         np.exp(((np.linalg.norm(np.mean(v_COM_time_array[i-n_mean:i+1], 0)))/max_COM_speed)**2)-1.0)
        ###
        if (graphic_mode and video_logging):
            img=p.getCameraImage(800, 640, renderer=p.ER_BULLET_HARDWARE_OPENGL)[2]
            video_writer.append_data(img)
        ###
        model.apply_torques(tau_time_array[i], filtered=False)
        p.stepSimulation()
        if graphic_mode:
            time.sleep(dt) 
        t += dt
        ### check simulation to avoid that the model get schwifty
        # can be substituted by soft constraints for the violation of the two conditions
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
    if plot_graph:
        fig_lat, axs_lat = plt.subplots(2,3)
        for i in range(0,num_lat):
            #axs_lat[i//3,i%3].plot(qd_time_array[:,model.mask_act][:,i], color="xkcd:teal", label="qd")
            #axs_lat[i//3,i%3].plot(qdd_time_array[:,model.mask_act][:,i], color="xkcd:light teal", label="qdd")
            axs_lat[i//3,i%3].plot(joint_power_time_array[:,model.mask_act][:,i], color="xkcd:salmon", label="Power")
            axs_lat[i//3,i%3].plot(joint_energy_time_array[:,model.mask_act][:,i], color="xkcd:light salmon", label="Energy")
            axs_lat[i//3,i%3].plot(tau_time_array[:,model.mask_act][:,i], color="xkcd:dark salmon", label="Torque")
            axs_lat[i//3,i%3].plot(q_time_array[:,model.mask_act_shifted][:,i], color="xkcd:dark teal", label="q")
            axs_lat[i//3,i%3].set_title("Lateral joint %d" %i)
        handles_lat, labels_lat = axs_lat[0,0].get_legend_handles_labels()
        #
        fig_lat.legend(handles_lat, labels_lat, loc='center left')
        fig_girdle, axs_girdle = plt.subplots(2,2)
        for i in range(0,len(model.mask_both_legs)):
            #axs_girdle[i//2,i%2].plot(qd_time_array[:,model.mask_both_legs][:,i], color="xkcd:teal", label="qd")
            #axs_girdle[i//2,i%2].plot(qdd_time_array[:,model.mask_both_legs][:,i], color="xkcd:light teal", label="qdd")
            axs_girdle[i//2,i%2].plot(joint_power_time_array[:,model.mask_both_legs][:,i], color="xkcd:salmon", label="Power")
            axs_girdle[i//2,i%2].plot(joint_energy_time_array[:,model.mask_both_legs][:,i], color="xkcd:light salmon", label="Energy")
            axs_girdle[i//2,i%2].plot(tau_time_array[:,model.mask_both_legs][:,i], color="xkcd:dark salmon", label="Torque yeah")
            axs_girdle[i//2,i%2].plot(q_time_array[:,model.mask_both_legs_shifted][:,i], color="xkcd:dark teal", label="q")
        axs_girdle[0,0].set_title("Right abduction")
        axs_girdle[0,1].set_title("Right flexion")
        axs_girdle[1,0].set_title("Left abduction")
        axs_girdle[1,1].set_title("Left flexion")
        handles_girdle, labels_girdle = axs_girdle[0,0].get_legend_handles_labels()
        fig_girdle.legend(handles_girdle, labels_girdle, loc='center left')
        #
        plt.show()
        #
        fig_COM = plt.figure()
        axs_COM = fig_COM.add_subplot(111)
        axs_COM.plot(p_COM_time_array[:,0], p_COM_time_array[:,1],color="xkcd:teal")
        axs_COM.set_title("COM x-y trajectory")
        fig_COM_3D = plt.figure()
        axs_COM_3D = fig_COM_3D.add_subplot(111, projection='3d')
        axs_COM_3D.plot(p_COM_time_array[:,0], p_COM_time_array[:,1], p_COM_time_array[:,2],color="xkcd:teal")
        axs_COM_3D.set_title("COM 3D trajectory")
        #
        plt.show()
    return loss

def generate_parameters_array6(t_stance,
                            A_lat, k_lat, A_abd, A_flex,
                            n_lat, n_abd,
                            t_lat, t_lat_delay, t_abd, t_flex,
                            delta_lat_1, delta_lat_2, delta_lat_3, delta_abd,
                            k_bias_abd, k_bias_flex):
    ### TIME VARIABLES
    dt = 1/240
    duration = 10
    f_walk = 1/(2*t_stance)
    ### TORQUES EVOLUTION VARIABLEs
    # A_lat*(2*k_lat) < 1, 1Nm is a lot already (tested)
    A = (
        (A_lat,
        A_lat*(1+k_lat),
        A_lat*(2+k_lat),
        A_lat*(1+k_lat),
        A_lat),
        (A_abd,A_flex),
        (A_abd,A_flex))
    f1 = ((f_walk,f_walk,f_walk,f_walk,f_walk),(f_walk,f_walk),(f_walk,f_walk))
    #TODO give the option to set n_lat=0 so that abduction can also be a pure cosine wave
    n = ((n_lat,n_lat,n_lat,n_lat,n_lat),(n_lat,0),(n_lat,0))
    # left leg is just as the right leg but translated by half a period
    t1_off = (
        (t_lat,
        t_lat + 1*t_lat_delay,
        t_lat + 2*t_lat_delay,
        t_lat + 3*t_lat_delay,
        t_lat + 4*t_lat_delay),
        (t_abd, t_flex),
        (t_abd+t_stance, t_flex+t_stance)
        )
    # For delta, considering symmetry on the middle spinal joint of the shape of the functions.
    # delta_abd = 0 and n_abd=0 to make the flexion a single sine wave.
    # if (int(n_lat) == 0):
    #     for delta_lat in (delta_lat_1,delta_lat_2,delta_lat_3):
    #         delta_lat = pi/2
    # if (int(n_abd) == 0):
    #     delta_abd = pi/2
    delta_off = (
        (delta_lat_1,
        delta_lat_2,
        delta_lat_3,
        delta_lat_2,
        delta_lat_1
        ),
        (delta_abd, 0),
        (delta_abd, 0))
    # for correct bias on flexion and abduction check how reference system are oriented
    bias = ((0,0,0,0,0),(-k_bias_abd*A_abd,k_bias_flex*A_flex),(k_bias_abd*A_abd,-k_bias_flex*A_flex))
    ### STARTING POSITION VARIABLES
    theta_rg0=pi/12
    theta_lg0=-pi/4
    A_lat_0=-pi/3.5
    theta_lat_0=0.0
    #
    girdle_friction = 0.3
    body_friction = 0.3
    leg_friction = 0.3
    return (dt, t_stance, duration,
    A, f1, n, t1_off, delta_off, bias,
    theta_rg0, theta_lg0, A_lat_0, theta_lat_0,
    girdle_friction, body_friction, leg_friction)

def objective_function6(params):
    #value not contained in params must be passed manually, as below
    loss = run_simulation6(
        *generate_parameters_array6(
            **params,
            t_stance = 0.5,
            n_lat=2, n_abd=2,
            k_bias_abd=0.5, k_bias_flex=0.5
        ),
        keep_in_check=True,soft_constraints=False,
        graphic_mode=False, plot_graph=False
    )
    return loss

def run_bayesian_optimization6(objective_function,param_dist,iterations,filename="trial.csv",save=True):
    trials = Trials()
    seed=int(time.time())
    best_param = fmin(
        objective_function, 
        param_dist, 
        algo=tpe.suggest, 
        max_evals=iterations, 
        trials=trials,
        rstate=np.random.RandomState(seed)
        )
    losses = [x['result']['loss'] for x in trials.trials]
    vals = [x['misc']['vals']for x in trials.trials]
    best_param_values = [val for val in best_param.values()]
    best_param_values = [x for x in best_param.values()]
    print("Best loss obtained: %f\n with parameters: %s" % (min(losses), best_param_values))
    if save:
        dict_csv={}
        dict_csv.update({'Score' : []})
        for key in vals[0]:
            dict_csv.update({key : []})
        for index,val in enumerate(vals):
            for key in val:
                dict_csv[key].append((val[key])[0])
            dict_csv['Score'].append(losses[index])
        df = pd.DataFrame.from_dict(dict_csv, orient='columns')
        df = df.sort_values("Score",axis=0,ascending=True)
        df.to_csv(path_or_buf=("./optimization results/"+filename),sep=',', index_label='Index')

param_dist6 = {
    "A_lat": hp.uniform("A_lat",0.05,0.2),
    "k_lat": hp.uniform("k_lat",0.5,1),
    "A_abd": hp.uniform("A_abd",0.05,0.2),
    "A_flex": hp.uniform("A_flex",0.05,0.1),
    "t_lat": hp.uniform("t_lat",0.0,0.5),
    "t_lat_delay": hp.uniform("t_lat_delay",0.0,0.25),
    "t_abd": hp.uniform("t_abd",0.0,0.5),
    "t_flex": hp.uniform("t_flex",0.0,0.5),
    "delta_lat_1": hp.uniform("delta_lat_1",0.0,pi),
    "delta_lat_2": hp.uniform("delta_lat_2",0.0,pi),
    "delta_lat_3": hp.uniform("delta_lat_3",0.0,pi),
    "delta_abd": hp.uniform("delta_abd",0.0,pi)
}
csv_name6="ultraweak_03friction_ycost.csv"
#run_bayesian_optimization6(objective_function=objective_function6,param_dist=param_dist6,iterations=n_iter, filename=csv_name6)
if True:
    trial_params_dict6 = {
        'A_abd': 0.5160641349761493,
        'A_flex': 0.5699905310095197,
        'A_lat': 0.37343856127784986,
        'delta_abd': 3.0972197864513813,
        'delta_lat_1': 1.1918035584969797,
        'delta_lat_2': 1.296461810132655,
        'delta_lat_3': 2.690594077074573,
        'k_lat': 0.6364078952413598,
        't_abd': 0.01087000771822172,
        't_flex': 0.16850242486762096,
        't_lat': 0.368409259583114,
        't_lat_delay': 0.07711040413095935}

    trial_params_tuple6 = (0.05015951320014225,0.05477850602460067,0.05007251545639248,2.4537658319308413,1.2656078148785885,1.6405327258931623,1.3654015451212307,0.7632445034247564,0.43654597847760085,0.06963046518053369,0.30650272676634915,0.000294771624209619)
    for key,val in zip(trial_params_dict6,trial_params_tuple6):
        trial_params_dict6[key] = val
    # print(trial_params_dict)

    loss = run_simulation6(
            *generate_parameters_array6(
                **trial_params_dict6,
                t_stance = 0.5,
                n_lat=2, n_abd=2,
                k_bias_abd=0.5, k_bias_flex=0.5
            ),
            keep_in_check=True,soft_constraints=False,
            graphic_mode=True, plot_graph=True
        )
    print("loss: ",loss)
#####################################################
def run_simulation7(dt, t_stance, duration,
    A, f1, n, t1_off, delta_off, bias,
    theta_rg0, theta_lg0, A_lat_0, theta_lat_0,
    girdle_friction, body_friction, leg_friction,
    keep_in_check=True, soft_constraints=False,
    graphic_mode=False, plot_graph=False, video_logging=False, video_name="./video/dump/dump"):
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
    z_rotation = 0#pi/3.5 #radians
    model = crw.Crawler(urdf_path="/home/fra/Uni/Tesi/crawler", dt_simulation=dt, base_position=[0,0,0.05], base_orientation=[0,0,sin(z_rotation/2),cos(z_rotation/2)])
    # for i,elem in enumerate(model.links_state_array):
    #     print("Link %d  "%i, elem["world_com_trn"])
    # print(model.links_state_array)
    # Set motion damping ("air friction")
    for index in range(-1,model.num_joints):
        p.changeDynamics(
            model.Id,
            index,
            linearDamping=0.01,
            angularDamping=0.01
        )
    # Girdle link dynamic properties
    p.changeDynamics(model.Id,
        linkIndex = -1,
        lateralFriction = girdle_friction,
        spinningFriction = girdle_friction/10,
        rollingFriction = girdle_friction/20,
        restitution = 0.1,
        #contactStiffness = 0,
        #contactDamping = 0
        )
    # Body dynamic properties
    for i in range(0,model.num_joints-4):
        p.changeDynamics(model.Id,
            linkIndex = i,
            lateralFriction = body_friction,
            spinningFriction = body_friction/10,
            rollingFriction = body_friction/20,
            restitution = 0.1,
            #contactStiffness = 0,
            #contactDamping = 0
            )
    # Legs dynamic properties
    for i in range(model.num_joints-4,model.num_joints):
        p.changeDynamics(model.Id,
            linkIndex = i,
            lateralFriction = leg_friction,
            spinningFriction = leg_friction/10,
            rollingFriction = leg_friction/20,
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
    #model.set_bent_position(theta_rg=theta_rg0, theta_lg=theta_lg0,A_lat=A_lat_0,theta_lat_0=0.0)
    ##########################################
    ####### MISCELLANEOUS ####################
    if graphic_mode:
        p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
        p.resetDebugVisualizerCamera(cameraDistance=0.5, cameraYaw = 50, cameraPitch=-60, cameraTargetPosition=[0,0,0])
    if (video_logging and graphic_mode):
        video_writer = imageio.get_writer(video_name, fps=int(1/dt))
    ##########################################
    ####### PERFORMANCE EVALUATION ###########
    loss = 0.0
    max_loss = 10000000.0 # value to set when something goes wrong, if constraints are "hard"
    max_COM_height = 2*model.body_sphere_radius
    max_COM_speed = 0.8 #TODO: check this value
    n_mean = 3
    ##########################################
    ####### TIME SERIES CREATION #############
    steps = int(duration/dt)
    num_lat = len(model.control_indices[0])
    # Torques evolution
    tau_array_fun = model.get_torques_profile_fun(A,f1,n,t1_off,delta_off,bias)
    tau_time_array = model.generate_torques_time_array(tau_array_fun,duration,t0=0.0,include_base=True)
    # Data array initialization
    p_COM_time_array = np.zeros((steps,3))
    v_COM_time_array = np.zeros((steps,3))
    joint_power_time_array = np.zeros((steps,model.state.nqd))
    q_time_array = np.zeros((steps,model.state.nq))
    qd_time_array = np.zeros((steps,model.state.nqd))
    qdd_time_array = np.zeros((steps,model.state.nqd))
    # Integrator for the work done by each joint
    joint_power_integrator = crw.Integrator_Forward_Euler(dt,np.zeros(model.state.nqd))
    joint_energy_time_array = np.zeros((steps,model.state.nqd))
    ##########################################
    ####### RUN SIMULATION ###################
    t = 0.0
    # let the leg fall
    for i in range(30):
        p.stepSimulation()
        if graphic_mode:
            time.sleep(dt)
    # walk and record data
    for i in range(steps):
        model.state.update()
        q_time_array[i] = model.state.q.copy()
        qd_time_array[i] = model.state.qd.copy()
        p_COM_curr = np.array(model.COM_position_world())
        v_COM_curr = np.array(model.COM_velocity_world())
        qdd_time_array[i] = model.state.qdd.copy()
        p_COM_time_array[i]=p_COM_curr
        v_COM_time_array[i]=v_COM_curr
        for j, (v_j, tau_j) in enumerate(zip(model.state.qd,tau_time_array[i])):
            joint_power_time_array[i,j] = abs((1+v_j)*tau_j)
        joint_energy = joint_power_integrator.integrate(joint_power_time_array[i])
        joint_energy_time_array[i] = joint_energy
        ### UPDATE LOSS
        # -(x-displacement) + (total energy expenditure)
        #loss += -(p_COM_time_array[i][0]-p_COM_time_array[0][0]) + (joint_energy_time_array[-1]).dot(joint_energy_time_array[-1])
        # VARIATIONAL VERSION -d(x-displacement) + d(total energy expenditure)
        loss += (
            -100*(p_COM_time_array[i][0]-p_COM_time_array[i-1][0]) + 
            10*(joint_power_time_array[i]).dot(joint_power_time_array[i])
            )
        # if soft_constraints:
        #     loss+= (np.exp((p_COM_curr[-1]/(max_COM_height))**2)-1.0 + 
        #         np.exp(((np.linalg.norm(np.mean(v_COM_time_array[i-n_mean:i+1], 0)))/max_COM_speed)**2)-1.0)
        ###
        if (graphic_mode and video_logging):
            img=p.getCameraImage(800, 640, renderer=p.ER_BULLET_HARDWARE_OPENGL)[2]
            video_writer.append_data(img)
        ###
        model.apply_torques(tau_time_array[i], filtered=False)
        p.stepSimulation()
        if graphic_mode:
            time.sleep(dt) 
        t += dt
        ### check simulation to avoid that the model get schwifty
        # can be substituted by soft constraints for the violation of the two conditions
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
    if plot_graph:
        fig_lat, axs_lat = plt.subplots(2,3)
        for i in range(0,num_lat):
            #axs_lat[i//3,i%3].plot(qd_time_array[:,model.mask_act][:,i], color="xkcd:teal", label="qd")
            #axs_lat[i//3,i%3].plot(qdd_time_array[:,model.mask_act][:,i], color="xkcd:light teal", label="qdd")
            axs_lat[i//3,i%3].plot(joint_power_time_array[:,model.mask_act][:,i], color="xkcd:salmon", label="Power")
            axs_lat[i//3,i%3].plot(joint_energy_time_array[:,model.mask_act][:,i], color="xkcd:light salmon", label="Energy")
            axs_lat[i//3,i%3].plot(tau_time_array[:,model.mask_act][:,i], color="xkcd:dark salmon", label="Torque")
            axs_lat[i//3,i%3].plot(q_time_array[:,model.mask_act_shifted][:,i], color="xkcd:dark teal", label="q")
            axs_lat[i//3,i%3].set_title("Lateral joint %d" %i)
        handles_lat, labels_lat = axs_lat[0,0].get_legend_handles_labels()
        #
        fig_lat.legend(handles_lat, labels_lat, loc='center left')
        fig_girdle, axs_girdle = plt.subplots(2,2)
        for i in range(0,len(model.mask_both_legs)):
            #axs_girdle[i//2,i%2].plot(qd_time_array[:,model.mask_both_legs][:,i], color="xkcd:teal", label="qd")
            #axs_girdle[i//2,i%2].plot(qdd_time_array[:,model.mask_both_legs][:,i], color="xkcd:light teal", label="qdd")
            axs_girdle[i//2,i%2].plot(joint_power_time_array[:,model.mask_both_legs][:,i], color="xkcd:salmon", label="Power")
            axs_girdle[i//2,i%2].plot(joint_energy_time_array[:,model.mask_both_legs][:,i], color="xkcd:light salmon", label="Energy")
            axs_girdle[i//2,i%2].plot(tau_time_array[:,model.mask_both_legs][:,i], color="xkcd:dark salmon", label="Torque yeah")
            axs_girdle[i//2,i%2].plot(q_time_array[:,model.mask_both_legs_shifted][:,i], color="xkcd:dark teal", label="q")
        axs_girdle[0,0].set_title("Right abduction")
        axs_girdle[0,1].set_title("Right flexion")
        axs_girdle[1,0].set_title("Left abduction")
        axs_girdle[1,1].set_title("Left flexion")
        handles_girdle, labels_girdle = axs_girdle[0,0].get_legend_handles_labels()
        fig_girdle.legend(handles_girdle, labels_girdle, loc='center left')
        #
        plt.show()
        #
        fig_COM = plt.figure()
        axs_COM = fig_COM.add_subplot(111)
        axs_COM.plot(p_COM_time_array[:,0], p_COM_time_array[:,1],color="xkcd:teal")
        axs_COM.set_title("COM x-y trajectory")
        fig_COM_3D = plt.figure()
        axs_COM_3D = fig_COM_3D.add_subplot(111, projection='3d')
        axs_COM_3D.plot(p_COM_time_array[:,0], p_COM_time_array[:,1], p_COM_time_array[:,2],color="xkcd:teal")
        axs_COM_3D.set_title("COM 3D trajectory")
        #
        plt.show()
    return loss

def generate_parameters_array7(t_stance,
                            A_lat, k_lat, A_abd, A_flex,
                            n_lat, n_abd,
                            t_lat, t_lat_delay, t_abd, t_flex,
                            delta_lat_1, delta_lat_2, delta_lat_3, delta_abd,
                            k_bias_abd, k_bias_flex):
    ### TIME VARIABLES
    dt = 1/240
    duration = 10
    f_walk = 1/(2*t_stance)
    ### TORQUES EVOLUTION VARIABLEs
    # A_lat*(2*k_lat) < 1, 1Nm is a lot already (tested)
    A = (
        (A_lat,
        A_lat*(1+k_lat),
        A_lat*(2+k_lat),
        A_lat*(1+k_lat),
        A_lat),
        (A_abd,A_flex),
        (A_abd,A_flex))
    f1 = ((f_walk,f_walk,f_walk,f_walk,f_walk),(f_walk,f_walk),(f_walk,f_walk))
    #TODO give the option to set n_lat=0 so that abduction can also be a pure cosine wave
    n = ((n_lat,n_lat,n_lat,n_lat,n_lat),(n_lat,0),(n_lat,0))
    # left leg is just as the right leg but translated by half a period
    t1_off = (
        (t_lat,
        t_lat + 1*t_lat_delay,
        t_lat + 2*t_lat_delay,
        t_lat + 3*t_lat_delay,
        t_lat + 4*t_lat_delay),
        (t_abd, t_flex),
        (t_abd+t_stance, t_flex+t_stance)
        )
    # For delta, considering symmetry on the middle spinal joint of the shape of the functions.
    # delta_abd = 0 and n_abd=0 to make the flexion a single sine wave.
    # if (int(n_lat) == 0):
    #     for delta_lat in (delta_lat_1,delta_lat_2,delta_lat_3):
    #         delta_lat = pi/2
    # if (int(n_abd) == 0):
    #     delta_abd = pi/2
    delta_off = (
        (delta_lat_1,
        delta_lat_2,
        delta_lat_3,
        delta_lat_2,
        delta_lat_1
        ),
        (delta_abd, 0),
        (delta_abd, 0))
    # for correct bias on flexion and abduction check how reference system are oriented
    bias = ((0,0,0,0,0),(-k_bias_abd*A_abd,k_bias_flex*A_flex),(k_bias_abd*A_abd,-k_bias_flex*A_flex))
    ### STARTING POSITION VARIABLES
    theta_rg0=pi/12
    theta_lg0=-pi/4
    A_lat_0=-pi/3.5
    theta_lat_0=0.0
    #
    girdle_friction = 0.2
    body_friction = 0.2
    leg_friction = 0.2
    return (dt, t_stance, duration,
    A, f1, n, t1_off, delta_off, bias,
    theta_rg0, theta_lg0, A_lat_0, theta_lat_0,
    girdle_friction, body_friction, leg_friction)

def objective_function7(params):
    #value not contained in params must be passed manually, as below
    loss = run_simulation7(
        *generate_parameters_array7(
            **params,
            t_stance = 0.5,
            n_lat=2, n_abd=2,
            k_bias_abd=0.5, k_bias_flex=0.5
        ),
        keep_in_check=True,soft_constraints=False,
        graphic_mode=False, plot_graph=False
    )
    return loss

def run_bayesian_optimization7(objective_function,param_dist,iterations,filename="trial.csv",save=True):
    trials = Trials()
    seed=int(time.time())
    best_param = fmin(
        objective_function, 
        param_dist, 
        algo=tpe.suggest, 
        max_evals=iterations, 
        trials=trials,
        rstate=np.random.RandomState(seed)
        )
    losses = [x['result']['loss'] for x in trials.trials]
    vals = [x['misc']['vals']for x in trials.trials]
    best_param_values = [val for val in best_param.values()]
    best_param_values = [x for x in best_param.values()]
    print("Best loss obtained: %f\n with parameters: %s" % (min(losses), best_param_values))
    if save:
        dict_csv={}
        dict_csv.update({'Score' : []})
        for key in vals[0]:
            dict_csv.update({key : []})
        for index,val in enumerate(vals):
            for key in val:
                dict_csv[key].append((val[key])[0])
            dict_csv['Score'].append(losses[index])
        df = pd.DataFrame.from_dict(dict_csv, orient='columns')
        df = df.sort_values("Score",axis=0,ascending=True)
        df.to_csv(path_or_buf=("./optimization results/"+filename),sep=',', index_label='Index')

param_dist7 = {
    "A_lat": hp.uniform("A_lat",0.05,0.15),
    "k_lat": hp.uniform("k_lat",0.5,1),
    "A_abd": hp.uniform("A_abd",0.05,0.2),
    "A_flex": hp.uniform("A_flex",0.05,0.1),
    "t_lat": hp.uniform("t_lat",0.0,0.5),
    "t_lat_delay": hp.uniform("t_lat_delay",0.0,0.25),
    "t_abd": hp.uniform("t_abd",0.0,0.5),
    "t_flex": hp.uniform("t_flex",0.0,0.5),
    "delta_lat_1": hp.uniform("delta_lat_1",0.0,pi),
    "delta_lat_2": hp.uniform("delta_lat_2",0.0,pi),
    "delta_lat_3": hp.uniform("delta_lat_3",0.0,pi),
    "delta_abd": hp.uniform("delta_abd",0.0,pi)
}
csv_name7="ultraultraweak_02friction_10xEnergycost.csv"
#run_bayesian_optimization7(objective_function=objective_function7,param_dist=param_dist7,iterations=n_iter, filename=csv_name7)
if False:
    trial_params_dict = {
        'A_abd': 0.5160641349761493,
        'A_flex': 0.5699905310095197,
        'A_lat': 0.37343856127784986,
        'delta_abd': 3.0972197864513813,
        'delta_lat_1': 1.1918035584969797,
        'delta_lat_2': 1.296461810132655,
        'delta_lat_3': 2.690594077074573,
        'k_lat': 0.6364078952413598,
        't_abd': 0.01087000771822172,
        't_flex': 0.16850242486762096,
        't_lat': 0.368409259583114,
        't_lat_delay': 0.07711040413095935}

    trial_params_tuple = (0.10938820258459488, 0.14140431170454476, 0.16853955114468958, 0.33235595592028844, 0.40463274872018373, 2.4379666733636345, 0.47872192307268224, 0.7161509762835492, 0.00035931453043262784, 0.03421267164449099, 0.014685591146372738, 0.2276936274930831)
    for key,val in zip(trial_params_dict,trial_params_tuple):
        trial_params_dict[key] = val
    # print(trial_params_dict)

    loss = run_simulation7(
            *generate_parameters_array7(
                **trial_params_dict,
                t_stance = 0.5,
                n_lat=2, n_abd=2,
                k_bias_abd=0.5, k_bias_flex=0.5
            ),
            keep_in_check=True,soft_constraints=False,
            graphic_mode=True, plot_graph=True
        )
    print("loss: ",loss)