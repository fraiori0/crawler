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
        # loss += (
        #     -100*(p_COM_time_array[i][0]-p_COM_time_array[i-1][0]) + 
        #     (joint_power_time_array[i]).dot(joint_power_time_array[i]) +
        #     (abs(p_COM_time_array[i][1]-p_COM_time_array[i-1][1]))
        #     )
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
            axs_lat[i//3,i%3].plot(qd_time_array[:,model.mask_act][:,i], color="xkcd:teal", label="qd")
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

def generate_parameters_array(t_stance,
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

def objective_function(params):
    #value not contained in params must be passed manually, as below
    loss = run_simulation(
        *generate_parameters_array(
            **params,
            t_stance = 0.5,
            n_lat=2, n_abd=2,
            k_bias_abd=0.7, k_bias_flex=0.7
        ),
        keep_in_check=True,soft_constraints=False,
        graphic_mode=False, plot_graph=False
    )
    return loss

def run_bayesian_optimization(objective_function,param_dist,iterations,filename="trial.csv",save=True):
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

param_dist = {
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
csv_name="trial_ultraweak_02friction_Energycost_ycost_07bias.csv"
#run_bayesian_optimization(objective_function=objective_function,param_dist=param_dist,iterations=100, filename=csv_name)

if True:
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

    loss = run_simulation(
            *generate_parameters_array(
                **trial_params_dict,
                t_stance = 0.5,
                n_lat=2, n_abd=2,
                k_bias_abd=0.5, k_bias_flex=0.5
            ),
            keep_in_check=True,soft_constraints=False,
            graphic_mode=True, plot_graph=True
        )
    print("loss: ",loss)




# loss = run_simulation(
#     *generate_parameters_array(t_stance = 0.5,
#         A_lat = 0.0, k_lat=0.0, A_abd=0.0, A_flex=0.0,
#         n_lat=2, n_abd=2,
#         t_lat=0.0, t_lat_delay=0.05, t_abd=0.0, t_flex=0.125,
#         delta_lat_1=0.0, delta_lat_2=0.0, delta_lat_3=0.0, delta_abd=0.0,
#         k_bias_abd=0.5, k_bias_flex=0.5
#     ),
#     keep_in_check=True, soft_constraints=False,
#     graphic_mode=True, plot_graph=False
# )
# print("loss: ", loss)

