import pybullet as p
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import time
import pybullet_data
import crawler_old as crw
from math import *
import pinocchio as pin
import pyparsing


matplotlib.use('TkAgg')

urdf_path="/home/fra/Uni/Tesi/crawler"
urdf_filename=urdf_path+"/crawler.urdf"

#########################################
####### SIMULATION SET-UP ###############
#########################################
### time-step used for simulation and total time passed variable
dt = 1./240.
#
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0,0,-9.81)
#p.setGravity(0.0,0.0,0.0)
#planeID = p.loadURDF("plane100.urdf", [0,0,0])
model = crw.Crawler(urdf_path=urdf_path, dt_simulation=dt, base_position=[0,0,0.5], base_orientation=[0,0,0,1])
np.random.seed()

print(model.trav_wave_theta(np.linspace(0,5,50),2,0.5,1,3,0))

model.set_low_pass_qd(fc=7000)
for index in range(-1,model.num_joints):
    p.changeDynamics(
        model.Id,
        index,
        linearDamping=0.0,
        angularDamping=0.0
    )
# qd_prev = model.get_joints_speeds_tuple()
# qdd_curr = model.get_joints_acc_nparr(qd_prev=qd_prev)
#print(model.solve_torques_lateral(np.array(list(model.control_indices[0])),qdd_curr))
model.turn_off_crawler()
# for i in range (240):
#     p.stepSimulation()
#     time.sleep(dt)
#model.turn_off_crawler()
###
M = p.calculateMassMatrix(model.Id, model.get_joints_pos_tuple())
M = np.array(M)
Ma = M[:,model.mask_act][model.mask_act,:]
print(np.round(M,3))
print("DIM M = ", M.shape)
print(np.round(Ma,3))
print("DIM Ma = ", Ma.shape)
###
q_list = list(model.get_joints_pos_tuple()) + ([0]*6)
qd_list = list(model.get_joints_speeds_tuple())
qd_list = [1]*(model.num_joints+6)
qdd_des_list= [1]*(model.num_joints+6)
tau_exact = np.array(
    p.calculateInverseDynamics(
        model.Id,
        q_list,
        qd_list,
        qdd_des_list,
        flags=1
        )
    )
print("TAU EXACT\n", tau_exact)
print("Q = ", model.get_q())
model.set_velocities()
p.stepSimulation()
time.sleep(dt)
model.set_velocities()
print("qd_prev", model.qd_prev)
print("qd_curr", model.qd_curr)
qdd_history = np.array(([0]*(model.num_joints+6)))
qdd_f_history = np.array(([0]*(model.num_joints+6)))
model.turn_off_crawler()
for i in range(20):
    p.stepSimulation()
    model.set_velocities()
    q = model.get_q()
    qd = model.qd_curr
    acc = model.get_qdd_nparr()
    qdd = acc[0]
    qdd_f = acc[1]
    qdd_history = np.vstack((qdd_history,qdd))
    qdd_f_history = np.vstack((qdd_f_history,qdd_f))
    time.sleep(dt)
print("\nEEEEEEEEEEEEEEEEEEEEEEEEEEEEE",qdd,"\n")

# fig, ax = plt.subplots()
# plt.plot(qdd_history[:,6],"b")
# plt.plot(qdd_f_history[:,6],"r")
# #plt.plot(qdaf3,"r")
# plt.show()
# print(qdd)
###

#########################################
####### PINOCCHIO TEST ##################
#########################################
pinmodel = pin.buildModelFromUrdf(urdf_filename,pin.JointModelFreeFlyer())
print('model name: ' + pinmodel.name)
# Create data required by the algorithms
data = pinmodel.createData()
###
#qdd = np.array(([0]*(model.num_joints+6)))
###
print("\n\n\nYOHHHHHHH")
q_pin = q[model.mask_q_pyb_to_pin]
qd_pin = qd[model.mask_qd_pyb_to_pin]
qdd_pin = qdd[model.mask_qd_pyb_to_pin]
# q_pin[2]=1
# qd_pin[2]=1
# qdd_pin[2]=-9.81
print("shape q_pin:", q_pin.shape)
print("Q_PIN\n",np.round(q_pin,5))
print("QD_PIN\n",np.round(qd_pin,5))
print("QDD_PIN\n",np.round(qdd_pin,5))
pin.crba(pinmodel,data,q_pin)
pin.computeCoriolisMatrix(pinmodel,data,q_pin,qd_pin)
datag = pin.computeGeneralizedGravity(pinmodel,data,q_pin)
pin.rnea(pinmodel,data,q_pin,qd_pin,qdd_pin)
print("M\n",np.round(data.M,5))
print("C\n",np.round(data.C,5))
print("g\n",np.round(datag,5))
print("TAU_PIN\n",np.round(data.tau,4))
print("\n")
indices = list(range(model.num_joints))
joint_states = p.getJointStates(model.Id,indices)
joint_states_np = np.array(joint_states)
tau_real_np = joint_states_np[:,3]
print("TAU REAL SHAPE: ",tau_real_np.shape)
print("TAU REAL: ",tau_real_np)
print("\n\n\n")


###
# Sample a random configuration
# print("Q RANDOM")
# q = pin.randomConfiguration(pinmodel)
# print(q)
# print(q.shape)
#print('q: %s' % q.T)
# Perform the forward kinematics over the kinematic tree

# print(pin.rnea(pinmodel,data,np.array(([0]*(pinmodel.nq+6))),np.array(([0]*(pinmodel.nq+6))),np.array(([0]*(pinmodel.nq+6)))))
# Print out the placement of each joint of the kinematic tree
# pin.forwardKinematics(pinmodel,data,q)
for name, oMi, tau in zip(pinmodel.names, data.oMi, data.tau):
    print(("{:<24} : {: .2f} {: .2f} {: .2f} : {: .2f}"
        .format( name, *oMi.translation.T.flat, tau )))
# print(model.mask_q_py_to_pin)
# print(model.mask_qd_py_to_pin)
p.disconnect()


# dt = 1/240
# low_pass = crawler.Discrete_Low_Pass(dim=3,dt=dt,tc=1/10,K=1)
# original_signal = np.array([0,0,0])
# original_signal = np.reshape(original_signal,(1,original_signal.shape[0]))
# filtered_signal = np.array([0,0,0])
# filtered_signal = np.reshape(filtered_signal,(1,filtered_signal.shape[0]))

# for t in np.arange(0,10,1/240) : 
#     s1 = 4*np.sin(pi*1 * t)# + 5*np.sin(0.01 * t)
#     s2 = 4*np.sin(10 * t)# + 5*np.sin(0.01 * t)
#     s3 = 1 + 4*np.sin(pi*1000 * t)# + 5*np.sin(0.01 * t)
#     s = np.array((s1,s2,s3))
#     sf = low_pass.filter(s)
#     s=np.reshape(s,(1,s.shape[0]))
#     sf=np.reshape(sf,(1,sf.shape[0]))
#     original_signal= np.concatenate((original_signal,s))
#     filtered_signal= np.concatenate((filtered_signal,sf))

# fig, axes = plt.subplots(nrows=1,ncols=3)
# axes[0].plot(original_signal[:,0], 'r+')
# axes[0].plot(filtered_signal[:,0], 'ro')
# axes[1].plot(original_signal[:,1], 'g+')
# axes[1].plot(filtered_signal[:,1], 'go')
# axes[2].plot(original_signal[:,2], 'b+')
# axes[2].plot(filtered_signal[:,2], 'bo')
# plt.show(fig)

# physicsClient = p.connect(p.GUI)
# model = crawler.Crawler()

# print(model.generate_fmax_array_lateral(5))
# print(model.generate_gain_matrix_lateral(10))

# p.disconnect()

# control_indices = [1,3,5,7,9,11,13,15]
# k_diag = list()
# half_spine_index = int(len(control_indices)/2)
# end_spine_index = len(control_indices)
# for i in range(1, half_spine_index+1):
#     k_diag.append(10*i)
# for i in range(half_spine_index,end_spine_index):
#     k_diag.append(10*(end_spine_index-i))
# print(k_diag)
# K = np.diag(k_diag)
# print(K)

# for index, value in enumerate(list((3,4,5,6,7))):
#     print("index: ", index, "value: ", value)

# def yoh_func(RL=(False,False)):
#     if not(RL[0]) and not(RL[1]):
#         print("yoh")
#         return
#     else:
#         print("not yoh")
#         return
# yoh_func(RL=(False,False))
# yoh_func(RL=(True,False))
# yoh_func(RL=(True,True))
# yoh_func(RL=(False,True))

# R=np.reshape(np.array(p.getMatrixFromQuaternion([0,0,sin(pi/5/2),cos(pi/5/2)])),(3,3))
# Rinv = np.linalg.inv(R)
# print(R.dot(Rinv))
# print(R,"\n",Rinv)
# print(p.getMatrixFromQuaternion([0,0,sin(pi/5/2),cos(pi/5/2)]))

# def yoh(x):
#     print(x+1)
#     return x+1

# a = np.array((1,2,3,4,5,6))
# b=list(map(yoh, a))
# print(b)

# def outer():
#     x0=0
#     def inner(signal):
#         nonlocal x0
#         x0 += signal
#         return x0
#     return inner

# class Discrete_Low_Pass:
#     def __init__(self, dt, tc, K=1):
#         self.x = 0
#         self.dt = dt
#         self.tc = tc
#         self.K = K
#     def reset(self):
#         self.x = 0
#     def filter(self, signal):
#         self.x = (1-self.dt/self.tc)*self.x + self.K*self.dt/self.tc * signal
#         return self.x

# dt = 1/240
# myfilter = Discrete_Low_Pass(dt, 1, 1)

# original_signal = [0]  
# filtered_signal = [0]

# for t in np.arange(0,30,1/240) : 
#     s = 1 + 3*np.sin(100 * t)# + 5*np.sin(0.01 * t)
#     original_signal.append(s)
#     filtered_signal.append(myfilter.filter(s))

# fig, ax = plt.subplots()
# plot(original_signal, 'r+')
# plot(filtered_signal, 'b+')
# plt.show()
