import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import plot
import pybullet as p
from math import *
import crawler

dt = 1/240
low_pass = crawler.Discrete_Low_Pass(dim=3,dt=dt,tc=1/10,K=1)
original_signal = np.array([0,0,0])
original_signal = np.reshape(original_signal,(1,original_signal.shape[0]))
filtered_signal = np.array([0,0,0])
filtered_signal = np.reshape(filtered_signal,(1,filtered_signal.shape[0]))

for t in np.arange(0,10,1/240) : 
    s1 = 4*np.sin(pi*1 * t)# + 5*np.sin(0.01 * t)
    s2 = 4*np.sin(10 * t)# + 5*np.sin(0.01 * t)
    s3 = 1 + 4*np.sin(pi*1000 * t)# + 5*np.sin(0.01 * t)
    s = np.array((s1,s2,s3))
    sf = low_pass.filter(s)
    s=np.reshape(s,(1,s.shape[0]))
    sf=np.reshape(sf,(1,sf.shape[0]))
    original_signal= np.concatenate((original_signal,s))
    filtered_signal= np.concatenate((filtered_signal,sf))

fig, axes = plt.subplots(nrows=1,ncols=3)
axes[0].plot(original_signal[:,0], 'r+')
axes[0].plot(filtered_signal[:,0], 'ro')
axes[1].plot(original_signal[:,1], 'g+')
axes[1].plot(filtered_signal[:,1], 'go')
axes[2].plot(original_signal[:,2], 'b+')
axes[2].plot(filtered_signal[:,2], 'bo')
plt.show(fig)

# physicsClient = p.connect(p.GUI)
# model = crawler.Crawler(spine_segments=8)filtered_signal = 

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
