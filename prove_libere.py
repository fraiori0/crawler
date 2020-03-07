import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import plot
import pybullet as p
from math import *

def yoh_func(RL=(False,False)):
    if not(RL[0]) and not(RL[1]):
        print("yoh")
        return
    else:
        print("not yoh")
        return
yoh_func(RL=(False,False))
yoh_func(RL=(True,False))
yoh_func(RL=(True,True))
yoh_func(RL=(False,True))

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
