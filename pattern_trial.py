from math import *
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import imageio
import imageio.plugins.pillow
import scipy.integrate as integrate
import scipy as sp


x = np.linspace(5,70,7)
x = x/1000
y = np.array((0.27185,0.52168,0.62944,0.58571,0.44329,0.23971,0.065048))
y = y/1000

coeff = np.polyfit(x,y,deg=3)


print(coeff)
poly = lambda x: coeff[0]*(x**3) + coeff[1]*(x**2) + coeff[2]*(x) + coeff[3]
poly_vec = np.vectorize(poly)
y_gen = poly_vec(x)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x,y)
ax.plot(x,y_gen)
plt.show()


# def return_lambda(A,f1,f2,th2,x_offset=0):
#     fun = lambda t: A*np.sin(2*pi*f*t + th0) + x_offset
#     return fun

# # x0 = return_lambda(1,1,0)
# # x1 = return_lambda(1,0.25,0)
# # x2 = return_lambda(2,0.5,0)
# # x3 = return_lambda(2,0.5,0,pi/2)
# # x4 = return_lambda(2,0.5,0,pi/4)

# fig_name = "last_frame.jpg"
# fig_path = "./animations_matplotlib/" + fig_name
# #matplotlib set-up
# fig, axs = plt.subplots(2,2)

# #axs[0,0].plot(lambda t: x0(t)*x2(t))

# list1 = [0,1,2,3,4]
# list2 = [["a","b","c","d","e"],[],[0,1,3]]
# list1[3]='coccodrillo'
# list1[4]= lambda x: x+1
# for i,(num,let) in enumerate(zip(list1,list2[0])):
#     print(i,"--> ",num," ",let," ", num)

# tuple1 = tuple(list1)
# for i in range(0,10,2):
#     print(tuple1[4](i))
# empty = np.zeros(4)
# empty = np.vstack((empty,np.array((1,1,1,1))))
# print(empty)
# matrix = np.array(((0,0,0),(1,1,1),(2,2,2)))
# print(matrix)
# for i in matrix[1][[1,2]]:
#     print("ue %d" %i)