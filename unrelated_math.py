from math import *
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import imageio
import imageio.plugins.pillow
import scipy.integrate as integrate
import scipy as sp

matplotlib.use('TkAgg')
#video
duration = 10
fps = 80
dt = 1/fps
frames = int(fps*duration)
time_array = np.linspace(0.0, duration, num=frames)
# imageio set-up
title="Conservation of AM - Orbit"
video_name = title+".mp4"
video_path = "./animations_matplotlib/" + video_name
writer = imageio.get_writer(video_path, fps=fps)
fig_name = "last_frame.jpg"
fig_path = "./animations_matplotlib/" + fig_name
#matplotlib set-up
fig = plt.figure()
ax = fig.add_subplot(111)
# Geometric parameters
ll = 0.5
wg = 0.5
lg = 0.5
lb = 1
# Point positions functions
R = lambda theta: np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
C_fun = lambda: np.array([0,0])
GR_fun = lambda th1: R(th1).dot(np.array([ll,0]))
GL_fun = lambda th1, th2: GR_fun(th1) + R((th1+th2)).dot(np.array([wg,0]))
H_fun = lambda th1, th2: GR_fun(th1) + R((th1+th2)).dot(np.array([wg/2,-lg/2]))
B_fun = lambda th1, th2: GR_fun(th1) + R((th1+th2)).dot(np.array([wg/2,lg/2]))
T_fun = lambda th1, th2, th3: B_fun(th1,th2) + R((th1+th2+th3)).dot(np.array([0,lb]))
mH, mB, mT = (1,1,1)
M_fun = lambda th1,th2,th3,mH,mB,mT: (H_fun(th1,th2)*mH + B_fun(th1,th2)*mB + T_fun(th1,th2,th3)*mT)/(mH+mB+mT)
# Quantities for conservation of angular momentum model
M0 = M_fun(0,0,0,mH,mB,mT)
Ig = (
    mH*np.linalg.norm(H_fun(0,0)-M0) + 
    mB*np.linalg.norm(H_fun(0,0)-M0) +
    mT*np.linalg.norm(T_fun(0,0,0)-M0)
)
l1 = ll
l2 = np.linalg.norm(M0-GL_fun(0,0))
thM_offset = np.arctan2(abs(M0[1]),abs(M0[0]-l1))
# Angles' time-evolution arrays
th2_0 = -pi/4
th2_f = 10*pi
th1_0 = 5*pi/4
span_th2 = 2*pi
dth2_fun = lambda t: span_th2/(duration)
dth2_fun = lambda t: -pi*(th2_0-th2_f)*np.sin(pi*t/duration)/(2*duration)
th2_fun = lambda t: (integrate.quad(dth2_fun,0,t))[0] + th2_0
th2 = np.empty(frames)
for i in range(frames):
    th2[i] = th2_fun(time_array[i])
thM_fun = lambda t: th2_fun(t)+ thM_offset
thM = th2 + thM_offset
dth1_fun = lambda t: (-dth2_fun(t) * (Ig+l1*l2*np.cos(thM_fun(t))+l2**2)/(Ig+l1**2+2*l1*l2*np.cos(thM_fun(t))+l2**2))
th1_fun = lambda t: (integrate.quad(dth1_fun,0,t))[0] + th1_0
th1 = np.empty(frames)
for i in range(frames):
    th1[i] = th1_fun(time_array[i])
th3 = np.zeros(frames)
# Points' positions arrays
C = np.empty((frames,2))
GR = np.empty((frames,2))
GL = np.empty((frames,2))
H = np.empty((frames,2))
B = np.empty((frames,2))
T = np.empty((frames,2))
for i in range(frames):
    C[i] = C_fun()
    GR[i] = GR_fun(th1[i])
    GL[i] = GL_fun(th1[i], th2[i])
    H[i] = H_fun(th1[i], th2[i])
    B[i] = B_fun(th1[i], th2[i])
    T[i] = T_fun(th1[i], th2[i], th3[i])
# COM
M = np.empty((frames,2))
for i in range(frames):
    M[i] = M_fun(th1[i],th2[i],th3[i],mH,mB,mT)
# graphical properties 
linewidth_body = 1.5
linewidth_joint = 1
radius_joint = 0.03
mass_scale = 0.1
#list of patches
line_list = list()
patch_list = list()
# generate figure and save them to mp4
def draw_fig(i):
    #BODY
    leg = matplotlib.lines.Line2D([C[i,0],GR[i,0]],[C[i,1],GR[i,1]],linewidth=linewidth_body, figure=fig, color="xkcd:teal")
    girdle1 = matplotlib.lines.Line2D([GR[i,0],GL[i,0]],[GR[i,1],GL[i,1]],linewidth=linewidth_body, figure=fig, color="xkcd:teal")
    girdle2 = matplotlib.lines.Line2D([B[i,0],H[i,0]],[B[i,1],H[i,1]],linewidth=linewidth_body, figure=fig, color="xkcd:teal")
    body = matplotlib.lines.Line2D([B[i,0],T[i,0]],[B[i,1],T[i,1]],linewidth=linewidth_body, figure=fig, color="xkcd:teal")
    line_list=[leg,girdle1,girdle2,body]
    #MASSES
    mhfig = matplotlib.patches.CirclePolygon(H[i], radius=mass_scale*sqrt(mH), linewidth=linewidth_joint, figure=fig, ec="xkcd:black", fc="xkcd:light teal", fill=True)
    mbfig = matplotlib.patches.CirclePolygon(B[i], radius=mass_scale*sqrt(mB), linewidth=linewidth_joint, figure=fig, ec="xkcd:black", fc="xkcd:light teal", fill=True)
    mtfig = matplotlib.patches.CirclePolygon(T[i], radius=mass_scale*sqrt(mT), linewidth=linewidth_joint, figure=fig, ec="xkcd:black", fc="xkcd:light teal", fill=True)
    patch_list=[mhfig,mbfig,mtfig]
    #COM
    comfig = matplotlib.patches.CirclePolygon(M[i], radius=0.04, linewidth=linewidth_joint, figure=fig, ec="xkcd:black", fc="xkcd:salmon", fill=True)
    patch_list.append(comfig)
    #JOINTS
    cjoint = matplotlib.patches.CirclePolygon(C[i], radius=radius_joint, linewidth=linewidth_joint, figure=fig, ec="xkcd:black", fc="xkcd:white", fill=True)
    grjoint = matplotlib.patches.CirclePolygon(GR[i], radius=radius_joint, linewidth=linewidth_joint, figure=fig, ec="xkcd:black", fc="xkcd:white", fill=True)
    bjoint = matplotlib.patches.CirclePolygon(B[i], radius=radius_joint, linewidth=linewidth_joint, figure=fig, ec="xkcd:black", fc="xkcd:white", fill=True)
    patch_list.extend([cjoint,grjoint,bjoint])
    for patch in patch_list:
        ax.add_patch(patch)
    for line in line_list:
        ax.add_line(line)
    plt.plot(M[:,0],M[:,1],figure=fig, color="xkcd:salmon")
    return 

# generate frame and mp4
ax.set_aspect('equal')
ax.set(xlim=(-2, 2), ylim=(-2, 2))
for i in range(frames):
    draw_fig(i)
    ax.set_aspect('equal')
    ax.set(xlim=(-2, 2), ylim=(-2, 2))
    ax.set_title(title)
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    img  = img.reshape(fig.canvas.get_width_height()[::-1]+(3,))
    writer.append_data(img)
    ax.clear()


writer.close()


