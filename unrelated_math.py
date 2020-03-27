from math import *
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import imageio
import imageio.plugins.pillow

matplotlib.use('TkAgg')
#video
duration = 5
fps = 40
dt = 1/fps
frames = int(fps*duration)
# imageio set-up
title="first try"
video_name = title+".mp4"
video_path = "./animations_matplotlib/" + video_name
writer = imageio.get_writer(video_path, fps=fps)
fig_name = "last_frame.jpg"
fig_path = "./animations_matplotlib/" + fig_name
#matplotlib set-up
fig = plt.figure()
ax = fig.add_subplot(111)
#geometrical paramaters
ll = 0.5
wg = 0.5
lg = 0.5
lb = 1
#np-arrays generation
theta_sum = pi/4
t2 = np.linspace(-pi/4,pi/4, frames)
t1 = theta_sum-t2
t3 = np.zeros(frames)
Cx = np.zeros(frames)
Cy = np.zeros(frames)
GRx = ll*np.cos(t1)
GRy = ll*np.sin(t1)
GLx = GRx + wg*np.cos(t1+t2)
GLy = GRy + wg*np.sin(t1+t2)
Hx = GRx + wg*np.cos(t1+t2)/2 - -lg*np.sin(t1+t2)/2
Hy = GRy + wg*np.sin(t1+t2)/2 + -lg*np.cos(t1+t2)/2
Bx = GRx + wg*np.cos(t1+t2)/2 - lg*np.sin(t1+t2)/2
By = GRy + wg*np.sin(t1+t2)/2 + lg*np.cos(t1+t2)/2
Tx = Bx - lb*np.sin(t1+t2+t3)
Ty = By + lb*np.cos(t1+t2+t3)
# trajectories
mH = 1
mB = 1
mT = 1
Mx= (Hx*mH + Bx*mB + Tx*mT)/(mH+mB+mT)
My= (Hy*mH + By*mB + Ty*mT)/(mH+mB+mT)
Mtraj = np.vstack((Mx,My))
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
    leg = matplotlib.lines.Line2D([Cx[i],GRx[i]],[Cy[i],GRy[i]],linewidth=linewidth_body, figure=fig, color="xkcd:teal")
    girdle1 = matplotlib.lines.Line2D([GRx[i],GLx[i]],[GRy[i],GLy[i]],linewidth=linewidth_body, figure=fig, color="xkcd:teal")
    girdle2 = matplotlib.lines.Line2D([Bx[i],Hx[i]],[By[i],Hy[i]],linewidth=linewidth_body, figure=fig, color="xkcd:teal")
    body = matplotlib.lines.Line2D([Bx[i],Tx[i]],[By[i],Ty[i]],linewidth=linewidth_body, figure=fig, color="xkcd:teal")
    line_list=[leg,girdle1,girdle2,body]
    #MASSES
    mhfig = matplotlib.patches.CirclePolygon((Hx[i],Hy[i]), radius=mass_scale*sqrt(mH), linewidth=linewidth_joint, figure=fig, ec="xkcd:black", fc="xkcd:light teal", fill=True)
    mbfig = matplotlib.patches.CirclePolygon((Bx[i],By[i]), radius=mass_scale*sqrt(mB), linewidth=linewidth_joint, figure=fig, ec="xkcd:black", fc="xkcd:light teal", fill=True)
    mtfig = matplotlib.patches.CirclePolygon((Tx[i],Ty[i]), radius=mass_scale*sqrt(mT), linewidth=linewidth_joint, figure=fig, ec="xkcd:black", fc="xkcd:light teal", fill=True)
    patch_list=[mhfig,mbfig,mtfig]
    #COM
    comfig = matplotlib.patches.CirclePolygon((Mx[i],My[i]), radius=0.04, linewidth=linewidth_joint, figure=fig, ec="xkcd:black", fc="xkcd:salmon", fill=True)
    patch_list.append(comfig)
    #JOINTS
    cjoint = matplotlib.patches.CirclePolygon((Cx[i],Cy[i]), radius=radius_joint, linewidth=linewidth_joint, figure=fig, ec="xkcd:black", fc="xkcd:white", fill=True)
    grjoint = matplotlib.patches.CirclePolygon((GRx[i],GRy[i]), radius=radius_joint, linewidth=linewidth_joint, figure=fig, ec="xkcd:black", fc="xkcd:white", fill=True)
    bjoint = matplotlib.patches.CirclePolygon((Bx[i],By[i]), radius=radius_joint, linewidth=linewidth_joint, figure=fig, ec="xkcd:black", fc="xkcd:white", fill=True)
    patch_list.extend([cjoint,grjoint,bjoint])
    for patch in patch_list:
        ax.add_patch(patch)
    for line in line_list:
        ax.add_line(line)
    plt.plot(Mx,My,figure=fig, color="xkcd:salmon")
    return 

# generate frame and mp4
ax.set_aspect('equal')
ax.set(xlim=(-2, 2), ylim=(-2, 2))
for i in range(frames):
    draw_fig(i)
    ax.set_aspect('equal')
    ax.set(xlim=(-2, 1), ylim=(-2, 1))
    ax.set_title(title)
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    img  = img.reshape(fig.canvas.get_width_height()[::-1]+(3,))
    writer.append_data(img)
    ax.clear()


writer.close()