import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import plot

def outer():
    x0=0
    def inner(signal):
        nonlocal x0
        x0 += signal
        return x0
    return inner

class Discretel_Low_Pass:
    def __init__(self, dt, tc, K=1):
        self.x = 0
        self.dt = dt
        self.tc = tc
        self.K = K
    def reset(self):
        self.x = 0
    def filter(self, signal):
        self.x = (1-self.dt/self.tc)*self.x + self.K*self.dt/self.tc * signal
        return self.x

dt = 1/240
myfilter = Discretel_Low_Pass(dt, 1, 1)

original_signal = [0]  
filtered_signal = [0]

for t in np.arange(0,30,1/240) : 
    s = 1 + 3*np.sin(100 * t)# + 5*np.sin(0.01 * t)
    original_signal.append(s)
    filtered_signal.append(myfilter.filter(s))

fig, ax = plt.subplots()
plot(original_signal, 'r+')
plot(filtered_signal, 'b+')
plt.show()
