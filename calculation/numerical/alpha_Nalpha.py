import numpy as np
import matplotlib.pyplot as plt

g = 3.72#9.81
gamma = 4.5e-3#3.e-3
theta_0 = 210#288
N = np.sqrt(gamma * g / theta_0)

deg2rad = np.pi / 180
alpha = np.linspace(0, 90, 1000)

N_alpha = N * np.sin(alpha*deg2rad)

T = 24 * 3600
omega = 2*np.pi / T

print("N:"+str(N))
plt.plot(alpha, N_alpha, label=r'$N_\alpha$')
plt.hlines(omega, -10, 90, color='red', linestyle='--', label='$\omega$')
plt.yscale('log')
plt.xlabel('slope angle [degree]')
plt.ylabel(r'frequency [1/s]')
plt.xlim(-2, 90)
plt.grid()
plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0, fontsize=12)
plt.show()
