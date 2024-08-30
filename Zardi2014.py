import matplotlib.pyplot as plt
import numpy as np
import math
import copy

#Zardi et al. 2014
deg2rad = np.pi / 180
alpha_deg = 30
alpha = alpha_deg * deg2rad

g = 9.81
Theta = 5
theta_0 = 288
gamma = 0.003 #K/m

omega = 7.28e-5 #1/s

K = 3 #m-2/s

N = np.sqrt( gamma * g / theta_0 )
N_alpha = N * np.sin( alpha ) 

phi = 0


omega_plus = N_alpha + omega
omega_minus = N_alpha - omega

l_plus = np.sqrt( 2*K /omega_plus)
l_minus =  np.sqrt( 2*K /omega_minus)

def u(n, omega_t, phi=0):
    const = Theta*N/2/gamma

    return const * ( (np.exp(-n/l_plus) * np.cos(omega_t - n/l_plus + phi )) - (np.exp( -n/l_minus ) *  np.cos(omega_t + n/l_minus + phi )) )


def theta(n, omega_t, phi=0):
    const = Theta/2

    return const * ( np.exp(-n/l_plus) * np.sin(omega_t - n/l_plus + phi ) + np.exp( -n/l_minus ) *  np.sin(omega_t + n/l_minus + phi ) )


n = np.linspace( 0, 200, 1000 )# * np.pi * l_minus
n_ = np.linspace( 0, 200, 1000 )# * np.pi * l_plus

omega_t=np.pi/4
u_bar = u(n, omega_t )
u_label= '$\overline{u}$ ($\omega$t='+str(round(omega_t,2))+')'#.format(omega_t)

theta_label= r"$\overline{\theta}$"+' ($\omega$t={}) '.format(round(omega_t,2))
plt.plot(u_bar,n, label= u_label, color='darkblue')
plt.vlines(0, -0.2, 2* np.pi* l_minus,linestyle='--',color='black')
xticks = np.array(['-$\Theta N $/2$\gamma$', '0', '$\Theta N $/2$\gamma$']) 
yticks = np.array(['0', '$\pi l_-$/2', '$\pi l_-$', '$3 \pi l_- / 2$', '$2\pi l_-$'] )
plt.xticks([-Theta/2 * N/gamma, 0, Theta/2 * N/gamma], xticks)
plt.yticks([0, np.pi*l_minus / 2,  np.pi*l_minus, 3/2 * np.pi*l_minus, 2* np.pi*l_minus], yticks)
plt.minorticks_on()

theta_bar = theta(n, omega_t)
plt.plot(theta_bar, n, label= theta_label, color='red')
plt.legend()
ax = plt.gca()
ax2 = ax.twinx()
secax = ax.secondary_xaxis('top')
secax.set_xticks([-Theta, 0, Theta])
secax.set_xticklabels([r'-$\Theta$', '0', r'$\Theta$'])
secax.minorticks_on()

ax2.set_yticks([0, np.pi*l_plus / 2,  np.pi*l_plus, 3/2 * np.pi*l_plus, 2* np.pi*l_plus])
ax2.set_yticklabels(['0', '$\pi l_+$/2', '$\pi l_+$', '$3 \pi l_+ / 2$', '$2\pi l_+$'])
ax2.minorticks_on()
#plt.xlabel('u')
plt.ylabel('n')
plt.xlim(-10,10)
#plt.grid()

plt.show()

print('--------------')
print( 'pi l_- = {} '.format(np.pi * l_minus ))
print( 'Theta N /2 gamma = {}'.format(Theta/2 * N/gamma))
