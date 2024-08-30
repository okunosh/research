import matplotlib.pyplot as plt
import numpy as np
from importlib import reload
import st_wind
reload(st_wind)

#Earth
etheta0, eg, eK, egamma = 293, 9.81, 10, 0.004
alpha1 = 0.5
theta_s = 20

D1, zp1, u1, theta1 = st_wind.slopewind(etheta0, eg, eK, egamma, alpha1, theta_s)

#Mars
mtheta0, mg, mK, mgamma = 210, 3.72, 10, 0.004

D2, zp2, u2, theta2 = st_wind.slopewind(mtheta0, mg, mK, mgamma, alpha1, theta_s)

#figure
mars, earth = 'Mars', 'Earth'
fig, ax = plt.subplots(1,2, figsize=(12, 6))
ax[0].plot(u1, zp1, color='blue', label=earth)
ax[0].plot(u2, zp2, color='red', label=mars)
ax[0].set_xlabel('u')
ax[0].set_ylabel('z')
ax[0].set_title('wind speed [m/s]')

ax[1].plot(theta1, zp1, color='blue', label=earth)
ax[1].plot(theta2, zp2, color='red', label=mars)
ax[1].set_xlabel(r'$\theta$')
ax[1].set_ylabel('z')
ax[1].set_title('potential temperature [K]')

plt.suptitle(r'$\alpha$ '+'= '+ str(alpha1)+'$\degree$')
for k in range(2):
    ax[k].grid()
    ax[k].legend()

plt.show()
