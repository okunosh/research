import numpy as np
import matplotlib.pyplot as plt

n = np.linspace(0, 2000, 10000)

#theta = np.exp(n / 10) + 192

#plt.plot(n, theta)

theta = np.linspace(190, 230, 1000)
#h = 10 * np.log(theta -192)

h14 = 10*np.arctan(theta-214)

plt.plot(theta, h14)
plt.ylim(0,7)
plt.grid()
plt.show()
