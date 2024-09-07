import numpy as np
from scipy.special import erfc

class WaveResolutions:
    def __init__(self, alpha_deg, Theta, theta_0, gamma, omega, K, n, omega_t, psi=0):
        self.alpha_deg = alpha_deg
        self.Theta = Theta
        self.theta_0 = theta_0
        self.gamma = gamma
        self.omega = omega
        self.K = K
        self.n = n
        self.omega_t = omega_t
        self.psi = psi

        self.deg2rad = np.pi / 180
        self.alpha = alpha_deg * self.deg2rad
        self.g = 9.81

        self.N = np.sqrt(gamma * self.g / theta_0)
        self.N_alpha = self.N * np.sin(self.alpha)
        self.omega_plus = self.N_alpha + omega
        self.omega_minus = self.N_alpha - omega
        self.l_plus = np.sqrt(2 * K / self.omega_plus)
        self.l_minus = np.sqrt(2 * K / self.omega_minus)

        self.u_bar, self.theta_bar = self.resolutions()

    def resolutions(self):
        const_u = self.Theta * self.N / (2 * self.gamma)
        const_theta = self.Theta / 2
        psi = self.psi
        
        if self.N_alpha == self.omega: #Critical
            omega_plus, omega_minus = 2*N_alpha, 0
            l_plus = np.sqrt(K/N_alpha)
            eta = self.n / 2 / np.sqrt(self.K*self.omega_t/t)

            print('Critical')
            return  const_u * ( (np.exp(-self.n / l_plus) * np.cos(self.omega_t - self.n / l_plus + psi)) - erfc(eta)*np.cos(self.omega_t + psi) ),
        const_theta * ( (np.exp(-self.n / l_plus) * np.sin(self.omega_t - self.n / l_plus + psi)) + erfc(eta)*np.sin(self.omega_t + psi) )
        
        elif self.N_alpha < self.omega: #SubCritical
            l_minus = abs(l_minus)
            print('SubCritical')

            return const_u * ( (np.exp(-self.n / self.l_plus) * np.cos(self.omega_t - self.n / self.l_plus + psi))-
                              (np.exp(-self.n / l_minus) * np.cos(self.omega_t + self.n / l_minus + psi)) ),
        const_theta * ( (np.exp(-self.n / self.l_plus) * np.sin(self.omega_t - self.n / self.l_plus + psi))+
                              (np.exp(-self.n / l_minus) * np.sin(self.omega_t + self.n / l_minus + psi)) )
        
        else: #SuperCritical
            print('SuperCritical')
            return  const_u * ( (np.exp(-self.n / self.l_plus) * np.cos(self.omega_t - self.n / self.l_plus + psi))
                - (np.exp(-self.n / self.l_minus) * np.cos(self.omega_t + self.n / self.l_minus + psi) )
            ),
         const_theta * ( (np.exp(-self.n / self.l_plus) * np.sin(self.omega_t - self.n / self.l_plus + psi))+
                              (np.exp(-self.n / self.l_minus) * np.sin(self.omega_t + self.n / self.l_minus + psi)) )
