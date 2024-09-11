import numpy as np
from scipy.special import erfc

class WaveResolutions:
    def __init__(self, alpha_deg, Theta, theta_0, gamma, omega, K, omega_t, psi=0):
        self.alpha_deg = alpha_deg
        self.Theta = Theta
        self.theta_0 = theta_0
        self.gamma = gamma
        self.omega = omega
        self.K = K
        #self.n = n
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

        self.const_u = self.Theta * self.N / (2 * self.gamma)
        self.const_theta = self.Theta / 2

        self.n, self.u_bar, self.theta_bar = self.resolutions()

    def resolutions(self):
        psi = self.psi
        
        if abs(self.omega - self.N_alpha) < 1.e-6:#self.N_alpha == self.omega: #Critical
            self.n = np.linspace(0,2.*np.pi*self.l_plus, 1000)
            self.omega_plus, self.omega_minus = 2*self.N_alpha, 0
            self.l_plus, self.l_minus = np.sqrt(self.K/self.N_alpha), 0
            self.eta = self.n / 2 / np.sqrt(self.K*self.omega_t/self.omega)
            

            print('Critical')
            return self.n,  self.const_u * ( (np.exp(-self.n / self.l_plus) * np.cos(self.omega_t - self.n / self.l_plus + psi)) - erfc(self.eta)*np.cos(self.omega_t + psi) ),self.const_theta * ( (np.exp(-self.n / self.l_plus) * np.sin(self.omega_t - self.n / self.l_plus + psi)) + erfc(self.eta)*np.sin(self.omega_t + psi) )
        
        elif self.N_alpha < self.omega: #SubCritical
            self.omega_minus = abs(self.omega_minus)
            self.l_minus = np.sqrt(2 * self.K / self.omega_minus)
            self.n = np.linspace(0, 2.*np.pi*self.l_plus, 1000)
            print('SubCritical')

            return self.n, self.const_u * ( (np.exp(-self.n / self.l_plus) * np.cos(self.omega_t - self.n / self.l_plus + psi))-
                              (np.exp(-self.n / self.l_minus) * np.cos(self.omega_t - self.n / self.l_minus + psi)) ),self.const_theta * ( (np.exp(-self.n / self.l_plus) * np.sin(self.omega_t - self.n / self.l_plus + psi)) + (np.exp(-self.n / self.l_minus) * np.sin(self.omega_t - self.n / self.l_minus + psi)) )

        else: #SuperCritical
            self.n = np.linspace(0,2.*np.pi*self.l_plus, 1000)
            print('SuperCritical')
            return  self.n, (self.const_u * ( (np.exp(-self.n / self.l_plus) * np.cos(self.omega_t - self.n / self.l_plus + psi))- (np.exp(-self.n / self.l_minus) * np.cos(self.omega_t + self.n / self.l_minus + psi) ))),self.const_theta * ( (np.exp(-self.n / self.l_plus) * np.sin(self.omega_t - self.n / self.l_plus + psi))+(np.exp(-self.n / self.l_minus) * np.sin(self.omega_t + self.n / self.l_minus + psi)) )
