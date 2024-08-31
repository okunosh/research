import numpy as np

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

        self.u_bar = self.compute_u()
        self.theta_bar = self.compute_theta()

    def compute_u(self):
        const = self.Theta * self.N / (2 * self.gamma)
        phi = self.psi
        return const * (
            (np.exp(-self.n / self.l_plus) * np.cos(self.omega_t - self.n / self.l_plus + phi))
            - (np.exp(-self.n / self.l_minus) * np.cos(self.omega_t + self.n / self.l_minus + phi))
        )

    def compute_theta(self):
        const = self.Theta / 2
        phi = self.psi
        return const * (
            np.exp(-self.n / self.l_plus) * np.sin(self.omega_t - self.n / self.l_plus + phi)
            + np.exp(-self.n / self.l_minus) * np.sin(self.omega_t + self.n / self.l_minus + phi)
        )
