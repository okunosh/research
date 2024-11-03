import numpy as np
import xarray as xr
import os
import sys

Class Simulation:
    def __init__(self, g, alpha_deg, Theta, theta_0, gamma, omega, K, num, dt):
        super().__init__(g, alpha_deg, Theta, theta_0, gamma, omega, K, None, num)
        self.dt = dt
        self.dn = 2* np.pi * self.l_plus / self.num

        self.mat_num = num+1
        self.A = self.K * self.dt / self.dn**2
        self.B = self.dt * self.N**2 * mp.sin(self.alpha) / self.gamma
        self.C = -self.dt * self.gamma * np.sin(self.alpha)

        self.w_init = self.make_w_init()
        self.matrix = self.make_block_matrix()
        self.hour = 24
        self.t_fin = 3600 * self.hour
        self.t = np.arange(0, self.t_fin+1, self.dt)
        
        
    def make_w_init(self):
        return np.zeros((self.mat_num)*2).reshape((self.mat_num)*2,1)

    #matrix
    def make_off_diagonal(self, val):
        a = np.diag([val]*self.mat_num)
        a[0,0], a[self.mat_num-1, self.mat_num-1] = 0, 0
        return a

    def make_1_1(self):
        a = np.zeros((self.mat_num)*2)
        for i in range(1, self.mat_num-1):
            a[i,i+1] = self.A
            a[i+1,i] = self.A
        np.fill_diagonal(a, 1-2*self.A)
        a[0,0], a[self.mat_num-1,self.mat_num-1] = 0, 0
        return a

    def make_2_2(self):
        a = make_1_1()
        a[1,0] = self.A
        return a

    def make_block_matrix(self):
        a = self.make_1_1()
        b = self.make_off_diagonal(self.B)
        c = self.make_off_diagonal(self.C)
        d = self.make_2_2()
        
        block_matrix = np.block([
            [a,b],
            [c,d]
            ])
        return block_matrix

    def run_simulation(self, time):
        for m, t in enumerate(time):
            w = self.matrix @ seljf.w
            w[self.num+1] = self.Thelta*np.sin(self.omega * t)
