import numpy as np
import xarray as xr
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_to_netcdf import DatasetToNetcdf

#depends on  directory structure-----------------
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
target_dir = os.path.join(parent_dir, 'analytical')
sys.path.append(target_dir)
sys.path.insert(0, target_dir)
#print(sys.path)
from Zardi2014_def import WaveResolutions
from ql_plot import process_netcdf_directory
from boundary_condition_theta import surface_theta_0
#-----------------------------------------------

class NumericalDatasetToNetcdf(DatasetToNetcdf):
    @staticmethod
    def make_dataset(sim):
        sim_data = {
            "u_bar": sim.u_bar,
            "theta_bar": sim.theta_bar,
            "K": sim.K,  #need to be changed!!
            "alpha": sim.alpha_deg,
            "theta_0": sim.theta_0,
            "Theta": sim.Theta,
            "gamma": sim.gamma,
            "N": sim.N,
            "N_alpha": sim.N_alpha,
            "omega": sim.omega,
            "omega_plus": sim.omega_plus,
            "omega_minus": sim.omega_minus,
            "l_plus": sim.l_plus,
            "l_minus": sim.l_minus,
            "altitude": sim.n,
            "time": np.array([sim.time]),
            "planet": sim.planet,
            "flow_regime": sim.resolutions()["regime"]
        }
        ds = DatasetToNetcdf.make_dataset(sim_data)
        ds = ds.assign_attrs(title="Numerical solution", flow_regime=sim_data["flow_regime"])
        #change attributes order
        existing_attrs = ds.attrs
        ordered_keys = ["title", "flow_regime", "planet", "history", "reference"]
        ordered_attrs = {key: existing_attrs[key] for key in ordered_keys if key in existing_attrs}
        ds.attrs = ordered_attrs
        return ds

    def make_new_dir_path(self, output_path, dt):
        new_dir_path = f"{output_path}/{self.planet}/{self.now}_{self.flow_regime}_num{self.num}_alp{self.alpha}_gamma{self.gamma}_dt{dt}"
        return new_dir_path

    
    def save_to_netcdf(self, new_dir_path):
        kind = "N"
        output_file = f"{new_dir_path}/{kind}_{self.now}_{self.flow_regime}_t{self.time}.nc"
        #print(output_file)
        self.ds.to_netcdf(output_file)
        
    
class Simulation(WaveResolutions):

    def __init__(self, g, alpha_deg, Theta, theta_0, gamma, omega, K, num, dt):
        super().__init__(g, alpha_deg, Theta, theta_0, gamma, omega, K, None, num)
        self.dt = dt
        self.dn = 2* np.pi * self.l_plus / self.num


        self.mat_num = self.num+1
        #self.A need to be changed!!
        #self.A = self.K * self.dt / self.dn**2
        self.B = self.dt * self.N**2 * np.sin(self.alpha) / self.gamma
        self.C = -self.dt * self.gamma * np.sin(self.alpha)

        self.w= self.make_w_init()
       # self.matrix = self.make_block_matrix() #change!
        #time--------------
        self.day = 4
        self.day2sec = 24 * 3600
        self.t_fin = self.day * self.day2sec 
        self.times = np.arange(0, self.t_fin+1, self.dt)
        #-----------------

        
    def make_w_init(self):
        return np.zeros((self.mat_num)*2).reshape((self.mat_num)*2,1)

    #block matrix
    def updateK(self, K): #K: column vector whose length is equal to self.mat_num
        coef = self.dt / self.dn**2
        E = np.ones_like(K)
        A = coef * K
        AA = E - 2*coef*K
        return A, AA
        
    def make_off_diagonal(self, val):
        a = np.diag([val]*self.mat_num)
        a[0,0], a[self.mat_num-1, self.mat_num-1] = 0, 0
        return a

    def make_1_1(self, A, AA):
        a = np.zeros((self.mat_num, self.mat_num))
        for i in range(1, self.num-1):
            a[i,i+1] = A[i]
            a[i+1,i] = A[i]
            a[i,i] = AA[i]
        #np.fill_diagonal(a, AA)#
        a[0,0], a[self.mat_num-1,self.mat_num-1] = 0, 0
        return a

    def make_2_2(self, A, AA):
        a = self.make_1_1(A, AA)
        a[1,0] = A[1]
        return a

    def update_block_matrix(self, KK):
        A, AA = self.updateK(KK)
        a = self.make_1_1(A, AA)
        b = self.make_off_diagonal(self.B)
        c = self.make_off_diagonal(self.C)
        d = self.make_2_2(A, AA)
        
        block_matrix = np.block([
            [a,b],
            [c,d]
            ])
        """
        print("a:", a)
        print("b:", b)
        print("c:", c)
        print("d:", d)
        print(A, AA)
        """
        return block_matrix
    
    def run_simulation(self, output_path, dt):
        surf_con = xr.open_dataset("TestGroundTheta0.nc")
        surface_theta = surf_con.theta_0.values
        K_ds = xr.open_dataset("K/constant_100.nc")
        K_vec = K_ds.K.values.reshape(K_ds.K.values.shape[1], K_ds.K.values.shape[0])
        
        for m, t in enumerate(self.times):
            num = m%(24*3600*10)
            K_ = K_vec[num]
            #print(K_, K_.shape)
            #print(self.dn, self.l_plus, self.N_alpha)
            matrix = self.update_block_matrix(K_)
            #input("stop")
            self.w = matrix @ self.w
            #self.w[self.num+1] = self.Theta*np.sin(self.omega * t)
            #self.w[self.num+1] = surface_theta_0(self.Theta, self.omega, t)
            self.w[self.num+1] = surface_theta[num]
                
            if t%3600 == 0:
                self.time = t
                self.u_bar = self.w[:self.num +1 ]
                self.theta_bar = self.w[self.num + 1:]
                ds = NumericalDatasetToNetcdf.make_dataset(self)
                data = NumericalDatasetToNetcdf(ds)
                if m==0:
                    new_dir_path = data.make_new_dir_path(output_path, dt)
                    data.make_new_dir(new_dir_path)
                data.save_to_netcdf(new_dir_path)
        process_netcdf_directory(new_dir_path)
