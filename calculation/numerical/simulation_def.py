import numpy as np
import xarray as xr
import os
import sys
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_to_netcdf import DatasetToNetcdf

#depends on  directory structure-----------------
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
target_dir = os.path.join(parent_dir, 'analytical')
sys.path.append(target_dir)
sys.path.insert(0, target_dir)
#print(sys.path)
from Zardi2014_def import WaveResolutions
from ql_plot import process_netcdf_directory, process_netcdf_directory_scatter
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
            "Theta": sim.Theta,#No use. change to "surface_temperature": sim.surface_temp
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
            "dn": np.array([sim.dn]),
            "planet": sim.planet,
            "flow_regime": sim.resolutions()["regime"]
        }
        ds = DatasetToNetcdf.make_dataset(sim_data)
        ds["surface_forcing"] = (["time"], np.array([sim.surface_temp]), {"description": "file path of surface forcing"})
        ds["K_file"] = (["time"], np.array([sim.K_file]), {"description": "file path of K distribution"})
        ds["dn"] = (["time"], np.array([sim.dn]), {"description": "the number of grids for spatioal dirction"})
        
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

day2sec = 24 * 3600
    
class Simulation(WaveResolutions):

    def __init__(self, g, alpha_deg, Theta, theta_0, gamma, omega, K, num, dt, K_file, surface_temp):
        ds = xr.open_dataset(K_file)
        K_val = ds.K.values[:,0][0] #空間方向に一様な場合は最初の値だけを採用する。一様でない場合は使えない。
        
        super().__init__(g, alpha_deg, Theta, theta_0, gamma, omega, K_val, None, num)
        self.dt = dt
        self.dn = 2* np.pi * self.l_plus / self.num
        self.mat_num = self.num+1

        #temporal and spatial gamma
        self.gamma_file = "gamma/gamma_const_4.020e-03.nc"
        ds_gamma = xr.open_dataset(self.gamma_file)
        self.gamma_zt = ds_gamma["gamma"].values
        #-----------------
        
        ##will be changed!
        self.B = self.dt * self.N**2 * np.sin(self.alpha) / self.gamma
        self.C = -self.dt * self.gamma * np.sin(self.alpha)
        #-----------------

        self.w= self.make_w_init()
        #self.matrix = self.make_block_matrix() #change!
        #time--------------
        self.day = 4
        self.t_fin = self.day * day2sec 
        #self.times = np.arange(0, self.t_fin+1, self.dt)
        self.times = np.linspace(0, self.t_fin, int(self.t_fin/self.dt)+1)
        #-----------------
        self.K_file = K_file
        self.surface_temp = surface_temp

        
    def make_w_init(self):
        return np.zeros((self.mat_num)*2).reshape((self.mat_num)*2,1)

    #block matrix
    def updateK(self, K): #K: raw vector whose length is equal to self.mat_num
        coeff = self.dt *  self.num**2 * self.omega_plus / 8/ np.pi**2
        E = np.ones_like(K)
        A = np.ones_like(K)*coeff
        AA = E - 2*A
        return A, AA
    
    def make_off_diagonal(self, vec):
        #a = np.diag([val]*self.mat_num)
        a = np.array(np.diag(vec))
        a[0,0], a[self.mat_num-1, self.mat_num-1] = 0, 0
        #print(a)
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

    def update_block_matrix(self, KK, B, C):
        A, AA = self.updateK(KK)
        a = self.make_1_1(A, AA)
        b = self.make_off_diagonal(B)
        c = self.make_off_diagonal(C)
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
        #print("block matrix:", np.shape(block_matrix))
        return block_matrix
    
    def run_simulation(self, output_path, dt):
        surf_con = xr.open_dataset(self.surface_temp)
        surface_theta = surf_con.theta_0.values
        #print(datetime.now().time())
        K_ds = xr.open_dataset(self.K_file)
        K_vec = K_ds.K.values#.reshape(K_ds.K.values.shape[1], K_ds.K.values.shape[0])
        
        for m, t in enumerate(self.times):
            num = m%(24*3600*10)
            K_ = K_vec[:,num]
            self.K = K_[0]
            self.l_plus = np.sqrt(2 * self.K / self.omega_plus)
            self.l_minus = np.sqrt(2 * self.K / self.omega_minus)
            self.n = np.linspace(0, 2.*np.pi*self.l_plus, self.num+1)

            #gamma
            gamma_profile = self.gamma_zt[:, num]
            gamma_profile = np.clip(gamma_profile, 1e-4, None)
            N = np.sqrt((self.g / self.theta_0) * gamma_profile)
            B = self.dt * N**2 * np.sin(self.alpha)/gamma_profile
            C = -self.dt * gamma_profile * np.sin(self.alpha)

            matrix = self.update_block_matrix(K_, B ,C)
            self.dn =  2* np.pi * self.l_plus / self.num
            self.w = matrix @ self.w
            self.w[self.num+1] = surface_theta[num]

            """
            print("gamma_profile: ", gamma_profile,
                  gamma_profile.shape,
                  np.max(gamma_profile),
                  np.min(gamma_profile)
            )
            print("self.N: ", self.N)
            print("N: ", N)
            print("B: ", B)
            print("self.B: ", self.B)
            print("C: ", C)
            print("self.C :", self.C)
            #input("stop")
            print("-----")
            """
             
            if t%3600 == 0:
                #update parameters
                self.time = t
                self.u_bar = self.w[:self.num +1 ]
                self.theta_bar = self.w[self.num + 1:]
                
                ds = NumericalDatasetToNetcdf.make_dataset(self)
                data = NumericalDatasetToNetcdf(ds)
                if m==0:
                    new_dir_path = data.make_new_dir_path(output_path, dt)
                    data.make_new_dir(new_dir_path)
                    #condition manager!
                data.save_to_netcdf(new_dir_path)
                #print("finished:"+str(t), datetime.now().time())
        process_netcdf_directory_scatter(new_dir_path, confirm=True, save=True)
        #print(datetime.now().time())
