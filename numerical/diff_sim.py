import numpy as np
import xarray as xr
from datetime import datetime
import os
import sys
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_to_netcdf import DatasetToNetcdf

# today = datetime.datetime.today() 
# date = today.date()


# class NumericalDatasetToNetcdf(DatasetToNetcdf):
#     @staticmethod
#     def make_Dataset(wave)
#     ds = DatasetToNetcdf.make_Dataset(wave)


#parameters-------------------------------
K = 3.

alpha_deg = 30.
deg2rad = np.pi/180
alpha = alpha_deg * deg2rad

#T = 88445 #sec
omega = 7.28e-5 #7.08e-5#2*np.pi / T
gamma = 3.e-3
theta_0 = 288
Theta = 5
g = 9.81 #3.72
N = np.sqrt(gamma*g/theta_0)
N_alpha = N * np.sin(alpha)

omega_plus = N_alpha + omega
omega_minus = N_alpha - omega

l_plus = np.sqrt(2.*K / omega_plus)
l_minus = np.sqrt(2.*K/ omega_minus)

hour = 24
t_fin =3600*hour + 1

num = 700#
dt = 1.e-3#-8
dn = 2*np.pi*l_plus / num #3.e-1

nn = np.linspace(0,2.*np.pi*l_plus,num+1) #space
time = np.arange(0,t_fin,dt) #time

A = K * dt / dn**2
B = dt * N**2 * np.sin(alpha)  /gamma
C = -dt * gamma * np.sin(alpha)

def make_diag(A, n): #(1,2) & (2,1) in block matrix
    l = n+1
    a = np.diag([A]*l) #num
    a[0,0], a[l-1,l-1] = 0,0 #[0,0], [num-1, num-1]
    return a

def make_1_1(A, n): #(1,1) in block matrix
    l = n+1
    a = np.zeros((l, l)) #(num, num)
    for i in range(1,n-1): #??(1, n-2)
        a[i,i+1] = A 
        a[i+1,i] = A
    np.fill_diagonal(a, 1-2*A)
    a[0,0], a[l-1,l-1] = 0, 0 #
    return a

def make2_2(A, n):
    a = make_1_1(A, n)
    a[1,0] = A 
    return a


def format_number(number):
    return f"{number:.3e}"

a = make_1_1(A, num)
b = make_diag(B, num)
c = make_diag(C, num)
d = make2_2(A, num)


matrix = np.block([
    [a,b],
    [c,d]
])


format_dt = format_number(round(dt,3))
format_dn = format_number(round(dn,3))

#make new directory
new_dir_path = f"results/{datetime.now().strftime('%Y%m%d_%H%M')}/"
os.makedirs(new_dir_path, exist_ok=True)

#make text file whose name displays the dt, dn, K  of the calculation       
text_file_name = "dt={}_dn={}_K={}.txt".format(format_dt, format_dn, K)
text_file_path = os.path.join(new_dir_path, text_file_name)

with open(text_file_path, 'w') as file:
    file.write(f"dt={dt}\ndn={dn}\nK={K}\nnumber of grid points(alt):{num}")

#calculation-----------------------------------
init = np.zeros((num+1)*2).reshape((num+1)*2,1)
w = init

for m, t in enumerate(time):
    #print(t, m)
    #matrix[num,num] = Theta*np.sin(omega*t)
    w = matrix@w
    #print(f"Boundary condition set: w[{num+1}] = {w[num+1]}")
    w[num+1] = Theta*np.sin(omega*t)
    #print(f"After matrix multiplication: w[{num+1}] = {w[num+1]}")
    
     
    if t%3600==0:
        u = w[:num+1]
        theta = w[num+1:]
        
        ds = xr.Dataset(
            data_vars=dict(
                u_bar=(["altitude", "time"], u, {"unit":"m/s", "description":"along slope velocity"}),
                theta_bar = (["altitude", "time"], theta, {"unit": "kelvin", "description":"mean potential temperature anomaly"}),
                K = (["time"], np.array([K]), {"unit": "m^2 s^-1", "description":"diffusion coefficient"}),
                dt = (["time"], np.array([dt])),
                dn = (["time"], np.array([dn])),
                alpha = (["time"], np.array([alpha_deg]), {"unit": "degree", "description":"slope angle"}),
                theta_0 = (["time"], np.array([theta_0]), {"description":""}),
                gamma = (["time"], np.array([gamma]), {"description":"vertical gradient of potential temperature"}), 
                N = (["time"], np.array([N]), {"description": "Bulant-Visala"}),
                N_alpha = (["time"], np.array([N_alpha]),{"description":"Bulant-Visala along slope"}),
                omega = (["time"], np.array([omega]), {"description":"angular frequency"}),
                omega_plus = (["time"], np.array([omega_plus]), {"description":""}),
                omega_minus = (["time"], np.array([omega_minus]), {"description":""}),
                l_plus = (["time"], np.array([l_plus]), {"description":""}),
                l_minus = (["time"], np.array([l_minus]), {"description":""})
                ),
            coords = dict(
                altitude = ("altitude", nn),
                time = ("time", np.array([t]))
                ),
            attrs = dict(title="differensial simulation",
                         history=f"Created on {datetime.now().isoformat()}"
                ),
            )
        #input("stop")
        
        #save as netcdf
        f_name = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}_t{int(t)}.nc"
        file_name = new_dir_path + f_name

        ds.to_netcdf(file_name)


