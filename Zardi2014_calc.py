import numpy as np
import matplotlib.pyplot as plt
import itertools
from Zardi2014_def import WaveResolutions  # クラスをインポート

# パラメータのリストを用意
alpha_deg_values = [30]
Theta_values = [5]
theta_0_values = [288]
gamma_values = [3.e-3]
omega_values = [7.28e-5]
K_values = [3]
omega_t_values = [np.pi/4]
psi_values = [0]

# 全組み合わせを生成
param_combinations = itertools.product(
    alpha_deg_values, 
    Theta_values, 
    theta_0_values, 
    gamma_values, 
    omega_values, 
    K_values, 
    omega_t_values, 
    psi_values
)

# 各組み合わせで計算
results = []
n_values = np.linspace(0, 200, 1000)

for combination in param_combinations:
    params = {
        'alpha_deg': combination[0],
        'Theta': combination[1],
        'theta_0': combination[2],
        'gamma': combination[3],
        'omega': combination[4],
        'K': combination[5],
        'omega_t': combination[6],
        'psi': combination[7],
        'n': n_values,
    }
    wave = WaveResolutions(**params)
    results.append({
        'params': params,
        'u_bar': wave.u_bar,
        'theta_bar': wave.theta_bar,
        'l_plus': wave.l_plus,
        'l_minus': wave.l_minus,
        'N': wave.N
    })

# 結果のプロット
for result in results:
    params = result['params']
    label = f"alpha={params['alpha_deg']}, Theta={params['Theta']}"
    plt.plot(result['u_bar'], n_values, color='darkblue', label='u($\omega$t={})'.format(round(params['omega_t'],2)))# result['theta_bar'], label=label)
    plt.plot(result['theta_bar'], n_values, color='red', label=r'$\theta$($\omega$t={})'.format(round(params['omega_t'],2)))

plt.legend()

yticks_label = np.array(['0', '$\pi l_-$/2', '$\pi l_-$', '$3 \pi l_- / 2$', '$2\pi l_-$'] )
yticks = [0, np.pi*result['l_minus'] / 2,  np.pi*result['l_minus'], 3/2 * np.pi*result['l_minus'], 2* np.pi*result['l_minus']]

xticks_label = np.array(['-$\Theta N $/2$\gamma$', '0', '$\Theta N $/2$\gamma$'])
xticks = [-params['Theta']/2 * result['N']/params['gamma'], 0, params['Theta']/2 * result['N']/params['gamma']]

twinx_label = [r'-$\Theta$', '0', r'$\Theta$']
twinx_ticks = [-params['Theta'], 0, params['Theta']]
             
twiny_label = ['0', '$\pi l_+$/2', '$\pi l_+$', '$3 \pi l_+ / 2$', '$2\pi l_+$']
twiny_ticks = [0, np.pi*result['l_plus'] / 2,  np.pi*result['l_plus'], 3/2 * np.pi*result['l_plus'], 2* np.pi*result['l_plus']]

plt.xticks(xticks, xticks_label)
plt.yticks(yticks, yticks_label)
plt.minorticks_on()

ax = plt.gca()
ax2 = ax.twinx()
ax2.set_yticks(twiny_ticks)
ax2.set_yticklabels(twiny_label)
ax2.minorticks_on()

secax = ax.secondary_xaxis('top')
secax.set_xticks(twinx_ticks)
secax.set_xticklabels(twinx_label)
secax.minorticks_on()

#plt.xlabel('u_bar')
plt.ylabel('n')
#plt.title('Comparison of Wave Resolutions')
plt.show()
