import numpy as np
import matplotlib.pyplot as plt
import itertools
from Zardi2014_def import WaveResolutions

# パラメータのリストを用意
alpha_deg_values = [0.41]
Theta_values = [5]
theta_0_values = [288]
gamma_values = [3.e-3]
omega_values = [7.28e-5]
K_values = [3]
omega_t_values = [0, 0.25*np.pi, 0.5*np.pi, 0.75*np.pi, np.pi, 1.25*np.pi, 1.5*np.pi, 1.75*np.pi, 2*np.pi]
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
#n_values = np.linspace(0, 200, 1000)#supercritical
#n_values = np.linspace(0, 3000, 1000)#subcritical
#n_values = np.linspace(0, 1300, 1000)#critical
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
        #'n': n_values,
    }
    wave = WaveResolutions(**params)
    results.append({
        'params': params,
        'n': wave.n,
        'const_u': wave.const_u,
        'const_theta': wave.const_theta, 
        'u_bar': wave.u_bar,
        'theta_bar': wave.theta_bar,
        'l_plus': wave.l_plus, #
        'l_minus': wave.l_minus,#
        'N': wave.N
    })


#Plot
phase_labels = ['0', '$\pi$/4', '$\pi$/2', '3$\pi$/4', '$\pi$', '5$\pi$/4', '3$\pi$/2', '7$\pi$/4', '2$\pi$']

fig, ax = plt.subplots(2,1, figsize=(15,12))
for i, result in enumerate (results):
    params = result['params']
    label = f"alpha={params['alpha_deg']}, Theta={params['Theta']}"

    ax[0].vlines(result['const_u']*i, -0.1, 2*np.pi*result['l_plus'], color='black', linestyle='--', linewidth=1)
    ax[0].plot(result['const_u']*i +result['u_bar'], result['n'], color='darkblue')#, label='u($\omega$t={})'.format(round(params['omega_t'],2)))# result['theta_bar'], label=label)
    ax[0].text(result['const_u']*i, 1.9*np.pi*result['l_plus'], phase_labels[i])

    ax[1].vlines(params['Theta']*i, -0.5, 2*np.pi*result['l_plus'], color='black', linestyle='--', linewidth=1)
    ax[1].plot(params['Theta']*i +result['theta_bar'],result['n'], color='red')#, label=r'$\theta$($\omega$t={})'.format(round(params['omega_t'],2)))
    ax[1].text(params['Theta']*i, 1.9*np.pi*result['l_plus'], phase_labels[i])


yticks_label = np.array(['0', '$\pi l_-$/2', '$\pi l_-$', '$3 \pi l_- / 2$', '$2\pi l_-$'] )
yticks = [0, np.pi*result['l_minus'] / 2,  np.pi*result['l_minus'], 3/2 * np.pi*result['l_minus'], 2* np.pi*result['l_minus']]

xticks_label = np.array(['-$\Theta N $/2$\gamma$', '0', '$\Theta N $/2$\gamma$'])
xticks = [-params['Theta']/2 * result['N']/params['gamma'], 0, params['Theta']/2 * result['N']/params['gamma']]

twinx_label = [r'-$\Theta$', '0', r'$\Theta$']
twinx_ticks = [-params['Theta'], 0, params['Theta']]
             
twiny_label = ['0', '$\pi l_+$/2', '$\pi l_+$', '$3 \pi l_+ / 2$', '$2\pi l_+$']
twiny_ticks = [0, np.pi*result['l_plus'] / 2,  np.pi*result['l_plus'], 3/2 * np.pi*result['l_plus'], 2* np.pi*result['l_plus']]

ax[0].set_title('u')
ax[0].set_xticks(xticks, xticks_label)

ax[1].set_title(r'$\theta$')
ax[1].set_xticks(twinx_ticks, twinx_label)
for k in range(2):
    ax[k].set_yticks(yticks, yticks_label)
    ax[k].set_ylabel('n')
    ax[k].twinx().set_yticks(twiny_ticks, twiny_label)
    #ax[k].twinx().minorticks_on()
    #ax[k].minorticks_on()

plt.show()
