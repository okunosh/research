import numpy as np
import matplotlib.pyplot as plt

day = 4
hour = 24
hour2sec = 3600

t = np.arange(0, day * hour * hour2sec +1, 0.1)
def ft(t):
    t = (t/hour2sec)%24
    if 6<t<=13:
        return 220 + 40 * np.cos(2*np.pi *(t-13)/14)
    elif 13<t:
        #return 180 + 80 *(0.5*np.cos(2*np.pi*(t-13)/22)+0.5)**0.5
        return 220 + 40 *(np.cos(2*np.pi*(t-13)/22))
    else:
        return 180
    
y = np.zeros(len(t))*np.nan
for i in range(len(y)):
    y[i] = ft(t[i])

hour_ticks = np.linspace(t[0], t[-1], num=int(day*hour/12) +1)  # 9点（0, 6, 12, ..., 48 時間）
tick_labels = [str(int(tick // 3600)) for tick in hour_ticks]  # 秒をhourに変換
                                
plt.plot(t, y)
plt.grid()
plt.xticks(hour_ticks, tick_labels)
plt.xlabel('hour')
plt.ylabel(r'$\theta$')
plt.title(r'$\theta$(0,t)')
plt.show()
