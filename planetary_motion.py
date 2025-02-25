import numpy as np
import matplotlib.pyplot as plt

G = 6.67430e-11

def planet2D(x_0, y_0, vx_0, vy_0, Mc, m, dt, N):
    r = np.zeros((N, 3))  
    t = np.zeros(N)       

    r[0] = [x_0, y_0, 0]  
    v = np.array([vx_0, vy_0, 0])  
    t[0] = 0  

    for i in range(1, N):
        r_norm = np.linalg.norm(r[i - 1])
        a = -G * Mc * r[i - 1] / r_norm**3

        r[i] = r[i - 1] + v * dt + 0.5 * a * dt**2

        r_norm_new = np.linalg.norm(r[i])
        a_new = -G * Mc * r[i] / r_norm_new**3

        v += 0.5 * (a + a_new) * dt

        t[i] = t[i - 1] + dt

    return r, t

x_0 = 1.0e11 
y_0 = 0.0     
vx_0 = 0.0    
vy_0 = 3.0e4  
Mc = 1.989e30  
m = 5.972e24   
dt = 86400     
N = 1000      


r, t = planet2D(x_0, y_0, vx_0, vy_0, Mc, m, dt, N)

def area(R, t, T_0, tau):
    idx_start = np.searchsorted(t, T_0)
    idx_end = np.searchsorted(t, T_0 + tau)

    R_segment = R[idx_start:idx_end]
    t_segment = t[idx_start:idx_end]

    total_area = 0

    for i in range(1, len(R_segment)):
        r1 = R_segment[i - 1]
        r2 = R_segment[i]
        cos_theta = np.dot(r1,r2)/(np.linalg.norm(r1)*np.linalg.norm(r2))
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = np.arccos(cos_theta)
        area_segment = 0.5 * np.linalg.norm(r1)**2 * theta
        total_area += area_segment
    return total_area

def angular_momentum(R, V, m):
    return np.cross(R, V) * m
T_0_vals = np.linspace(0, t[-1] - 86400, 10)  
tau_const = 86400  

S_T0_tau = np.array([area(r, t, T_0, tau_const) for T_0 in T_0_vals])

plt.figure(figsize=(10, 6))
plt.plot(T_0_vals, S_T0_tau, label=r'$S(T_0, \tau=const)$')
plt.xlabel(r'$T_0$')
plt.ylabel('S')
plt.title('tau=const')
plt.grid(True)
plt.legend()
plt.show()

T_0_fixed = 0  
tau_vals = np.linspace(1, 10 * 86400, 10)  

S_T0_const_tau = np.array([area(r, t, T_0_fixed, tau) for tau in tau_vals])

plt.figure(figsize=(10, 6))
plt.plot(tau_vals, S_T0_const_tau, label=r'$S(T_0=const, \tau)$')
plt.xlabel(r'$\tau$')
plt.ylabel('S')
plt.title('T_0=const')
plt.grid(True)
plt.legend()
plt.show()


angular_moments = np.array([angular_momentum(r[i], np.array([vx_0, vy_0, 0]), m) for i in range(N)])

plt.figure(figsize=(10, 6))
plt.plot(t, angular_moments[:, 2], label="Момент импульса", color='blue')
plt.xlabel('Время')
plt.ylabel('Момент импульса (J·s)')
plt.grid(True)
plt.legend()
plt.show()

momentum_error = np.abs(angular_moments[:, 2] - angular_moments[0, 2]) 
plt.figure(figsize=(10, 6))
plt.plot(t, momentum_error, label='Ошибка сохранения момента импульса', color='red')
plt.xlabel('Время')
plt.ylabel('Ошибка')
plt.grid(True)
plt.legend()
plt.show()
