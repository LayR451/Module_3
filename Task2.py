## Pharmacokinetics of Cavalitide. First, review other code.
# Then, replace Euler's 
# adapt to incoporate simulations of a daily cavalitde and plot over time 
# draw conclusion about optimal dosing 
# adjust dosing until you have steady state blood concentration near .5

# scipy.integrate.solve_ivp

# ncreased absorption rate ka = 0.006 1/h





# CODE TIME: 
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
# ===== choose product: 'Ozempic' or 'Wegovy' =====
# drug = 'Ozempic'
# drug = 'Wegovy'
drug = "Cavalitide"

# product-specific parameters
products = {
    'Ozempic': dict(Favailable=0.89, # bioavailability
                    ka=0.029, # 1/h, absorption rate
                    D=2.0, # mg per dose
                    interval=168, # h between doses (weekly)
                    n_doses=8, # number of doses
                    depot='subcutaneous'),

    'Wegovy': dict(Favailable=0.01,
                    ka=0.0029,
                    D=25, # mg
                    interval=24, # h between doses (daily)
                    n_doses=100,
                    depot='gut (oral)'),
    'Cavalitide': dict(Favailable=0.89,
                       ka=0.006,
                       D=0.25,
                       interval=24,
                       n_doses=100,
                       depot='gut (oral)'),
}
p = products[drug]
Favailable, ka, D = p['Favailable'], p['ka'], p['D']
interval, n_doses, depot = p['interval'], p['n_doses'], p['depot']

# shared model parameters
ke = 0.0039 # 1/h, elimination rate
V = 10.8 # L
MW = 4113.6 # g/mol
Ffree = 0.01 # rest of semaglutide bound to albumin

# simulation parameters
dt = 0.1 # h, Euler step size
def ODEfunc(t, y):
    Asc, Cc = y
    dAsc = -ka*Asc # Asc: depot amount (mg)
    dCc = ka*Asc/V - ke*Cc # Cc: blood concentration (mg/L)
    return [dAsc, dCc]

def simulate(D, n_doses, interval, dt):
    """Repeated dosing every `interval` h; returns time (h), depot amount (mg),
    and blood concentration (nmol/L)."""
    t_out, Asc_out, Cc_out = [], [], []
    y = np.array([0.0, 0.0])
    n_steps = int(round(interval / dt)) # Euler steps per dosing interval
    for i in range(n_doses):
        y[0] += Favailable*D # absorbed fraction of dose enters the depot

        t0 = i * interval
        t_seg = t0 + dt * np.arange(n_steps + 1)
        y_seg = np.zeros((2, n_steps + 1))
        y_seg[:, 0] = y

        # explicit Euler: y[n+1] = y[n] + dt * f(t[n], y[n])
        # for n in range(n_steps):
        #     y_seg[:, n + 1] = y_seg[:, n] + dt * np.array(
        #         ODEfunc(t_seg[n], y_seg[:, n])
        #     )

        # Euler's via the formula! 
        sol = scipy.integrate.solve_ivp(
            ODEfunc,
            (t0, t0 + interval),
            y,
            t_eval=t_seg,
        )
        y_seg = sol.y
        
        t_out.append(t_seg)
        Asc_out.append(y_seg[0])
        Cc_out.append(y_seg[1])
        y = y_seg[:, -1].copy() # carry state into next interval

    t = np.concatenate(t_out)
    Asc_out = np.concatenate(Asc_out)
    Cc_out = np.concatenate(Cc_out)
    Cc_nM = Cc_out * 1e-3 / MW * 1e9
    return t, Asc_out, Cc_nM


# --- run ---
t, Asc, Cc = simulate(D, n_doses, interval, dt)

# --- plot ---
plt.plot(t / 24, Asc, label=f"{depot} depot [mg]")
plt.plot(t / 24, Ffree*Cc, label="free in blood [nM]")
plt.xlabel("Time (days)")
plt.ylabel("Semaglutide")
plt.title(f"{drug}: {D} mg every {interval} h, {n_doses} doses")
plt.legend()
plt.grid(alpha=0.3)
plt.show()


# Conclusion: Cavalitide has a higher bioavailability and faster absorption rate compared to Wegovy, resulting in higher peak concentrations and more rapid onset of action.
# in order to keep the blood concentration around 0.5 nM, the dose should be at a rate of 1.0 mg, taken every other day, or lower the dose to 0.25 daily. the second route provides a more even expereince for the user, as there are not as many peaks.