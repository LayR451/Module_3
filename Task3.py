# Task 3: How the fraction of activated receptors cahnges over time 

# Embed the ligand-receptor model of Cavalitide (from Task 1, but just using your
# optimized Kd) into your cavalitidePK.py model. Let’s call this combined model
# cavalitidePKPD.py. Predict how the fraction of active GLP1R receptors changes over
# the first 100 days of treatment.


# fit_semaglutide_GLP1R.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.integrate


def Fact(L1, L2, Kd1, Kd2):
    """Fractional receptor activation with two agonists L1 and L2"""
   
    return (Kd2*L1+Kd1*L2) / (Kd1*Kd2 + Kd2*L1 + Kd1*L2)
# Parameters [nM]
Kd1 = 0.17 # L1, GLP1
Kd2 = 0.38 # default 0.38 L2, semaglutide (to fit)
L1 = 0.015 # [nM] GLP1 (fasting)
L2 = np.logspace(-3, 2, 500) # [nM] semaglutide or cavalitide

# Load the data
data = np.loadtxt("cavalitide_data.csv", delimiter=",", skiprows=1)
L2_loaded, Fact_loaded = data[:, 0], data[:, 1]


# Approach 2: Python curvefit of Kd2: related to steepest descent method
popt, pcov = curve_fit(lambda L2, Kd2: Fact(L1, L2, Kd1, Kd2),
L2_loaded, Fact_loaded, p0=[0.1], bounds=(0, np.inf))
Kd2_fit = popt[0]
SSE_fit = np.sum((Fact_loaded - Fact(L1, L2_loaded, Kd1, Kd2_fit))**2)


print(f"Fitted Kd2 = {Kd2_fit:.3f} nM (SSE = {SSE_fit:.4f})")
plt.figure()
line, = plt.semilogx(L2, Fact(L1, L2, Kd1, Kd2), lw=2,
label="model prediction")
plt.semilogx(L2_loaded, Fact_loaded, "o", c=line.get_color(),
label="experimental data")
plt.semilogx(L2, Fact(L1, L2, Kd1, Kd2_fit), "--", lw=2,
label=f"curvefit (Kd2 = {Kd2_fit:.3f} nM)")
plt.ylim(-0.1, 1.1)
plt.xlabel("[Cavalitide] (nM)")
plt.ylabel("Fraction active (Fact)")
plt.title("GLP1R activation by Cavalitide with GLP1 present")
plt.legend(loc="upper left")
plt.show()


# ===== choose product: 'Ozempic' or 'Wegovy' =====
# drug = 'Ozempic'
# drug = 'Wegovy'
drug = "Cavalitide"

# product-specific parameters
products = {
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

#finding fraction activated just for cavalitide 
def FactC(L, Kd):
    """Fractional receptor activation: L / (Kd + L)"""
    return L / (Kd + L)

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
    L = Ffree*Cc_nM # free semaglutide in blood [nM]
    Fact_out = FactC(L, Kd2_fit) # fraction of activated receptors
    return t, Asc_out, Cc_nM, Fact_out


# --- run ---
t, Asc, Cc, Fact_out = simulate(D, n_doses, interval, dt)

# --- plot ---
plt.plot(t / 24, Asc, label=f"{depot} depot [mg]")
plt.plot(t / 24, Ffree*Cc, label="free in blood [nM]")
plt.xlabel("Time (days)")
plt.ylabel("Semaglutide")
plt.title(f"{drug}: {D} mg every {interval} h, {n_doses} doses")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

plt.figure()
plt.plot(t / 24, Fact_out, label="Fraction of activated receptors")
plt.xlabel("Time (days)")
plt.ylabel("Fraction active (Fact)")
plt.title(f"{drug}: receptor activation over time")
plt.ylim(0, 1)
plt.legend()
plt.grid(alpha=0.3)
plt.show()


# Conclusion: Cavalitide has a higher bioavailability and faster absorption rate compared to Wegovy, resulting in higher peak concentrations and more rapid onset of action.
# in order to keep the blood concentration around 0.5 nM, the dose should be at a rate of 1.0 mg, taken every other day, or lower the dose to 0.25 daily. the second route provides a more even expereince for the user, as there are not as many peaks.