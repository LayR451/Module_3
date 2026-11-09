# ligand_receptor.py
import numpy as np
import matplotlib.pyplot as plt
def Fact(G, S, Kd1, Kd2):
    """formula from slideshow"""
    return (Kd2*G + Kd1*S) / (Kd1*Kd2 + Kd2*G + Kd1*S)

Kd_GLP = 0.17 # dissociation constant [nM]
Kd_semaG = 0.38 
L = np.logspace(-2, 4, 500) # ligand concentration, 0.01 to 10,000 [nM]

G_fasted = 0.0150  # fixed GLP-1, fasting [nM]
G_fed = 0.040      # fixed GLP-1, after meal [nM]

plt.figure()
plt.semilogx(L, Fact(G_fasted, L, Kd_GLP, Kd_semaG), lw=2, label="GLP-1 fasting and semaglutide")
plt.semilogx(L, Fact(G_fed, L, Kd_GLP, Kd_semaG), lw=2, label="GLP-1 after meal and semaglutide")
plt.axhline(0.5, ls="--", c="gray", lw=1)

plt.scatter(1.0, Fact(G_fasted, 1.0, Kd_GLP, Kd_semaG), c="k", zorder=3)
plt.scatter(1.0, Fact(G_fed, 1.0, Kd_GLP, Kd_semaG), c="k", zorder=3)

plt.legend()
plt.ylim(0, 1)
plt.xlabel("[Semaglutide] (nM, log scale)")
plt.ylabel("GLP1 Levels ()")
plt.show()