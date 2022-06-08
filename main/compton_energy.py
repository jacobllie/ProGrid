import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0.001,100,10000)  #MeV
phi =np.array([0.0,10.0,30.0,90.0,180.0])*np.pi/180

def compton_energy(energy, scattering_angle):
    rest_mass = 0.511 #MeV
    return energy/(1+energy/rest_mass*(1-np.cos(scattering_angle)))


plt.style.use("seaborn")
plt.xlabel(r"h$\nu_{log}$")
plt.ylabel(r"h$\nu_{log}'$")
for i in range(len(phi)):
    plt.plot(np.log(x),np.log(compton_energy(x,phi[i])), label = r"$\phi$ = {:.1f}".format(phi[i]*180/np.pi))
plt.legend()
plt.savefig("Thesis\\hv_vs_hv.png",
                           bbox_inches='tight',
                           pad_inches=0.1,
                           dpi = 1200)
plt.show()
