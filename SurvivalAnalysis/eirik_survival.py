import numpy as np
import matplotlib.pyplot as plt


dat=np.loadtxt("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\main\\data.txt")

dose=dat[:,0]

kol=dat[:,1]

coefs, sigma = np.polyfit(dose, np.log(kol),2, cov=True)

print(sigma)

print('intercept:', coefs[2], sigma[2,2])

print('alpha:', -coefs[1], sigma[1,1])

print('beta:', -coefs[0], sigma[0,0])


dn=np.linspace(0,10,100)

ffit = np.polyval(coefs, dn)

plt.scatter(dose,np.log(kol))



plt.plot(dn, ffit)
