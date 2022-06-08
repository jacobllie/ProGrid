import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0,5*np.pi,10)
lambda_ = 0.1
w = 1
A = 1
phi = 0


y = np.cos(w*t+phi)*A*np.exp(-lambda_*t)


fit = np.polyfit(t,y,deg = 1)

t_ = np.linspace(0,5*np.pi,100)
val = np.polyval(fit,t_)

fit2 = np.polyfit(t,y,deg=10)

val2 = np.polyval(fit2,t_)

plt.style.use("seaborn")

plt.plot(t_,val, label = "underfitting")
plt.plot(t_,val2,label = "overfitting")
plt.plot(t,y, "o", label = "data")
plt.xlabel("x [a.u.]")
plt.ylabel("y [a.u.]")
plt.legend()

plt.savefig("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\Thesis\\figures\\underfitting_vs_overfitting.png", bbox_inches = "tight", pad_inches = 0.1, dpi = 1200)
plt.show()
