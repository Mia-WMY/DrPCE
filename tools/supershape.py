import numpy as np
import matplotlib.pyplot as plt

def supershape(theta, a, b, m, n1, n2, n3):
    t1 = np.abs((1/a)*np.cos(m*theta/4))**n2
    t2 = np.abs((1/b)*np.sin(m*theta/4))**n3
    r = (t1 + t2)**(-1/n1)
    return r

# 生成圆形
theta = np.linspace(0, 2*np.pi, 1000)
r = supershape(theta, a=1, b=1, m=0, n1=1, n2=1, n3=1)
x = r*np.cos(theta)
y = r*np.sin(theta)

plt.plot(x, y, linewidth=2)
plt.axis('equal')
plt.show()

# 生成三角形
theta = np.linspace(0, 2*np.pi, 1000)
r = supershape(theta, a=1, b=1, m=3, n1=1, n2=1, n3=1)
x = r*np.cos(theta)
y = r*np.sin(theta)

plt.plot(x, y, linewidth=2)
plt.axis('equal')
plt.show()

# 生成矩形
theta = np.linspace(0, 2*np.pi, 1000)
r = supershape(theta, a=1, b=1, m=4, n1=1, n2=1, n3=1)
x = r*np.cos(theta)
y = r*np.sin(theta)

plt.plot(x, y, linewidth=2)
plt.axis('equal')
plt.show()

# 生成正六边形
theta = np.linspace(0, 2*np.pi, 1000)
r = supershape(theta, a=1, b=1, m=6, n1=1, n2=1, n3=1)
x = r*np.cos(theta)
y = r*np.sin(theta)

# 生成正八边形
theta = np.linspace(0, 2*np.pi, 1000)
r = supershape(theta, a=1, b=1, m=8, n1=1, n2=1, n3=1)
x = r*np.cos(theta)
y = r*np.sin(theta)

# 生成正十边形
theta = np.linspace(0, 2*np.pi, 1000)
r = supershape(theta, a=1, b=1, m=10, n1=1, n2=1, n3=1)
x = r*np.cos(theta)
y = r*np.sin(theta)

plt.plot(x, y, linewidth=2)
plt.axis('equal')
plt.show()
