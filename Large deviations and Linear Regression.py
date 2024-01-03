# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 13:16:02 2023

@author: James
"""

"""
Linear regression using a Maximum Likelihood Estimation. We assume the errors are not Gaussian distributed,
instead they satisfy a large deviation principle, i.e. err_i \sim e^{-P} where P is known
as the 'rate function' and err_i is the ith error.

 The rate functions we consider are taken from:
     
'The large deviation approach to statistical mechanics'-Hugo Touchette https://arxiv.org/pdf/0804.0327.pdf

For simplicity we will focus on data with one covariate. We wish to maximise the rate function seen as a 
function of the estimation parameters of the linear fit - err_i = y_i- (b_0 + b_1 x_i). 
As log is an increasing function we can also just maximise P using gradient descent. 
"""
import numpy as np
import matplotlib.pyplot as plt 

#initialising
time = 100000 #number of iterations
learning_rate =0.0002
x = np.array([5, 15, 25, 35, 45, 55]) #inputs
y = np.array([5, 20, 14, 32, 22, 38]) #outputs
theta_0 = [0.2,0.1] #b_0, b_1

#Gradient descent algorithm
def gradient_descent(learning_rate, theta_0, time, gradient, x, y):
    theta_new = theta_0
    for i in range(time):
        theta_new += -learning_rate*np.array(gradient(x,y,theta_new))
    return theta_new

#Gaussian least squares
def grad(x,y,theta):
    derivative =theta[0] + theta[1]*x-y
    return derivative.mean(), (derivative*x).mean()


#Maximum likelihood for P(err_i) = err_i log(err_i) + (1- err_i) log(1- err_i)
def grad_1(x,y, theta):
    derivative = -np.log(np.abs((y- theta[0] - theta[1]*x)/(1-(y- theta[0] - theta[1]*x))))
    return derivative.mean(), (derivative*x).mean()

def P_err_1(x,y, theta):
    func = (y- theta[0] - theta[1]*x)*np.log(np.abs((y- theta[0] - theta[1]*x))) + (1-(y- theta[0] - theta[1]*x))*np.log(np.abs(1-(y- theta[0] - theta[1]*x)))
    return func.mean()

Len =301
beta_0 = np.linspace(-5, 5, Len)
beta_1 = np.linspace(-5, 5, Len)
h = np.zeros((Len,Len))
for i,m in enumerate(beta_0):
    for j,n in enumerate(beta_1):
        h[i][j] = P_err_1(x,y,(m,n))


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(beta_0, beta_1, h)
#plt.colorbar()
plt.xlabel('b_0')
plt.ylabel('b_1')
plt.show()

print(gradient_descent(learning_rate, theta_0, time, grad, x, y))
print(gradient_descent(learning_rate, theta_0, time, grad_1, x, y))
    
    

