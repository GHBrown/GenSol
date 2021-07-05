
import numpy as np
from .context import gensol


def n_quadratic(x,extra_parameters):
    """
    a quadratic function in len(x) dimensions
    analytical minimum: x=[0, ..., 0]
    ---Inputs---
    x: input of objective function, 1D numpy array
    extra_parameters: optional extra parameters, usually dictionary
    ---Ouputs---
    val: value of quadratic at x, float
    """
    c_vec=extra_parameters["c_vec"] #coefficients for quadratic terms
    val=np.dot(c_vec,np.power(x,2.0)) #f(x)=c_0*(x_0)^2+...+c_n*(x_n)^2
    return val

def test_gd(n):
    extras_dict={
        "c_vec": np.random.rand(n)
        }
    x0=1e2*np.random.rand(n) #starting point, far from actual answer

    x_min,f_min=gensol.bbopt.gd(x0,n_quadratic,extra_parameters=extras_dict)
    print('x_min:',x_min)

if (__name__=='__main__'):
    n=10 #dimension of optimization problem
    test_gd(n)


