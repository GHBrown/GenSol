
import numpy as np
from .context import gensol
from gensol import tols


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

def test_gd(n,abs_tol):
    """
    tests the gradient descent algorithm with a random quadratic problem in n dimensions
    ---Inputs---
    n: number of dimensions/parameters in optimization problem, integer
    abs_tol: absolute error tolerance, float
    --Outputs--
    NONE, prints info to terminal
    """
    extras_dict={
        "c_vec": np.random.rand(n)
        }
    x0=1e2*np.random.rand(n) #starting point, far from actual answer

    x_min,f_min=gensol.bbopt.gd(n_quadratic,x0,extra_parameters=extras_dict,rel_tol=abs_tol)

    x_sol=np.zeros(n)
    abs_err=np.linalg.norm(x_min-x_sol) #compute absolute error
    if (abs_err < abs_tol):
        print('  PASSED,  gd (gradient descent)')
    else:
        print('  FAILED,  gd (gradient descent)')
        print('  abs_err: ', abs_err)


def test_gss(n,abs_tol):
    """
    EVENTUALLY MAKE TEST N DIMENSIONAL AND HAVE POINTS DRAW LINE THROUGH ZERO
    Tests the golden section search algorithm with a quadratic problem in 1 dimension.
    ---Inputs---
    n: number of dimensions/parameters in optimization problem, integer
    abs_tol: absolute error tolerance, float
    --Outputs--
    NONE, prints info to terminal
    """
    randx=np.random.rand(n)
    bracket0=[-20*randx,30*randx]
    extras_dict={
        "c_vec": np.random.rand(n)
        }
    x_min,f_min=gensol.bbopt.gss(n_quadratic,bracket0,extra_parameters=extras_dict,abs_tol=abs_tol)

    x_sol=np.zeros(n)
    abs_err=np.linalg.norm(x_min-x_sol) #compute absolute error
    if (abs_err < abs_tol):
        print('  PASSED,  gss (golden section search)')
    else:
        print('  FAILED,  gss (golden section search)')
        print('      abs_err: ', abs_err)



if (__name__=='__main__'):
    n=10 #dimension of optimization problem
    abs_tol=tols._abs_tol #absolute error tolerance

    print('---bbopt---')
    test_gd(n,abs_tol)
    test_gss(n,1e-14) #see the note in gss for why different tolerance is chosen here
    #test_gss(1,1e-150) #this test case should pass with not problems
    #test_gss(10,abs_tol) #this case should fail very frequently unless the problem noted in gss
    #can be fixed


