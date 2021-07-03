
import numpy as np


def fdg(x,dq,controlDims,getValue,extraParameters):
    """
    Finite Difference Gradient (FDG), supports BBSD
    Uses central difference to approximate gradient of objective function.
    ---Inputs---
    x: current vector defining free parameters, 1D numpy array
    dq: finite difference step size, floating point scalar
    controlDims: array of 0s and 1s which determines which parameters are free, 1D numpy array
    getValue: function to compute objective function, function pointer
    extraParameters: optional extraParameters for getValue, dictionary
    ---Outputs---
    grad: approximate gradient, 1D numpy array
    """
    grad=np.zeros(controlDims.shape[0])
    for idx,control in enumerate(controlDims):
        if (control): #only use finite difference in directions which are controllable
            finiteDifferenceStep=np.zeros(controlDims.shape[0])
            finiteDifferenceStep[idx]=dq
            objFFD=getValue(x+finiteDifferenceStep,extraParameters) #value of objective function at forward finite difference point
            objBFD=getValue(x-finiteDifferenceStep,extraParameters) #value of objective function at backward finite difference point
            grad[idx]=(objFFD-objBFD)/(2*dq)
    return grad


def bals(x,grad,descDir,objCur,beta,tau,getValue,extraParameters):
    """
    backtracking Armijo line search (bals)
    Implementation of the backtracking-Armijo line search algorithm.
    ---Inputs---
    x: current vector defining free parameters, 1D numpy array
    grad: gradient of the objective function at x, 1D numpy array
    descDir: descent direction, 1D numpy array
    objCur: value of objective function at x, scalar
    beta: scaling constant used in evaluating Armijo condition (typically 0.1 to 0.001), floating point scalar
    tau: coefficient used to shrink alpha each line search step (between 0 and 1, exlusive), floating point scalar
    getValue: function to compute objective function, function pointer
    extraParameters: optional extraParameters for getValue, dictionary
    ---Outputs---
    xnew: vector defining free parameters for next iteration, 1D numpy array
    """
    alpha=1
    iterations=0
    while (getValue(x+alpha*descDir,extraParameters)>objCur+alpha*beta*np.inner(grad,descDir)):
        alpha=tau*alpha
        iterations+=1
    xnew=x+alpha*descDir
    return xnew, iterations


def bbsd(x,fun,grad_fun=None,extra_parameters=None,rel_tol=1e-10,dq=1e-4,control_dims=None,beta=1e-3,tau=0.5,max_it=1e6):
    """
    TO DO:
    allow gradient function as optional input, if not use finite difference
    allow a vector of finite difference step sizes or better yet dynamic step
        size based on previous gradient

    black box steepest descent (bbsd)
    Minimizes black box objective function.
    ---Inputs---
    x: initial result, 1D numpy array
    fun: function to compute objective function, function pointer
    extra_parameters: optional extra parameters for fun, intended as dictionary (but technically could be anything)
    rel_tol: convergence tolerance to relative change in objective function value, floating point scalar
    dq: size of finite element step size, floating point scalar
    control_dims: list of 1s and 0s determining which free parameters are to be controlled, list
    beta: scaling constant used in evaluating Armijo condition (typically 0.1 to 0.001), floating point scalar
    tau: coefficient used to shrink alpha each line search step (between 0 and 1, exclusive), floating point scalar
    ---Outputs---
    obj_min: minimized value of the objective function, scalar
    x_minimizing: minimizing vector of free parameters, 1D numpy array
    """

    if (not control_dims): #default of control_dims cannot be set in function definition
        control_dims=np.ones(x.shape)
    
    fun_prev=1e3*np.info(dtype=float).tiny #initialize small previous value of objective function
    #to ensure that first iteration does not erroneously terminate
    rel_change=rel_tol
    while ((rel_change>=rel_tol) and (num_it<=max_it)):
        fun_cur=fun(x,extra_parameters) #get current value of objective function
        if (grad_fun): #use analytic gradient if it exists
            grad=grad_fun(x,extra_parameters)
        else: #or use finite difference gradient if no closed form provided
            grad=fdg(x,dq,control_dims,fun,extra_parameters) #compute gradient
        x,ls_iterations=bals(x,grad,-grad,fun_cur,beta,tau,fun,extra_parameters) #find new trial point using line search in descent direction (negative gradient)
        rel_change=np.abs((fun_cur-fun_prev)/fun_prev)
        obj_prev=obj_cur
    obj_min=obj_cur
    x_minimizing=x
    return x_minimizing, obj_min
