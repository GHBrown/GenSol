
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
    Backtracking Armijo Line Search (BALS), supports BBSD
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


def bbsd(x,getValue,extraParameters=None,**kwargs):
    """
    TO DO:
    allow a vector of finite difference step sizes or better yet dynamic step
        size based on previous gradient

    Black Box Steepest Descent (BBSD)
    Minimizes black box objective function.
    ---Inputs---
    x: initial result, 1D numpy array
    getValue: function to compute objective function, function pointer
    extraParameters: optional extr parameters for getValue, intended as dictionary (but technically could be anything)
    **relTol: convergence tolerance to relative change in objective function value, floating point scalar
    **dq: size of finite element step size, floating point scalar
    **controlDims: list of 1s and 0s determining which free parameters are to be controlled, list
    **beta: scaling constant used in evaluating Armijo condition (typically 0.1 to 0.001), floating point scalar
    **tau: coefficient used to shrink alpha each line search step (between 0 and 1, exclusive), floating point scalar
    ---Outputs---
    objMin: minimized value of the objective function, scalar
    xMinimizing: minimizing vector of free parameters, 1D numpy array
    """
    
    relTol=0.00001 #set default values of parameters
    dq=0.0001
    controlDims=np.ones(x.shape)
    beta=0.001
    tau=0.5
    maxIt=1e6

    if "relTol" in kwargs: #overwrite defaults if necessary
        relTol=kwargs["relTol"]
    if "dq" in kwargs:
        dq=kwargs["dq"]
    if "controlDims" in kwargs:
        controlDims=kwargs["controlDims"]
    if "beta" in kwargs:
        beta=kwargs["beta"]
    if "tau" in kwargs:
        tau=kwargs["tau"]
    if "maxIt" in kwargs:
        maxIt=kwargs["maxIt"]
    
    objPrev=1
    relChange=relTol
    while (relChange>=relTol):
        objCur=getValue(x,extraParameters) #get current value of objective function
        grad=FDG(x,dq,controlDims,getValue,extraParameters) #compute gradient
        x,LSIterations=BALS(x,grad,-grad,objCur,beta,tau,getValue,extraParameters) #find new trial point using line search in descent direction (negative gradient)
        relChange=np.abs((objCur-objPrev)/objPrev)
        objPrev=objCur
    return objCur, x
