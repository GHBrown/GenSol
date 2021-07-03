
import numpy as np


def fdg(x,dq,fun,control_dims=None,extra_parameters=None):
    """
    finite difference gradient (fdg)
    Uses central difference to approximate gradient of objective function.
    ---Inputs---
    x: current vector defining free parameters, 1D numpy array
    dq: finite difference step size, floating point scalar
    fun: function to compute objective function, function pointer
    control_dims: array of 0s and 1s which determines which parameters are free, 1D numpy array
    extra_parameters: optional extra_parameters for fun, dictionary
    ---Outputs---
    grad: approximate gradient, 1D numpy array
    """
    if (not control_dims): #default of control_dims cannot be set in function definition
        control_dims=np.ones(x.shape)

    grad=np.zeros(control_dims.shape[0])
    for idx,control in enumerate(control_dims):
        if (control): #only use finite difference in directions which are controllable
            finite_difference_step=np.zeros(control_dims.shape[0])
            finite_difference_step[idx]=dq
            obj_ffd=fun(x+finite_difference_step,extra_parameters) #value of objective function at forward finite difference point
            obj_bfd=fun(x-finite_difference_step,extra_parameters) #value of objective function at backward finite difference point
            grad[idx]=(obj_ffd-obj_bfd)/(2*dq)
    return grad


def bals(x,grad,desc_dir,fun_cur,beta,tau,fun,extra_parameters=None,min_alpha=1e-16):
    """
    backtracking Armijo line search (bals)
    Implementation of the backtracking-Armijo line search algorithm.
    ---Inputs---
    x: current vector defining free parameters, 1D numpy array
    grad: gradient of the objective function at x, 1D numpy array
    desc_dir: descent direction, 1D numpy array
    fun_cur: value of objective function at x, scalar
    beta: scaling constant used in evaluating Armijo condition (typically 0.1 to 0.001), floating point scalar
    tau: coefficient used to shrink alpha each line search step (between 0 and 1, exlusive), floating point scalar
    fun: function to compute objective function, function pointer
    extra_parameters: optional extra parameters for fun, dictionary
    ---Outputs---
    x_new: vector defining free parameters for next iteration, 1D numpy array
    """
    alpha=1
    cur_it=1
    while ((fun(x+alpha*desc_dir,extra_parameters) > fun_cur+alpha*beta*np.inner(grad,desc_dir)) \
           and (alpha > alpha_min)):
        alpha=tau*alpha
        cur_it+=1

    x_new=x+alpha*desc_dir #compute the optimal line search point
    return x_new


def bbsd(x,fun,grad_fun=None,extra_parameters=None,rel_tol=1e-10,dq=1e-4,control_dims=None,beta=1e-3,tau=0.5,max_it=1e6):
    """
    TO DO:
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
