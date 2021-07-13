
import gensol.tols as tols
import numpy as np

#---only for testing---
import matplotlib.pyplot as plt
#---only for testing---

"""
Functions for optimizing black box functions
(though some allow analytic gradients).
"""


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
    if (not isinstance(control_dims,list)): #default of control_dims cannot be set in function definition
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


#---1D optimization---
def gss(obj,bracket,extra_parameters=None,abs_tol=tols._abs_tol,max_it=1e3):
    """
    golden section search (gss)
    Implementation of golden section search that works for one and n-dimensions.
    NOTE: Due to numerical error, the points do not remain exactly on the original line
          specified by bracket, and so the error may not be driven down arbitrarily.
          For example, for starting bracket [-2*x0,x0] the points are only linearly dependent
          up to a certain precision.
          **I believe this is what causes the problem when testing the quadratic case.**
    ---Inputs---
    obj: function pointer to objective function, function pointer
    bracket: one dimensional bracket on which there is one minimum, 2 element numpy array
    abs_tol: absolute tolerance, float
    max_it: maximum number of iterations
    ---Outputs---
    x_min: point at which obj is minimized, float 
    obj_min: value of obj at x_min, float
    """

    """
    the four points of golden section search
    0-----0---0-----0
    x0    x1  x2    x3
    """
    ep=extra_parameters #for brevity
    #precopute common quantities for speed
    frac_left=(3-np.power(5,0.5))/2 #fraction of interval left of x1, equal to (x1-x0)/(x3-x0)
    frac_right=1.0-frac_left #fraction of interval right of x1

    #setup points, generically x0,..,x3 are vectors
    x0=bracket[0] #endpoints
    x3=bracket[1]
    x_disp=x3-x0 #displacement vector between endpoints
    x_disp_norm=np.linalg.norm(x_disp) #norm of displacement vector, interval length
    x_disp_unit=x_disp/x_disp_norm #unit vector giving direction between two endpoints
    x1=x0+frac_left*x_disp_norm*x_disp_unit #internal points
    x2=x0+frac_right*x_disp_norm*x_disp_unit
    xvec=[x0,x1,x2,x3]
    objxvec=[obj(x0,extra_parameters=ep),obj(x1,extra_parameters=ep),
           obj(x2,extra_parameters=ep),obj(x3,extra_parameters=ep)] #objective function values at 4 points

    #perform search
    num_it=0
    while ((x_disp_norm > abs_tol) and (num_it < max_it)):
        x_disp_norm*=frac_right #four points symmetric about interval center, so change in interval length same regardless of which internal point is smaller
        if (objxvec[1] < objxvec[2]): #internal points become righmost points of new bracket
            xvec[2:]=xvec[1:3]
            xvec[1]=xvec[0]+frac_left*x_disp_norm*x_disp_unit
            objxvec[2:]=objxvec[1:3]
            objxvec[1]=obj(xvec[1],extra_parameters=ep)
        else: #internal points become leftmost points of new bracket
            xvec[0:2]=xvec[1:3]
            xvec[2]=xvec[0]+frac_right*x_disp_norm*x_disp_unit
            objxvec[0:2]=objxvec[1:3]
            objxvec[2]=obj(xvec[2],extra_parameters=ep)
        num_it+=1

    x_min=(xvec[0]+xvec[3])/2
    obj_min=(objxvec[0]+objxvec[3])/2
    return x_min, obj_min
        

#---line searches---
def bals(x,grad,desc_dir,fun_cur,beta,tau,fun,extra_parameters=None,alpha_min=tols._abs_tol):
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
    if (not alpha_min): #define minimum line search alpha relative to float type minimum
        alpha_min=1e2*np.finfo(x[0]).tiny

    alpha=1
    cur_it=1
    while ((fun(x+alpha*desc_dir,extra_parameters) > fun_cur+alpha*beta*np.inner(grad,desc_dir)) \
           and (alpha > alpha_min)):
        alpha=tau*alpha
        cur_it+=1

    x_new=x+alpha*desc_dir #compute the optimal line search point
    return x_new


#---main optimization functions---
def gd(obj,x,grad_obj=None,extra_parameters=None,rel_tol=tols._rel_tol,dq=1e-4,control_dims=None,beta=1e-3,tau=0.5,max_it=1e6):
    """
    TO DO:
    allow a vector of finite difference step sizes or better yet dynamic step
        size based on previous gradient

    gradient descent (gd)
    Minimizes black box objective function using finite difference based gradient if
    no analytical gradient is provided.
    ---Inputs---
    x: initial result, 1D numpy array
    obj: function pointer to objective function, function pointer
    extra_parameters: optional extra parameters for fun, intended as dictionary (but technically could be anything)
    rel_tol: convergence tolerance on relative change in objective function value, float
    dq: size of finite difference step size, floating point scalar
    control_dims: list of 1s and 0s determining which entries of x may be changed, list
    beta: scaling constant used in evaluating Armijo condition (typically 0.1 to 0.001), floating point scalar
    tau: coefficient used to shrink alpha each line search step (between 0 and 1, exclusive), floating point scalar
    max_it: maximum number of iterations of gradient estimation and line search, int
    ---Outputs---
    obj_min: minimized value of the objective function, scalar
    x_minimizing: minimizing vector of free parameters, 1D numpy array
    """

    if (not isinstance(control_dims,list)): #default of control_dims cannot be set in function definition
        control_dims=np.ones(x.shape)
    
    obj_prev=1e-6 #initialize small previous value of objective function to ensure that first
    #iteration does not erroneously terminate with small rel_change
    rel_change=rel_tol
    num_it=0
    while ((rel_change>=rel_tol) and (num_it<max_it)):
        obj_cur=obj(x,extra_parameters) #get current value of objective function
        if (grad_obj): #use analytic gradient if it exists
            grad=grad_obj(x,extra_parameters)
        else: #or use finite difference gradient if no closed form provided
            grad=fdg(x,dq,obj,control_dims=control_dims,extra_parameters=extra_parameters) #compute gradient
        x=bals(x,grad,-grad,obj_cur,beta,tau,obj,extra_parameters) #find new trial point using line search in descent direction (negative gradient)
        rel_change=np.abs((obj_cur-obj_prev)/obj_prev)
        obj_prev=obj_cur
        num_it+=1
    obj_min=obj_cur
    x_minimizing=x
    return x_minimizing, obj_min


def nmm(obj,x0,extra_parameters=None,abs_tol=tols._abs_tol,o_tol=None,max_it=1e4,offset=10,offset_vec=None,alpha=1,beta=0.5,gamma=2,delta=0.5):
    """
    Nelder-Mead method (nmm)
    optimizes a black-box function using only function values via the Nelder-Mead method*
    *(implementation follows outline on Scholarpedia)
    ---Inputs---
    obj: function pointer to objective function, function pointer
    x0: one specified vertex of simplex ("starting point"), 1D numpy array
    extra_parameters: optional extra parameters for fun, intended as dictionary (but technically could be anything)
    abs_tol: convergence tolerance on the absolute closeness of simplex vertices, float
    o_tol: convergence tolerance on the closeness of objective function values, float
    offset: value determining how far away other vertices are from x0, float
    offset_vec: vector of length n specifying how far vertices of simplex should be from x0
                along coordinate directions, 1D numpy array
    max_it: maximum number of iterations, int
    """
    ep=extra_parameters #for brevity
    n=np.shape(x0)[0] #number of degrees of freedom

    #set all necessary parameters from inputs
    if (o_tol is None):
        o_tol=np.power(abs_tol,2.3) #define tolerance on function values as more than square of tolerance on vertices
    if offset_vec is None:
        offset_vec=offset*np.ones(n)

    #initialize vertices
    V=np.empty((n+1,n)) #2D array holding vertex locations in rows
    V[0,:]=x0
    V[1:,:]=[x0+offset_vec[i]*np.eye(n)[:,i] for i in range(n)] #set other vertices to the appropiate offset away in coordinate directions

    #initialize vector with objective function values
    O=np.empty(n+1)
    for i in range(O.shape[0]):
        O[i]=obj(V[i,:],extra_parameters=ep)
        
    #execute search
    num_it=0
    v_close=False #entry value for Boolean determining whether vertices are sufficiently close
    o_close=False #entry value for Boolean determining whether objective values are sufficiently close
    while ((not v_close) and (not o_close) and (num_it < max_it)):
        #find smallest, second largest, and largest objective values
        #and the vertices at which they occur
        i_min=np.argmin(O) #vertex number with smallest objective value
        o_min=O[i_min] #minimum function value on vertices
        v_min=V[i_min,:] #location of vertex at which obj is smallest

        i_maxes_unsorted=np.argpartition(O,-2)[-2:] #indices of 2 largest vertices (unordered)
        o_maxes_unsorted=O[i_maxes_unsorted] #two largest objective values (unsorted)
        o_sort_indices=np.argsort(o_maxes_unsorted) #indices that sort o_maxes_unsorted (ascending)
        i_maxes=i_maxes_unsorted[o_sort_indices] #indices of 2 largest vertices
        i_2max=i_maxes[0] #second largest objective value quantities
        o_2max=o_maxes_unsorted[o_sort_indices][0]
        v_2max=V[i_2max,:]
        i_max=i_maxes[1] #largest objective value quantities 
        o_max=o_maxes_unsorted[o_sort_indices][1]
        v_max=V[i_max,:]

        v_bary=(np.sum(V,axis=0)-v_max)/(n+1) #compute barycenter of points, excluding v_max
        v_ref=v_bary+alpha*(v_bary-v_max) #v_max reflected about v_bary by distance alpha
        o_ref=obj(v_ref,extra_parameters=ep) #objective value at proposed vertex

        if ((o_min<=o_ref) and (o_ref<o_2max)): #reflect
            V[i_max,:]=v_ref #accept reflected point
            O[i_max]=o_ref
        elif (o_ref < o_min): #expand
            v_greed=v_bary+gamma*(v_ref-v_bary) #propose point reflected even further than v_ref
            o_greed=obj(v_greed,extra_parameters=ep)
            if (o_greed < o_ref):
                V[i_max,:]=v_greed
                O[i_max]=o_greed
            else:
                V[i_max,:]=v_ref #accept standard reflected, greedy point was not better
                O[i_max]=o_ref
        elif (o_ref > o_2max): #contract
            if ((o_2max <= o_ref) and (o_ref < o_max)):
                v_con=v_bary+beta*(v_ref-v_bary) #compute contracted point external to simplex
                o_con=obj(v_con,extra_parameters=ep)
                if (o_con < o_ref):
                    V[i_max,:]=v_con #accept external contracted point
                    O[i_max]=o_con
            elif (o_ref > o_max):
                v_con=v_bary+beta*(v_max-v_bary) #compute contracted point internal to simplex
                o_con=obj(v_con,extra_parameters=ep)
                if (o_con < o_max):
                    V[i_max,:]=v_con #accept internal contracted point
                    O[i_max]=o_con
        else: #shrink (most expensive, last resort)
            shrink_vertices=np.arange(n+1)
            np.delete(shrink_list,i_min)
            for i_v in shrink_vertices: #shrink vertices (except minimum) and compute objective
                V[i_v]=v_min+delta*(V[i_v]-v_min)
                O[i_v]=obj(V[i_v],extra_parameters=ep)

        #determine if vertices are close enough
        v0_array=np.vstack(tuple([V[0,:]]*n))
        v0_disp_array=np.linalg.norm(V[1:,:]-v0_array,axis=1) #vector containing displacement of all points from vertex 0
        v_close=(np.mean(v0_disp_array) < abs_tol)

        #determine if function values are close enough
        o_close=(np.std(O) < o_tol)

        num_it+=1

    #compute final parameters and objective value estimate
    x_min=np.mean(V,axis=0) #average of all vertices
    obj_min=obj(x_min,extra_parameters=ep)
    return x_min, obj_min


#powell's method (pm)

