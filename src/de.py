
import numpy as np


def gfem(initial_conditions,deltat,t_start,t_stop,xp_function,constants,
    algebraic_function=None,algebraic_initial_conditions=None):
    """
    TO DO:
    test for higher order systems
    test algebraic functionality

    generalized forward euler method (gfem)
    Solves a system of first order differential equations using Euler's method
    given the initial values (and other inputs). There may be any number of variables,
    and each variable may be arbitrarily dimensioned (scalar, vector, matrix, tensor)
    and heterogeneously dimensioned (not all scalars, or all vectors, etc.).
    ---Inputs---
    initial_conditions: initial conditions, nvariable element list
    deltat: timestep, positive scalar
    t_start: time at start of solution, scalar
    t_stop: time at end of simulation, scalar (must be greater than t_start)
    xp_functionVector: vector of function pointers which point to functions which compute
        the first derivative of a variable
    ????computexpArguments: extra arguments (constants, coefficients, material properties, etc.)
        to be passed into the functions to compute xp, dictionary
    ---Outputs---
    solution: dictionary containing necessary elements of solution, fields:
        t: times in time mesh (times at which solution is given), numpy vector
        x: vector containing solution values at each point in time vector,
           variable number v at timestep s (both starting at 0) is given by x[v][s],
           length n_variables list
    """
    #Determine parameters related to time
    n_steps=int(np.ceil((t_stop-t_start)/deltat)) #compute number of deltat sized steps to simulate until (or just beyond) t_stop
    t_final_actual=t_start+n_steps*deltat #calculate the time at the actual final step
    t=np.linspace(t_start,t_final_actual,n_steps) #set time vector

    #Allocate and structure solution object
    n_variables=len(initial_conditions) #extract number of variables
    x=[0]*n_variables #initialize solution list (each entry of list will be a solution array) with dummy values
    for i_variable, variable in enumerate(initial_conditions):
        solution_shape_tuple=tuple([n_steps]+list(variable.shape)) #prepend number of time steps to the variable size (for allocation of solution object)
        x[i_variable]=np.zeros(solution_shape_tuple) #allocate solution array for current variable
        x[i_variable][0]=initial_conditions[i_variable] #set initial condition for current variable
    if algebraic_function and algebraic_initial_conditions: #set solution arrays for algebraic variables if they exist
        algebraic_variables=[0]*len(algebraic_initial_conditions)
        for i_algebraic_variable, algebraic_variable in enumerate(algebraic_initial_conditions):
            alebraic_solution_shape_tuple=tuple([n_steps]+list(algebraic_variable.shape))) #prepend number of time steps to the variable size (for allocation of algebraic solution object)
            algebraic_variables[i_algebraic_variable]=np.zeros(alebraic_solution_shape_tuple) #allocate solution array for current algebraic variable
            algebraic_variables[i_algebraic_variable][0]=algebraic_initial_conditions[i_algebraic_variable] #set initial condition for current algebraic variable

    #Time step and compute solution
    x_cur=[0]*n_variables #initialize variable to keep track of current system state
    if algebraic_function and algebraic_variables: #initialize previous and current algbraic variable state lists
        algebraic_variables_prev=algebraic_variables_cur=[0]*len(algebraic_variables)
    for i_t in range(len(t)-1):
        for i_variable in range(len(x)): #extract current values of main variables
            x_cur[i_variable]=x[i_variable][i_t]
        if algebraic_function and algebraic_variables: #compute values of algebraic variables and xp
            if i_t>0:
                for i_algebriacVariable in range(len(algebraic_variables)): #extract previous state of algebraic variables
                    algebraic_variables_prev[i_algebraic_variable]=algebraic_variables[i_algebraic_variable][i_t-1]
                algebraic_variables_cur=algebraic_function(x_cur,t[i_t],constants, algebraic_variables_prev) #compute current values of algebraic variables
                for i_algebriacVariable in range(len(algebraic_variables)): #store current algebraic variables elementwise
                    algebraic_variables[i_algebraic_variable][i_t]=algebraic_variables_cur[i_algebraic_variable]
            xp=xp_function(x_cur,t[i_t],constants,*algebraic_variables) #compute the derivative
        else: #compute xprime (without algebraic variables)
            xp=xp_function(x_cur,t[i_t],constants) #compute the derivative
        for i_variable, variable in enumerate(x): #compute new approximation for x
            x[i_variable][i_t+1]=x[i_variable][i_t]+xp[i_variable]*deltat

    solution={
        't': t,
        'x': x
    }
    return solution
