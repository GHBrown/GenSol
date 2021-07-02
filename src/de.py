
import numpy as np


def GEM(initialConditions,deltat,t_start,t_stop,xpFunction,constants,
    algebraicFunction=None,algebraicVariableInitialConditions=None):
    """
    TO DO:
    test for higher order systems
    test algebraic functionality

    Generalized Euler Method (GEM)
    Solves a system of first order differential equations using Euler's method
    given the initial values (and other inputs). There may be any number of variables,
    and each variable may be arbitrarily dimensioned (scalar, vector, tensor, matrix)
    and heterogeneously dimensioned (not all scalars, or all vectors, etc.).
    ---Inputs---
    initialConditions: initial conditions, nvariable element list
    deltat: timestep, positive scalar
    t_start: time at start of solution, scalar
    t_stop: time at end of simulation, scalar (must be greater than t_start)
    xpFunctionVector: vector of function pointers which point to functions which compute
        the first derivative of a variable
    computexpArguments: extra arguments (constants, coefficients, material properties, etc.)
        to be passed into the functions to compute xp, dictionary
    ---Outputs---
    solution: dictionary containing necessary elements of solution, fields:
        t: times in time mesh (times at which solution is given), numpy vector
        x: vector containing solution values at each point in time vector, (len(t),dim) numpy array
    """
    #Determine parameters related to time
    n_steps=int(np.ceil((t_stop-t_start)/deltat)) #compute number of deltat sized steps to simulate to (or just beyond) t_stop
    t_final_actual=t_start+n_steps*deltat #calculate the time at the actual final step
    t=np.linspace(t_start,t_final_actual,n_steps) #set time vector

    #Set up solution structure
    n_variables=len(initialConditions) #extract number of variables
    x=[0]*n_variables #initialize solution list (each entry of list will be a solution array) with dummy values
    for i_variable, variable in enumerate(initialConditions):
        dimensionality=list(variable.shape) #compute dimensionality of the current variable
        dimensionality.insert(0,n_steps) #prepend number of time steps to the variable size (for allocation of solution vector)
        dimensionalitySolutionArray=tuple(dimensionality) #convert to tuple
        x[i_variable]=np.zeros(dimensionalitySolutionArray) #allocate solution array for current variable
        x[i_variable][0]=initialConditions[i_variable] #set initial condition for current variable
    if algebraicFunction and algebraicVariableInitialConditions: #set solution arrays for algebraic variables if they exist
        algebraicVariables=[0]*len(algebraicVariableInitialConditions)
        for i_algebraicVariable, algebraicVariable in enumerate(algebraicVariableInitialConditions):
            dimensionality=list(algebraicVariable.shape)
            dimensionality.insert(0,n_steps) #prepend number of time steps to the variable size (for allocation of solution vector)
            dimensionalityAlgebraicVariableSolutionArray=tuple(dimensionality) #convert to tuple
            algebraicVariables[i_algebraicVariable]=np.zeros(dimensionalityAlgebraicVariableSolutionArray) #allocate solution array for current algebraic variable
            algebraicVariables[i_algebraicVariable][0]=algebraicVariableInitialConditions[i_algebraicVariable] #set initial condition for current algebraic variable

    x_cur=[0]*n_variables #initialize variable to keep track of current system state
    if algebraicFunction and algebraicVariables: #initialize previous and current algbraic variable state lists
        algebraicVariables_prev=algebraicVariables_cur=[0]*len(algebraicVariables)
    for i_t in range(len(t)-1):
        for i_variable in range(len(x)): #extract current values of main variables
            x_cur[i_variable]=x[i_variable][i_t]
        if algebraicFunction and algebraicVariables: #compute values of algebraic variables and xp
            if i_t>0:
                for i_algebriacVariable in range(len(algebraicVariables)): #extract previous state of algebraic variables
                    algebraicVariables_prev[i_algebraicVariable]=algebraicVariables[i_algebraicVariable][i_t-1]
                algebraicVariables_cur=algebraicFunction(x_cur,t[i_t],constants, algebraicVariables_prev) #compute current values of algebraic variables
                for i_algebriacVariable in range(len(algebraicVariables)): #store current algebraic variables elementwise
                    algebraicVariables[i_algebraicVariable][i_t]=algebraicVariables_cur[i_algebraicVariable]
            xp=xpFunction(x_cur,t[i_t],constants,*algebraicVariables) #compute the derivative
        else: #compute xprime (without algebraic variables)
            xp=xpFunction(x_cur,t[i_t],constants) #compute the derivative
        for i_variable, variable in enumerate(x): #compute new approximation for x
            x[i_variable][i_t+1]=x[i_variable][i_t]+xp[i_variable]*deltat

    solution={
        't': t,
        'x': x
    }
    return solution
