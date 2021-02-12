## GenSol (generalized solver library)
This repository contains solvers for various types of problems including optimization and (systems of) ordinary differential equations which are formulated to be as general as possible.


## Implementation and Use
These subroutines implement the common methods of the respective algorithm, while the specifics defining the system (for example the objective function in an optimization problem) are supplied by the user as function (pointer) inputs to the GenSol methods.

## Solvers
- GEM: generalized Euler method, for solving (systems) of first order differential equations
- BBSD: black box steepest descent (with line search), for optimization of black box objective functions (those which have no closed form Jacobian/Hessian)