## gensol (generalized solver library)
This repository contains solvers for various types of problems including optimization and (systems of) ordinary differential equations which are formulated to be as general as possible.
Note that Numpy and Scipy's routines should take preference over using those given here, especially if they have the same algorithm implemented.


## Implementation and Use
These subroutines implement the common methods of the respective algorithm, while the specifics defining the system (for example the objective function in an optimization problem) are supplied by the user as function (pointer) inputs to the `gensol` methods.

## Optimization
### `bbpm` (black box Powell method)
### `bbsd` (black box steepest descent)
- minimizes black box objective functions (those with no closed form gradient/Jacobian/Hessian) using steepest descent with line search
- uses central difference approximation to compute gradient

##Differential Equations
### `gfem` (generalized forward Euler method)
- solves (systems) of first order differential equations
- capable of handling multidimensional variables (scalar, vector, matrix) and dimensionally heterogenous systems (one scalar and one vector variable, etc.)
### `gbem` (generalized backward Euler method)
- requires some sort of inversion that may not be possible (look into)
