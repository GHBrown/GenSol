## gensol (generalized solver library)
This repository contains solvers for various types of problems including optimization and (systems of) ordinary differential equations which are formulated to be as general as possible.
Note that Numpy and Scipy's routines should take preference over using those given here, especially if they have the same algorithm implemented.


## Implementation and Use
These subroutines implement the common numerical algorithms for solving problems in ordinary differential equations (ODEs) and black-box, nonlinear optimization.

The general format of function inputs to these subroutines is given below. Note the `extra_parameters` is required, but may be left unused.

```python
def user_function(x,extra_parameters=None)
    """
    ---Inputs---
    x: main variable defining state
    extra_parameters: free variable for users to pass other values,
                      could be any type of Python variable, but dictionary
                      is most likely
    ---Output---
    f_of_x: result of state based calculation
    """
    #perform calculations using x and extra_parameters to determine f_of_x
    return  f_of_x
```

## Optimization
All optimizers can handle black box functions (those with no analytical derivatives).
### Direct Search
use no derivatives or derivative estimates, possibly better for nonsmooth optimization
- *`gss`* (golden section search)
- *`nmm`* (Nelder-Mead method)
- `pm` (Powell method)
### Gradient-based
estimate gradient via finite difference if not available, but can take gradient function as input
- *`gd`* (gradient descent)
- 'ncg' (nonlinear conjugate gradient, to implement)

##Differential Equations
- `gfem` (generalized forward Euler method)
- `gbem` (generalized backward Euler method)
