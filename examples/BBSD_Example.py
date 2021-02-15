
import numpy as np
import GenSol as gs

def parabaloid(x,extras):
    coefficients=extras["coefficients"] #extract parameters from the extras dictionary
    val=np.inner(np.power(x,2),coefficients) #compute value of parabaloid at x
    return val

coefficients=np.array([0.1,1,10,100,1000])
dim=coefficients.shape[0]
guess=100*np.ones(dim)
tol=1e-5
dq=0.001
beta=0.01
tau=0.5
extras={
    "coefficients":coefficients
    }

print(' ')
print('This may take up to 45 seconds due to the varied')
print('orders of magnitudes of the coefficients.')
print(' ')
objMin,xMinimizing=gs.BBSD(guess,parabaloid,extras,relTol=tol,dq=dq,beta=beta,tau=tau)
print('Objective function minimum:')
print('   ',objMin)
print('Minimizing input:')
print('   ',xMinimizing)
