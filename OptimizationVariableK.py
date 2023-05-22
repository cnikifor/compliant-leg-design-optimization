# Christina Nikiforidou, Last edited in 12/14/2021

# This script calls the simulation function of a robot's locomotion system for a variable spring rate
# and it uses the Nelder-Mead Simplex Algorithm to optimize the quadratic spring rate so that it
# maximizes the total x-distance traversed in 5 seconds.

# Call libraries
from scipy.optimize import minimize
from SimulationVariableK import simulation
import numpy as np
import matplotlib.pyplot as plt

# Define boundaries for the Spring Rate (k) values
bnds = ((1, 1e6), (-1e6, 1e6))
Nfeval = 1

# Produce arrays with random values for the initial guesses of the alpha and beta coefficients
a = np.random.randint(1, 1e6, 10)
b = np.random.randint(-1e6, 1e6, 10)
init_guess = np.column_stack((a,b))

# Pre-define empty arrays where the optimal alphas, betas and x-distances will be stored
alpha_opt = np.empty((1, 0), float)
beta_opt = np.empty((1, 0), float)
f_eval = np.empty((1, 0), float)


# Define callback function to request from the optimization solver to print all the examined values of the decision
# variables
def callbackF(vec_i):
    global Nfeval
    print('pass callback', str(Nfeval))
    print(Nfeval, vec_i, simulation(vec_i))
    Nfeval += 1


# Create loop that runs the optimization multiple times (500 times)
# Each time an optimization runs, store the values into pre-defined arrays
j = 0
for i in range(10):
    c = minimize(simulation, init_guess[i,:], bounds=bnds, method='nelder-mead', callback=callbackF,
                options={'disp': True})
    alpha_opt = np.append(alpha_opt, np.zeros([1, 1]), 1)
    alpha_opt[0, j] = c.x[0]
    beta_opt = np.append(beta_opt, np.zeros([1, 1]), 1)
    beta_opt[0, j] = c.x[1]
    f_eval = np.append(f_eval, np.zeros([1, 1]), 1)
    f_eval[0, j] = -c.fun
    j = j + 1


# Plot results in 3D graph
fig = plt.figure(figsize=(10,10))
ax = plt.axes(projection='3d')
surf = ax.scatter3D(alpha_opt, beta_opt, f_eval)
ax.set_title("Optimization for Variable Spring Rate")
ax.set_xlabel("2nd order coefficient (alpha)")
ax.set_ylabel("1st order coefficient (beta)")
ax.set_zlabel("Overall X-distance (m)")
plt.show()
