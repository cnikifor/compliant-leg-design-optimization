# compliant-leg-design-optimization
A gradient-free optimization algorithm that uses the Nelder-Mead Simplex method to optimize the compliant leg design of a two-legged robot. This project is a continuation of the [2d-legged-locomotion-on-2-legs](https://github.com/cnikifor/2d-legged-locomotion-on-2-legs), where the Spring-Loaded Inverted Pendulum (SLIP) model is used to simulate a two-legged robot's gait in 2D. The simulation and optimization are both developed in Python. The optimization is centered on the selection of a spring rate that would yield a stable gait which could reach maximum distance in 5 seconds. The program allows the experimentation of different spring rate models, such as a progressive spring rate that is modeled as a quadratic function of the leg’s displacement. Overall, this program can be developed into an intuitive, optimizing tool that strives to design a robust robot with locomotion on compliant legs. 

# Description
The long term goal of the study is to optimize the locomotion of a hexapod robot on six compliant legs. In robotics, optimization pertains to the maximization of the precision in completing a certain gait in 3D space as well as the reduction of unnecessary vibrations to maintain stability. Defining the goals necessary for achieving the desired motion, in terms of position or velocity, and analyzing the kinematics and dynamics of the examined system is the start towards setting up a mathematical problem. In this problem, the objective is to maximize or minimize a function that describes the desired motion, taking into consideration certain restrictions that are imposed by physics laws. Solving the problem, either exactly or numerically, helps with the determination of optimal design parameters for the successful completion of the task. 

Even though the robot is planned to be hexapedal, this study focuses on the study of a two-legged system in 2D space. From previous works, it occurs that a two-legged system simulation is sufficient to study locomotion on multiple compliant legs. The simulation of the system that describes the kinematics and dynamics is based on the Spring Loaded Inverted Pendulum physics model. Within this model, it is possible to impose specific initial conditions to the system, in terms of position and velocity, which determine the resulting motion plan of the robot. Furthermore, the simulation allows the modeling of the legged locomotion system by specifying the compliant legs’ free length and spring ratio. Of course, there are certain assumptions made for the simplification of the physical simulation: the compliant legs do not contain a damping element in their design, the ground is considered to be a frictionless and flat plane and the exact material or manufacturing process for creating the legs is not taken into account for their design parameterization. 

Given that the simulation of the robot’s system is prepared, the objective of this study is to maximize the horizontal distance that the robot traverses for a given amount of time. The only decision variable that will be optimized is the spring ratio, $K$. $K$ is variable and simulated as a quadratic function with respect to the spring displacement, given by:

$$K = |\alpha(\Delta l)^2 + \beta (\Delta l)|$$

Hence, the mathematical optimization problem is expressed as:

$$\max_{\substack{t \leq 5s \atop {0 \leq \alpha \leq 10^6 \atop -10^6 \leq \beta \leq 10^6}}} x(\alpha, \beta)$$

In the simulation, the two-legged robot is modeled as a system consisting of a rigid body with mass $m$, connected to two massless legs. The legs, in turn, are modeled as two separate spring systems, with a natural length $l_0$, and a spring stiffness $K$. The dynamics of the system is described by vector $\vec{q}=[x,y,vx,vy]$, where $x$ and $y$ are the coordinates of the time-varying position of the body with respect to the support leg, while $v_x$ and $v_y$ are the $x$ and $y$ components of the body’s velocity. According to the initial conditions of the system, the simulation numerically approaches the governing equations for the motion of the robot, which are expressed in the form of a system of differential equations. There are certain events that are predefined and when they occur, they determine the sequence of phases or the termination of the system’s gait.

Taking into account the non-linearity of the physical model as well as the fact that the derivatives within the simulation are unknown, the optimization problem is non-linear and it has been decided to use a gradient-free method to approach the problem. Specifically, the Nelder-Mead Simplex Algorithm was chosen as the ideal optimization algorithm because it is one of the simplest ways to minimize an objective function by requiring only function evaluations and comparison, as opposed to gradients of the parameters. 

In Python, the Nelder-Mead Simplex Algorithm can be implemented using the [minimize](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html) function that is offered by the [scipy](https://docs.scipy.org/doc/scipy/index.html) library. This optimization solver allows the incorporation of boundaries and constraints to the problem as well as an initial guess for the optimal values. Since the objective is to find the values of K that maximize the horizontal distance for a given amount of time, the [minimize](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html) function can be used for this purpose as long as the objective function is defined with an opposite sign, so that the biggest absolute value of the $x$-distance is discovered. 

# Getting Started
## Dependencies
For the program to run, the following libraries/sub-packages need to be imported:
* [numpy](https://numpy.org/)
* [scipy.integrate](https://docs.scipy.org/doc/scipy/tutorial/integrate.html)
* [scipy.optimize](https://docs.scipy.org/doc/scipy/tutorial/optimize.html)
* [matplotlib.pyplot](https://matplotlib.org/3.5.3/api/_as_gen/matplotlib.pyplot.html)

## Executing Program
* Within the VariableK_class, the overall SLIP physics model is defined just like in [2d-legged-locomotion-on-2-legs](https://github.com/cnikifor/2d-legged-locomotion-on-2-legs), with the only difference of implementing a progressive spring rate and neglecting damping. For the progressive spring rate, which is modeled as a quadratic function, the user can select specific 1st and 2nd order coefficient of the quadratic non-linear spring, by determining the values within the coeff argument. 
* Within the SimulationVariableK script, the "simulation" function is defined; the user can input the initial conditions. These initial conditions are imported as instance variables while calling the VariableK_class. Then, based on the initial conditions, the system starts with a specific phase, and consecutively shifts between phases until the simulation stops. Each time a phase occurs, the data generated by the numerical integration are stored in an array, called sim_data. In the end, the "simulation" function returns the overall horizontal distance made by the robot in the given time period (~5sec).
* The OptimizationVariableK script calls the the simulation function of a robot's locomotion system for a variable spring rate and it uses the Nelder-Mead Simplex Algorithm to optimize the quadratic spring rate so that it maximizes the total x-distance traversed in 5 seconds. It returns the optimal decision variables' ($\alpha$ and $\beta$) values. 
