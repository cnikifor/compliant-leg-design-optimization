# Christina Nikiforidou, Last edited in 12/14/2021

# This simulation function calls the methods contained in the SLIP class and simulates the sequence of the dynamic
# events that occur during the robot's locomotion. The function returns the total x-distance traversed by the robot in
# 5 seconds and it serves as the objective function of the examined optimization problem.

# Call Libraries
import numpy as np
from VariableK_Class import SLIP

# Function's argument is a vector that contains the alpha and beta coefficients of the spring rate's quadratic function
def simulation(coeff):
    # Initiate Parameters of SLIP Model:

    # Pre-define the initial conditions within the SLIP class instead of pass them in the simulation function
    slip = SLIP(coeff,
                -0.2572,
                0.9342,
                0.8426,
                0.5573,
                0,
                0,
                0,
                0,
                5)

    sim_data = np.empty((10, 0), float)
    temp_size = 0

    ############ Simulation ############

    while True:

        ########### Initiate single stance ###########

        sol_single = slip.sol_single()
        sol = sol_single
        tspan_single = np.linspace(0, sol_single.t[-1], 25)
        z_single = sol.sol(tspan_single)

        """
        Create a 10-T array that stores all simulation data:
        1st row: Time at current point
        2nd row: x
        3rd row: y
        4th row: x'
        5th row: y'
        6th row: Phase Flag (0 for flight, 1 for single stance, and 2 for double stance)
        7th row: x coordinate of 1st support
        8th row: y coordinate of 1st support
        9th row: x coordinate of 2nd support
        10th row: y coordinate of 2nd support
        """

        sim_data = np.append(sim_data, np.zeros([10, z_single.shape[1]]), 1)
        for i in range(z_single.shape[1]):
            sim_data[0, i + temp_size] = tspan_single[i]
            sim_data[1, i + temp_size] = z_single[0, i] + slip.x_0
            sim_data[2, i + temp_size] = z_single[1, i] + slip.y_0
            sim_data[3, i + temp_size] = z_single[2, i]
            sim_data[4, i + temp_size] = z_single[3, i]
            sim_data[5, i + temp_size] = 1
            sim_data[6, i + temp_size] = slip.x_0
            sim_data[7, i + temp_size] = slip.y_0
            sim_data[8, i + temp_size] = z_single[0, i] + slip.x_0
            sim_data[9, i + temp_size] = z_single[1, i] + slip.y_0 - 0.02

        temp_size = temp_size + z_single.shape[1]
        time_remaining = slip.time - sol_single.t[-1]
        if time_remaining < 10e-5:
            print('Simulation time exceeded')
            break

        elif sol_single.t_events[0].size != 0:
            print('Body touches the ground (single)')
            break

        elif sol_single.t_events[1].size != 0:
            print('Second leg touches the ground (single)')

            x_0 = slip.x_0
            y_0 = slip.y_0
            x1 = z_single[0, -1] + slip.l_0 * np.cos(np.deg2rad(slip.alpha_2nd_leg))
            y1 = z_single[1, -1] - slip.l_0 * np.sin(np.deg2rad(slip.alpha_2nd_leg))

            # Update instances in the SLIP class
            slip = SLIP(coeff,
                        z_single[0, -1],
                        z_single[1, -1],
                        z_single[2, -1],
                        z_single[3, -1],
                        x_0,
                        y_0,
                        x1,
                        y1,
                        time_remaining)

            ########### Proceed to Double Stance ###########

            sol_double = slip.sol_double()
            sol = sol_double
            tspan_double = np.linspace(0, sol_double.t[-1], 25)
            z_double = sol.sol(tspan_double)

            if sol_double.t_events[0].size != 0:
                print("Backward motion started (double)")
                break

            if sol_double.t_events[2].size != 0:
                print("Body touched ground (double)")
                break

            if sol_double.t_events[1].size != 0:
                print("First leg takes off (double)")

            # Add extra columns to the sim_data array and include the double stance data with a for loop
            sim_data = np.append(sim_data, np.zeros([10, z_double.shape[1]]), 1)

            # Loop for storing the double stance data in the columns of the 6-T sim_data array
            for i in range(z_double.shape[1]):
                sim_data[0, i + temp_size] = tspan_double[i]
                sim_data[1, i + temp_size] = z_double[0, i] + slip.x_0
                sim_data[2, i + temp_size] = z_double[1, i] + slip.y_0
                sim_data[3, i + temp_size] = z_double[2, i]
                sim_data[4, i + temp_size] = z_double[3, i]
                sim_data[5, i + temp_size] = 2
                sim_data[6, i + temp_size] = slip.x_0
                sim_data[7, i + temp_size] = slip.y_0
                sim_data[8, i + temp_size] = slip.x_0 + slip.x1
                sim_data[9, i + temp_size] = slip.y_0 + slip.y1

            temp_size = temp_size + z_double.shape[1]
            time_remaining = slip.time - sol_double.t[-1]
            if time_remaining < 10e-5:
                print("Simulaton time exceeded")
                break

        elif sol_single.t_events[2].size != 0:
            print('Leg length exceeds uncompressed spring length (single)')

            x_0 = slip.x_0
            y_0 = slip.y_0
            x1 = slip.x1
            y1 = slip.y1

            # Update instances in the SLIP class
            slip = SLIP(coeff,
                        z_single[0, -1],
                        z_single[1, -1],
                        z_single[2, -1],
                        z_single[3, -1],
                        x_0,
                        y_0,
                        x1,
                        y1,
                        time_remaining)

            ########### Proceed to Flight mode ###########

            sol_flight = slip.sol_flight()
            sol = sol_flight
            tspan_flight = np.linspace(0, sol_flight.t[-1], 25)
            z_flight = sol.sol(tspan_flight)

            if sol_flight.t_events[1].size != 0:
                print('Body touches ground (flight)')
                break

            x1 = z_flight[0, -1] + slip.l_0 * np.cos(np.deg2rad(slip.alpha_1st_leg))
            y1 = z_flight[1, -1] - slip.l_0 * np.sin(np.deg2rad(slip.alpha_1st_leg))

            # Add extra columns to the sim_data array and include the flight phase data with a for loop
            sim_data = np.append(sim_data, np.zeros([10, z_flight.shape[1]]), 1)

            # Loop for storing the flight phase data in the columns of the 6-T sim_data array
            for i in range(z_flight.shape[1]):
                sim_data[0, i + temp_size] = tspan_flight[i]
                sim_data[1, i + temp_size] = z_flight[0, i] + slip.x_0
                sim_data[2, i + temp_size] = z_flight[1, i] + slip.y_0
                sim_data[3, i + temp_size] = z_flight[2, i]
                sim_data[4, i + temp_size] = z_flight[3, i]
                sim_data[5, i + temp_size] = 0
                sim_data[6, i + temp_size] = z_flight[0, i] + slip.x_0
                sim_data[7, i + temp_size] = z_flight[1, i] + slip.y_0 - 0.02
                sim_data[8, i + temp_size] = z_flight[0, i] + slip.x_0
                sim_data[9, i + temp_size] = z_flight[1, i] + slip.y_0 - 0.02

            temp_size = temp_size + z_flight.shape[1]
            time_remaining = slip.time - sol_flight.t[-1]
            if time_remaining < 10e-5:
                print('Simulation time exceeded')
                break

            elif sol_flight.t_events[0].size != 0:
                print('First leg touches the ground (flight)')

        elif sol_single.t_events[3].size != 0:
            print('Body starts backward motion (single)')
            break

        else:
            print("Unknown termination cause")
            break

        # New support leg coordinates
        x_0 = x1 + slip.x_0
        y_0 = y1 + slip.y_0
        slip = SLIP(coeff,
                    sim_data[1, -1] - x_0,
                    sim_data[2, -1] - y_0,
                    sim_data[3, -1],
                    sim_data[4, -1],
                    x_0,
                    y_0,
                    x1,
                    y1,
                    time_remaining)

    # Simulation returns the negative overall distance made by the robot in the given time period (5 sec)
    return -(sim_data[1, -1] - sim_data[1, 0])

