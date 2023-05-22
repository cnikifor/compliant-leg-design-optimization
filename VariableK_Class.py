# Christina Nikiforidou, Last edited in 12/14/2021

# This class contains all the functions of the dynamics and kinematics of a two-legged robot. Each function
# either describes the governing equations for a flight, single stance and double stance phase or represents
# an event that signals the change of phase or the termination of the simulation.

# Call Libraries
import numpy as np
from scipy.integrate import solve_ivp


class SLIP:
    """
    q is [x, y, v_x, v_y],
    where x, y are the x coordinate and y coordinate of the
    point mass's position, and v_x, v_y are the x component
    and y component of the velocity of the point mass
    """

    l_0 = 1  # Natural length of spring leg [m]
    mass = 80  # Mass [kg]
    g = 9.81  # Gravitational acceleration
    prof_x = [0, 1000]  # Ground profile x coordinates
    prof_y = [0, 0]  # Ground profile y coordinates
    C = 0  # Damping coefficient
    alpha_1st_leg = 69.754  # Touchdown angle of first leg
    alpha_2nd_leg = 69.103  # Touchdown angle of second leg:

    def __init__(self,
                 coeff,  # Coefficient of quadratic non-linear springs
                 x,
                 y,
                 v_x,
                 v_y,
                 x_0,  # Initial x-coordinate of support legs
                 y_0,  # Initial y-coordinate of support legs
                 x1,  # Initial horizontal distance between two support legs
                 y1,  # Initial vertical distance between two support legs
                 time):  # Time remaining for simulation

        self.alpha = coeff[0]  # 2nd order coeff of quadratic non-linear springs
        self.beta = coeff[1]  # 1st order coeff of quadratic non-linear springs
        self.q = np.asarray([x, y, v_x, v_y])
        self.x_0 = x_0  # Initial x-coordinate of support legs
        self.y_0 = y_0  # Initial y-coordinate of support legs
        self.x1 = x1  # Initial horizontal distance between two support legs
        self.y1 = y1  # Initial vertical distance between two support legs
        self.time = time
        self.params = (self.x_0, self.y_0, self.x1, self.y1)

    ########### Equations of Motion ###########

    @staticmethod
    def EoM_flight(t, q):
        """Governing equations of the system's motion during flight"""

        dqdt = np.zeros_like(q)
        dqdt[0] = q[2]
        dqdt[1] = q[3]
        dqdt[2] = 0
        dqdt[3] = -SLIP.g

        return dqdt

    def EoM_single(self, t, q):
        """Governing equations of the system's motion during single stance"""

        dqdt = np.zeros_like(q)
        dqdt[0] = q[2]
        dqdt[1] = q[3]

        l_1 = np.sqrt(q[0] ** 2 + q[1] ** 2)  # Instant spring length
        delta_l = SLIP.l_0 - l_1
        k = self.alpha * delta_l**2 + self.beta * delta_l  # Instant spring rate

        a_1 = (np.abs(k) * delta_l) / SLIP.mass

        assert a_1 * delta_l >= 0, "Acceleration and Displacement do not have opposite direction"

        dqdt[2] = a_1 / l_1 * q[0]
        dqdt[3] = a_1 / l_1 * q[1] - SLIP.g

        return dqdt

    def EoM_double(self, t, q):
        """Governing equations of the system's motion during double stance"""
        (x_0, y_0, x1, y1) = self.params

        dqdt = np.zeros_like(q)
        dqdt[0] = q[2]
        dqdt[1] = q[3]

        l_1 = np.sqrt(q[0] ** 2 + q[1] ** 2)
        l_2 = np.sqrt((q[0] - x1) ** 2 + (q[1] - y1) ** 2)
        delta_l1 = SLIP.l_0 - l_1
        delta_l2 = SLIP.l_0 - l_2
        k1 = self.alpha * delta_l1**2 + self.beta * delta_l1  # Instant spring rate at first leg
        k2 = self.alpha * delta_l2**2 + self.beta * delta_l2  # Instant spring rate at second leg

        # Acceleration due to first leg:
        a_1 = (np.abs(k1) * delta_l1) / SLIP.mass
        # Acceleration due to second leg:
        a_2 = (np.abs(k2) * delta_l2) / SLIP.mass

        assert a_1 * delta_l1 >= 0 and a_2 * delta_l2 >= 0, "Acceleration and Displacement do not have opposite " \
                                                            "direction "

        dqdt[2] = a_1 * q[0] / l_1 + a_2 * (q[0] - x1) / l_2
        dqdt[3] = a_1 * q[1] / l_1 + a_2 * (q[1] - y1) / l_2 - SLIP.g

        return dqdt

    ########### Event Functions for Single Stance ###########

    def single_2nd_leg_ground(self, t, q):
        """Single stance ends due to second leg touching the ground"""
        (x_0, y_0, x1, y1) = self.params

        impact = np.array([q[1] - SLIP.l_0 * np.sin(
            np.deg2rad(SLIP.alpha_2nd_leg)) - np.interp(x_0 + q[0] +
                                                        SLIP.l_0 * np.cos(
            np.deg2rad(SLIP.alpha_2nd_leg)), SLIP.prof_x,
                                                        SLIP.prof_y) + y_0])

        return impact

    single_2nd_leg_ground.terminal = True
    single_2nd_leg_ground.direction = -1

    def single_leg_length_overexceeded(self, t, q):
        """Single stance ends due to leg's leg over-exceeding natural length"""

        impact = np.array([SLIP.l_0 ** 2 - q[0] ** 2 - q[1] ** 2])
        return impact

    single_leg_length_overexceeded.terminal = True
    single_leg_length_overexceeded.direction = -1

    def single_backward_motion(self, t, q):
        """Single stance ends due to body starting backward motion"""

        impact = np.array([q[2]])
        return impact

    single_backward_motion.terminal = True
    single_backward_motion.direction = -1

    def single_body_ground(self, t, q):
        """Single stance ends due to second leg touching the ground"""
        (x_0, y_0, x1, y1) = self.params

        impact = np.array([q[1] - np.interp(x_0 + q[0], SLIP.prof_x, SLIP.prof_y)])
        return impact

    single_body_ground.terminal = True
    single_body_ground.direction = -1

    ########### Event Functions for Flight Phase ###########

    def flight_1st_leg_ground(self, t, q):
        """Flight Phase ends due to first leg touching the ground"""
        (x_0, y_0, x1, y1) = self.params

        impact = np.array([q[1] - SLIP.l_0 * np.sin(np.deg2rad(SLIP.alpha_1st_leg)) -
                           np.interp(x_0 + q[0] + SLIP.l_0 * np.cos(np.deg2rad(SLIP.alpha_1st_leg)),
                                     SLIP.prof_x, SLIP.prof_y) + y_0])
        return impact

    flight_1st_leg_ground.terminal = True
    flight_1st_leg_ground.direction = -1

    def flight_body_ground(self, t, q):
        """Flight Phase ends due to body falling to the ground"""
        (x_0, y_0, x1, y1) = self.params

        impact = np.array([q[1] - np.interp(x_0 + q[0], SLIP.prof_x, SLIP.prof_y)])
        return impact

    flight_body_ground.terminal = True
    flight_body_ground.direction = -1

    ########### Event Functions for Double Stance ###########

    def first_leg_takeoff(self, t, q):
        """Double stance ends due to first leg taking off"""

        impact = np.array([q[0] ** 2 + q[1] ** 2 - SLIP.l_0 ** 2])
        return impact

    first_leg_takeoff.terminal = True
    first_leg_takeoff.direction = 1

    def double_backward_motion(self, t, q):
        """Single stance ends due to second leg touching the ground"""

        impact = np.array([q[2]])
        return impact

    double_backward_motion.terminal = True
    double_backward_motion.direction = -1

    def second_leg_takeoff(self, t, q):
        """Single stance ends due to second leg touching the ground"""
        (x_0, y_0, x1, y1) = self.params

        impact = np.array([(q[0] - x1) ** 2 + (q[1] - y1) ** 2 - SLIP.l_0 ** 2])
        return impact

    second_leg_takeoff.terminal = True
    second_leg_takeoff.direction = 1

    def double_body_ground(self, t, q):
        """Flight Phase ends due to body falling to the ground"""
        (x_0, y_0, x1, y1) = self.params

        impact = np.array([q[1] - np.interp(x_0 + q[0], SLIP.prof_x, SLIP.prof_y)])
        return impact

    double_body_ground.terminal = True
    double_body_ground.direction = -1


    ########### Integrating Functions ###########

    def sol_single(self):
        return solve_ivp(self.EoM_single, [0, self.time], self.q, events=(self.single_body_ground,
                                                                          self.single_2nd_leg_ground,
                                                                          self.single_leg_length_overexceeded,
                                                                          self.single_backward_motion),
                         dense_output=True)

    def sol_flight(self):
        return solve_ivp(self.EoM_flight, [0, self.time], self.q, events=(self.flight_1st_leg_ground,
                                                                          self.flight_body_ground),
                         dense_output=True)

    def sol_double(self):
        return solve_ivp(self.EoM_double, [0, self.time], self.q, events=(self.double_backward_motion,
                                                                          self.first_leg_takeoff,
                                                                          self.double_body_ground),
                         dense_output=True)
