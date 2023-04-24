import numpy as np
import sympy as sp
from sympy.utilities.lambdify import lambdify


class Single_Vehicle_Model:

    def __init__(self, parameter) -> None:
        """
        state: [x, y, theta] where x, y are the in the rear axis
        current pose: [x + l * cos(theta) y + l * sin(theta), theta]
        control: [v, w]

        for symbolic:
        state: [x, y, theta]
        robot_end_state: [x, y, theta]
        
        for collision avoidance:
        robot_state: [x, y, theta, radius] 

        obstacle_state: [o_x, o_y, o_radius]
        """
        # system states
        self.state_dim = 3
        self.control_dim = 2

        x, y, theta = sp.symbols('x y theta')
        robot_radius = sp.Symbol('robot_radius')
        self.state = sp.Matrix([x, y, theta])
        self.robot_state = sp.Matrix([x, y, theta, robot_radius])
        self.l = parameter['l']
        self.margin = parameter['margin']

        e_x, e_y, e_theta = sp.symbols('e_x e_y e_theta')
        self.target_state = sp.Matrix([e_x, e_y, e_theta])

        o_x, o_y, o_radius = sp.symbols('o_x, o_y, o_radius')
        self.obstacle_state = sp.Matrix([o_x, o_y, o_radius])
        self.obstacle_status = sp.Matrix([o_x, o_y])

        o_vx, o_vy = sp.symbols('o_vx, o_vy')
        self.obstacle_dynamics = sp.Matrix([o_vx, o_vy])

        # system dynamics and clf, cbf
        self.f = None
        self.f_symbolic = None

        self.g = None
        self.g_symbolic = None

        # CLF
        self.clf = None
        self.clf_symbolic = None

        # Lie derivative of clf w.r.t f / g as a function
        self.dx_clf = None
        self.dx_clf_symbolic = None
        self.lf_clf = None
        self.lf_clf_symbolic = None
        self.lg_clf = None
        self.lg_clf_symbolic = None

        # target dynamics, for trajectory tracking
        # To do
        self.dtarget_clf = None
        self.dtarget_clf_symbolic = None
        self.dt_clf = None
        self.dt_clf_symbolic = None

        # CBF
        self.cbf = None
        self.cbf_symbolic = None

        # Lie derivative of cbf w.r.t f / g as a function
        self.dx_cbf = None
        self.dx_cbf_symbolic = None
        self.lf_cbf = None
        self.lf_cbf_symbolic = None
        self.lg_cbf = None
        self.lg_cbf_symbolic = None

        # dcbf / dt, time derivative of other items
        self.dox_cbf = None
        self.dox_cbf_symbolic = None
        self.dt_cbf = None
        self.dt_cbf_symbolic = None

        self.init_system()

    def init_system(self):

        # Define the system dynamics, CLF and CBF in a symbolic way
        self.f_symbolic, self.g_symbolic = self.define_system_dynamics()
        self.f = lambdify([self.state], self.f_symbolic)
        self.g = lambdify([self.state], self.g_symbolic)

        # choose for a type of clf
        self.clf_symbolic = self.define_clf_cross_term()
        # self.clf_symbolic = self.define_clf_center()
        self.clf = lambdify([self.state, self.target_state], self.clf_symbolic)

        self.cbf_symbolic = self.define_cbf()
        self.cbf = lambdify([self.robot_state, self.obstacle_state], self.cbf_symbolic)

        self.lie_derivatives_calculator()
        self.time_differential_cbf()

    def define_system_dynamics(self):
        f = sp.Matrix([0, 0, 0])
        g = sp.Matrix([[sp.cos(self.state[2]), 0], [sp.sin(self.state[2]), 0], [0, 1]])
        return f, g
    
    def define_clf_center(self):
        """ 
        define the clf with the center position of robot represented by the rear 
        x = x_p + l * cos(theta)
        y = y_p + l * sin(theta)
        """
        H = sp.Matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
        relative_x = self.state[0] + self.l * (sp.cos(self.state[2]) - sp.cos(self.target_state[2])) - self.target_state[0]
        relative_y = self.state[1] + self.l * (sp.sin(self.state[2]) - sp.sin(self.target_state[2])) - self.target_state[1]
        relative_orientation = self.state[2] - self.target_state[2]

        relative_state = sp.Matrix([relative_x, relative_y, relative_orientation])
        clf = (relative_state.T @ H @ relative_state)[0, 0]
        return clf

    def define_clf_cross_term(self):
        """ define the clf with croos term """
        relative_x = self.state[0] - self.target_state[0]
        relative_y = self.state[1] - self.target_state[1]
        relative_orientation = self.state[2] - self.target_state[2]
        H = sp.Matrix([[1.00, 0.00, 0.05], 
                       [0.00, 1.00, 0.05], 
                       [0.05, 0.05, 0.32]])  # 0.26
        
        relative_state = sp.Matrix([relative_x, relative_y, relative_orientation])
        clf = (relative_state.T @ H @ relative_state)[0, 0]
        return clf

    def time_differential_clf_center(self):
        """ contain the differential of the target position """
        self.dtarget_clf_symbolic = sp.Matrix([self.clf_symbolic]).jacobian(self.target_state)
        self.dtarget_clf = lambdify([self.state, self.target_state], self.dx_clf_symbolic) 

        # only change the x coordinate
        self.dt_clf_symbolic = (self.dtarget_clf_symbolic @ sp.Matrix([0.5, 0, 0]))[0, 0]
        self.dt_clf = lambdify([self.state, self.target_state], self.dt_clf_symbolic) 

    def define_cbf(self):

        relative_x = self.robot_state[0] + self.l * sp.cos(self.robot_state[2]) - self.obstacle_state[0]
        relative_y = self.robot_state[1] + self.l * sp.sin(self.robot_state[2]) - self.obstacle_state[1]
        cbf = relative_x ** 2 + relative_y ** 2 - (self.robot_state[3] + self.obstacle_state[2] + self.margin) ** 2
        return cbf
    
    def lie_derivatives_calculator(self):
        
        # dx_clf, lf_clf, lg_clf shape in (1, 3) and (1, 1) (1, u_dim)
        self.dx_clf_symbolic = sp.Matrix([self.clf_symbolic]).jacobian(self.state)
        self.dx_clf = lambdify([self.state, self.target_state], self.dx_clf_symbolic) 
        
        self.lf_clf_symbolic = (self.dx_clf_symbolic @ self.f_symbolic)[0, 0]
        self.lf_clf = lambdify([self.state, self.target_state], self.lf_clf_symbolic) 

        self.lg_clf_symbolic = self.dx_clf_symbolic @ self.g_symbolic
        self.lg_clf = lambdify([self.state, self.target_state], self.lg_clf_symbolic) 
        
        # dx_cbf, lf_cbf, lg_cbf shape in (1, 3) and (1, 1) (1, u_dim)
        self.dx_cbf_symbolic = sp.Matrix([self.cbf_symbolic]).jacobian(self.state)
        self.dx_cbf = lambdify([self.state, self.obstacle_state], self.dx_cbf_symbolic) 

        self.lf_cbf_symbolic = (self.dx_cbf_symbolic @ self.f_symbolic)[0, 0]
        self.lf_cbf = lambdify([self.state, self.obstacle_state], self.lf_cbf_symbolic) 

        self.lg_cbf_symbolic = self.dx_cbf_symbolic * self.g_symbolic
        self.lg_cbf = lambdify([self.state, self.obstacle_state], self.lg_cbf_symbolic) 

    def time_differential_cbf(self):
        """ calculate the other terms of cbf: dcbf / dt, and obstacle.dynamics as additional parameter """
        self.dox_cbf_symbolic = sp.Matrix([self.cbf_symbolic]).jacobian(self.obstacle_status)
        self.dox_cbf = lambdify([self.state, self.obstacle_state], self.dox_cbf_symbolic)

        self.dt_cbf_symbolic = (self.dox_cbf_symbolic @ self.obstacle_dynamics)[0, 0]
        self.dt_cbf = lambdify([self.state, self.obstacle_state, self.obstacle_dynamics], self.dt_cbf_symbolic)

    def __str__(self) -> str:
        return f'Class contains the states {self.state}, \n' + \
                f'system dynamic f {self.f} and g {self.g} \n' \
                f'CBF {self.cbf}, \n'
    
    def dynamic(self, x, u):
        return self.f(x) + self.g(x) @ np.array(u).reshape(self.control_dim, -1)
    
    def get_next_state(self, current_state, u, dt):
        """ Fourth-order Rungekutta method """
        f1 = self.dynamic(current_state, u).T[0]
        f2 = self.dynamic(current_state + dt * f1 / 2, u).T[0]
        f3 = self.dynamic(current_state + dt * f2 / 2, u).T[0]
        f4 = self.dynamic(current_state + dt * f3, u).T[0]

        next_state = current_state + dt / 6 * (f1 + 2 * f2 + 2 * f3 + f4)
        return next_state
    
    def next_state(self, current_state, u, dt):
        """ simple one step """
        next_state = current_state
        next_state = next_state + dt * (self.f(current_state).T[0] + (self.g(current_state) @ np.array(u).reshape(self.control_dim, -1)).T[0])

        return next_state
    
if __name__ == "__main__":
    test_target = Single_Vehicle_Model({'l': 0.1, 'margin': 0.05})

    # print(test_target.cbf(np.array([0.5, 0.5, np.pi/3, 0.3]), [2.0, 2.0, 0.3]))
    # print(test_target.dx_cbf([0.5, 0.5, np.pi/3], [2.0, 2.0, 0.3]))
    # print(test_target.lf_cbf([0.5, 0.5, np.pi/3], [2.0, 2.0, 0.3]))
    # print(test_target.lg_cbf([0.5, 0.5, np.pi/3], [2.0, 2.0, 0.3]))
    print(test_target.dox_cbf([0.5, 0.5, np.pi/3], [2.0, 2.0, 0.3]))
    print(test_target.dt_cbf([0.5, 0.5, np.pi/3], [2.0, 2.0, 0.3], [0.5, 0.28]))

    # print(test_target.clf([0.5, 0.5, 0], [2.0, 2.0, 0]))
    # print(test_target.dx_clf([0.5, 0.5, 0], [2.0, 2.0, 0]))
    # print(test_target.lf_clf([0.5, 0.5, 0], [2.0, 2.0, 0]))
    # print(test_target.lg_clf([0.5, 0.5, 0], [2.0, 2.0, 0]))

