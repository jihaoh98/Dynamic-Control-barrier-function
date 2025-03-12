import numpy as np
import casadi as ca


class CBF_CLF_Qp:
    def __init__(self, system, parameter) -> None:

        # dimension
        self.state_dim = system.state_dim 
        self.control_dim = system.control_dim

        # get the parameter for optimal control
        self.weight_input = parameter['weight_input']
        self.weight_slack = parameter['weight_slack']
        self.clf_lambda = parameter['clf_lambda']
        self.cbf_gamma = parameter['cbf_gamma']

        self.u_max = parameter['u_max']
        self.u_min = parameter['u_min']
        self.target_state = parameter['target_state']
        self.robot_radius = parameter['robot_radius']

        # CLF
        self.clf = system.clf
        self.lf_clf = system.lf_clf
        self.lg_clf = system.lg_clf

        # CBF
        self.cbf = system.cbf
        self.lf_cbf = system.lf_cbf
        self.lg_cbf = system.lg_cbf
        self.dt_cbf = system.dt_cbf
        
        # optimize
        self.opti = ca.Opti()
        # solver
        opts_setting = {
            'ipopt.max_iter': 100,
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.acceptable_tol': 1e-8,
            'ipopt.acceptable_obj_change_tol': 1e-6
        }
        self.opti.solver('ipopt', opts_setting)

        # optimize variable
        self.u = self.opti.variable(self.control_dim)
        self.slack = self.opti.variable()
        
        self.obj = None
        self.H = None
        self.feasible = None

    def set_objective_function(self, u_ref, u_pre):

        self.H = np.array([[1.0, 0.0],
                           [0.0, 0.05]])
        
        self.R = np.array([[1.0, 0.0],
                           [0.0, 1.0]])
        
        # object function
        self.obj = (self.u - u_ref).T @ self.H @ (self.u - u_ref)
        self.obj = self.obj + self.weight_slack * self.slack ** 2
        self.obj = self.obj + 0.25 * (self.u - u_pre).T @ self.R @ (self.u - u_pre)
        self.opti.minimize(self.obj)

    def set_objective_function_clf(self, u_ref):
        self.H = np.array([[1.0, 0.0],
                           [0.0, 0.10]])
        
        # object function
        self.obj = (self.u - u_ref).T @ self.H @ (self.u - u_ref)
        self.opti.minimize(self.obj)

    def clf_qp(self, current_state, u_ref=None):
        """ current_state: [x, y, theta], only with clf qp """
        if u_ref is None:
            u_ref = np.zeros(self.control_dim)

        # empty the constraint set and set the objective function
        self.set_objective_function_clf(u_ref)
        self.opti.subject_to()
    
        # constraint forclf: LfV + LgV * u + lambda * V <= 0
        self.opti.subject_to(self.opti.bounded(self.u_min, self.u, self.u_max))

        clf = self.clf(current_state, self.target_state)
        lf_clf = self.lf_clf(current_state, self.target_state)
        lg_clf = self.lg_clf(current_state, self.target_state)
        self.opti.subject_to(lf_clf + (lg_clf @ self.u)[0, 0] + self.clf_lambda * clf <= 0)

        # optimize the Qp problem
        try:
            sol = self.opti.solve()
            self.feasible = True
        except:
            print(self.opti.return_status())
            self.feasible = False

        optimal_control = sol.value(self.u)
        return optimal_control, clf, self.feasible

    def cbf_clf_qp(self, current_state, obstacle_state_list, obstacle_dynamics_list, u_pre, u_ref=None):
        """
        current_state: [x, y, theta]
        obstacle_state: [x, y, radius]
        input_state for cbf: [x, y, theta, radius]
        """
        if u_ref is None:
            u_ref = np.zeros(self.control_dim)
 
        # empty the constraint set and set the objective function
        self.opti.subject_to()
        self.set_objective_function(u_ref, u_pre)

        # constraint
        self.opti.subject_to(self.opti.bounded(self.u_min, self.u, self.u_max))
        self.opti.subject_to(self.opti.bounded(-np.inf, self.slack, np.inf))

        # constraint for clf: LfV + LgV * u + lambda * V <= slack
        clf = self.clf(current_state, self.target_state)
        lf_clf = self.lf_clf(current_state, self.target_state)
        lg_clf = self.lg_clf(current_state, self.target_state)
        self.opti.subject_to(lf_clf + (lg_clf @ self.u)[0, 0] + self.clf_lambda * clf - self.slack <= 0)

        # constraint for cbf: LfB + LgB * u + gamma * B  >= 0
        cbf_list = []
        robot_state = [current_state[0], current_state[1], current_state[2], self.robot_radius]
        for i in range(len(obstacle_state_list)):
            cbf = self.cbf(robot_state, obstacle_state_list[i])
            lf_cbf = self.lf_cbf(current_state, obstacle_state_list[i])
            lg_cbf = self.lg_cbf(current_state, obstacle_state_list[i]) 
            
            dt_cbf = self.dt_cbf(current_state, obstacle_state_list[i], obstacle_dynamics_list[i])
            self.opti.subject_to(lf_cbf + (lg_cbf @ self.u)[0, 0] + dt_cbf + self.cbf_gamma * cbf >= 0)
            cbf_list.append(cbf)

        # optimize the Qp problem
        try:
            sol = self.opti.solve()
            self.feasible = True
        except:
            print(self.opti.return_status())
            self.feasible = False

        optimal_control = sol.value(self.u)
        slack = sol.value(self.slack)
        
        return optimal_control, slack, cbf_list, clf, self.feasible