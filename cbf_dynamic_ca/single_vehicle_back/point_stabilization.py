import numpy as np
import single_vehicle
import cbf_clf_qp
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches


# plt.rc('font',family='Times New Roman') 
class Point_Stabilization_Execute:
    def __init__(self) -> None:
        self.parameter = {
                    'weight_input': 0.5,
                    'weight_slack': 1000.0,
                    'clf_lambda': 0.4,  # 0.5
                    'cbf_gamma': 1.5,
                    'step_time': 0.1,
                    'u_max': [2.0, 1.5],
                    'u_min': [-2.0, -1.5],
                    'initial_state': [0.0, 0.0, 0.0],
                    'target_state': [5.0, 4.0, 0.0],  # static: [5.0, 4.0, 0.0] dynamic: [5.0, 4.2, 0.0]
                    'robot_radius': 0.25,
                    'l': 0.15,
                    'obstacle_list': [[3.0, 1.5, 0.7], [1.0, 3.0, 0.55]],
                    'obstacle_dynamics_list': [[0.0, 0.0], [0.0, 0.0]],
                    # 'obstacle_list': [[3.5, 1.5, 0.7], [5.0, 3.0, 0.6]],
                    # 'obstacle_dynamics_list': [[-0.5, 0.0], [-0.5, 0.0]],
                    'margin': 0.05
                }
        
        self.vehicle = single_vehicle.Single_Vehicle_Model(self.parameter)
        self.qp = cbf_clf_qp.CBF_CLF_Qp(self.vehicle, self.parameter)
        
        # initial the state of robot
        self.init_state = np.array(self.parameter['initial_state'])
        self.current_state = self.init_state.copy()
        self.robot_radius = self.parameter['robot_radius']
        self.l = self.parameter['l']

        self.target_state = np.array(self.parameter['target_state'])
        
        # state of obstacle_list
        self.obstacle_num = len(self.parameter['obstacle_list'])
        self.obstacle_init_state_list = np.array(self.parameter['obstacle_list'])
        self.obstacle_state_list = self.obstacle_init_state_list.copy()
        self.obstacle_dynamics_list = np.array(self.parameter['obstacle_dynamics_list'])
        
        self.T = 30
        self.step_time = self.parameter['step_time']
        self.time_steps = int(np.ceil(self.T / self.step_time))
        self.terminal_time = self.time_steps
        
        # storage
        self.xt = np.zeros((3, self.time_steps))
        self.obstacle_list_t = np.zeros((self.obstacle_num, 3, self.time_steps))
        self.ut = np.zeros((2, self.time_steps))
        self.slackt = np.zeros((1, self.time_steps))
        self.clf_t = np.zeros((1, self.time_steps))
        self.cbf_t = np.zeros((1, self.time_steps))

        # plot
        self.fig, self.ax = plt.subplots()
        self.robot_body = None
        self.robot_arrow = None

        self.start_circle = None
        self.start_arrow = None
        self.target_circle = None
        self.target_arrow = None
        self.obs = [None for i in range(self.obstacle_num)]

    def qp_solve_cbf_clf(self):
        """ solve the qp with cbf and clf"""
        u = np.zeros(2)
        t = 0
        process_time = []
        while np.linalg.norm(self.current_state - self.target_state) > 0.05 and t - self.time_steps < 0.0:
            if t % 100 == 0:
                print(f't = {t}')
            
            start_time = time.time()
            u, delta, cbf, clf, feas = self.qp.cbf_clf_qp(self.current_state, self.obstacle_state_list, 
                                                          self.obstacle_dynamics_list, u_pre=u)
            process_time.append(time.time() - start_time)

            if not feas:
                print('This problem is infeasible!')
                break
            else:
                pass
            
            self.xt[:, t] = np.copy(self.current_state)
            self.ut[:, t] = u
            self.slackt[:, t] = delta
            self.cbf_t[:, t] = cbf
            self.clf_t[:, t] = clf

            self.current_state = self.vehicle.next_state(self.current_state, u, self.step_time)
            # stop when the obstacle arrive at some place and update the state of obstacle
            for i in range(self.obstacle_num):
                # storage
                self.obstacle_list_t[i][:, t] = np.copy(self.obstacle_state_list[i])

                if self.obstacle_state_list[i][0] < 0.3:
                    self.obstacle_dynamics_list[i] = np.zeros((2, ))
                self.obstacle_state_list[i][0:2] = self.obstacle_state_list[i][0:2] + self.step_time * self.obstacle_dynamics_list[i]
            
            t = t + 1
        print('Finish the solve of qp with cbf and clf!')
        print('Average_time:', sum(process_time) / len(process_time))
        self.terminal_time = t

    def qp_solve_clf(self):
        """ solve the qp with clf """
        t = 0
        while np.linalg.norm(self.current_state - self.target_state) > 0.1 and t - self.time_steps < 0.0:
            if t % 100 == 0:
                print(f't = {t}')
    
            u, clf, feas = self.qp.clf_qp(self.current_state)
            if not feas:
                print('This problem is infeasible!')
                break
            else:
                pass
            
            self.xt[:, t] = np.copy(self.current_state)
            self.ut[:, t] = u
            self.clf_t[:, t] = clf

            self.current_state = self.vehicle.next_state(self.current_state, u, self.step_time)

            t = t + 1
        print('Finish the solve of qp with clf!')
        self.terminal_time = t

    def render(self):
        
        self.fig.set_size_inches(7, 6.5)
        self.ax.set_aspect('equal')

        # set the text in Times New Roman
        config = {
            "font.family": 'serif',
            "font.size": 12,
            "font.serif": ['Times New Roman'],
            "mathtext.fontset": 'stix',
        }
        plt.rcParams.update(config)

        self.ax.set_xlim(-1.0, 6)
        self.ax.set_ylim(-1.0, 6)

        # set the label in Times New Roman and size
        label_font = {'family': 'Times New Roman',
                      'weight': 'normal',
                      'size': 16,
                      }
        self.ax.set_xlabel('x (m)', label_font)
        self.ax.set_ylabel("y (m)", label_font)

        # set the tick in Times New Roman and size
        self.ax.tick_params(labelsize=16)
        labels = self.ax.get_xticklabels() + self.ax.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]
        
        self.animation_init()

        position_x = self.init_state[0] + self.l * np.cos(self.init_state[2])
        position_y = self.init_state[1] + self.l * np.sin(self.init_state[2])
        self.robot_body = mpatches.Circle(xy=(position_x, position_y), radius=self.robot_radius, color='r')  # fill = False
        self.ax.add_patch(self.robot_body)

        self.robot_arrow = mpatches.Arrow(position_x,
                                          position_y,
                                          self.robot_radius * np.cos(self.init_state[2]),
                                          self.robot_radius * np.sin(self.init_state[2]),
                                          width=0.15,
                                          color='k')
        self.ax.add_patch(self.robot_arrow)

        for i in range(self.obstacle_num):
            self.obs[i] = mpatches.Circle(xy=(self.obstacle_init_state_list[i][0], self.obstacle_init_state_list[i][1]), 
                                          radius=self.obstacle_init_state_list[i][2], 
                                          color='k')
            self.ax.add_patch(self.obs[i])

        self.ani = animation.FuncAnimation(self.fig, 
                                           func=self.animation_loop, 
                                           frames=self.terminal_time, 
                                           init_func=self.animation_init, 
                                           interval=20, 
                                           repeat=False)
        plt.grid()

        # writergif = animation.PillowWriter(fps=30) 
        # self.ani.save('.gif', writer=writergif)

        writer = animation.PillowWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)
        self.ani.save('static.gif', writer=writer)

        plt.show()
        
    def animation_init(self):

        # start position
        start_position_x = self.init_state[0] + self.l * np.cos(self.init_state[2])
        start_position_y = self.init_state[1] + self.l * np.sin(self.init_state[2])

        self.start_circle = mpatches.Circle(xy=(start_position_x, start_position_y), radius=self.robot_radius, color='silver')
        self.ax.add_patch(self.start_circle)
        self.start_circle.set_zorder(0)

        self.start_arrow = mpatches.Arrow(start_position_x,
                                          start_position_y,
                                          self.robot_radius * np.cos(self.init_state[2]),
                                          self.robot_radius * np.sin(self.init_state[2]),
                                          width=0.15, color='k')
        self.ax.add_patch(self.start_arrow)
        self.start_arrow.set_zorder(1)

        # goal position
        target_position_x = self.target_state[0] + self.l * np.cos(self.target_state[2])
        target_position_y = self.target_state[1] + self.l * np.sin(self.target_state[2])

        self.target_circle = mpatches.Circle(xy=(target_position_x, target_position_y), radius=self.robot_radius, color='silver')
        self.ax.add_patch(self.target_circle)
        self.target_circle.set_zorder(0)

        self.target_arrow = mpatches.Arrow(target_position_x,
                                           target_position_y,
                                           self.robot_radius * np.cos(self.target_state[2]),
                                           self.robot_radius * np.sin(self.target_state[2]),
                                           width=0.15, color='k')
        self.ax.add_patch(self.target_arrow)
        self.target_arrow.set_zorder(1)

        return self.ax.patches + self.ax.texts + self.ax.artists

    def animation_loop(self, indx):

        self.robot_arrow.remove()
        self.robot_body.remove()
        for i in range(self.obstacle_num):
            self.obs[i].remove()

        position = self.xt[:, indx][:2]
        orientation = self.xt[:, indx][2]
        position_x = position[0] + self.l * np.cos(orientation)
        position_y = position[1] + self.l * np.sin(orientation)

        self.robot_body = mpatches.Circle(xy=(position_x, position_y), radius=self.robot_radius, color='r')
        self.ax.add_patch(self.robot_body)

        self.robot_arrow = mpatches.Arrow(position_x,
                                          position_y,
                                          self.robot_radius * np.cos(orientation),
                                          self.robot_radius * np.sin(orientation),
                                          width=0.15,
                                          color='k')
        self.ax.add_patch(self.robot_arrow)

        # add for obstacle
        for i in range(self.obstacle_num):
            self.obs[i] = mpatches.Circle(xy=(self.obstacle_list_t[i][:, indx][0], self.obstacle_list_t[i][:, indx][1]), 
                                          radius=self.obstacle_init_state_list[i][2], 
                                          color='k')
            self.ax.add_patch(self.obs[i])

        if indx != 0:
            # show past trajecotry of robot
            x_list = [self.xt[:, indx - 1][0] + self.l * np.cos(self.xt[:, indx-1][2]), self.xt[:, indx][0] + self.l * np.cos(self.xt[:, indx][2])]
            y_list = [self.xt[:, indx - 1][1] + self.l * np.sin(self.xt[:, indx-1][2]), self.xt[:, indx][1] + self.l * np.sin(self.xt[:, indx][2])]
        
            self.ax.plot(x_list, y_list, color='b',)

            # show past trajecotry of each obstacle
            for i in range(self.obstacle_num):
                ox_list = [self.obstacle_list_t[i][:, indx - 1][0], self.obstacle_list_t[i][:, indx][0]]
                oy_list = [self.obstacle_list_t[i][:, indx - 1][1], self.obstacle_list_t[i][:, indx][1]]
                self.ax.plot(ox_list, oy_list, color='k',)

        plt.savefig('figure/{}.png'.format(indx), format='png', dpi=300)
        return self.ax.patches + self.ax.texts + self.ax.artists

    def show_clf(self):
        """show the changes in clf over time"""
        t = np.arange(0, self.terminal_time / 10, self.step_time)
        plt.plot(t, self.clf_t[0][0: self.terminal_time].reshape(self.terminal_time, ), linewidth=3, color='cyan')

        # set the label in Times New Roman and size
        label_font = {'family': 'Times New Roman',
                      'weight': 'normal',
                      'size': 16,
                      }
        plt.title('Control Lyapunov Function', label_font)
        plt.xlabel('Time (s)', label_font)
        plt.ylabel('V(x)', label_font)

        # set the tick in Times New Roman and size
        self.ax.tick_params(labelsize=16)
        labels = self.ax.get_xticklabels() + self.ax.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]

        plt.grid() 
        plt.savefig('clf.png', format='png', dpi=300)
        plt.show()
        plt.close(self.fig)

    def show_slack(self):
        """show the changes in slack over time"""
        t = np.arange(0, self.terminal_time / 10, self.step_time)
        plt.plot(t, self.slackt[0][0: self.terminal_time].reshape(self.terminal_time, ), linewidth=3, color='orange')

        # set the label in Times New Roman and size
        label_font = {'family': 'Times New Roman',
                      'weight': 'normal',
                      'size': 16,
                      }
        plt.title('Relaxation Variable', label_font)
        plt.xlabel('Time (s)', label_font)
        plt.ylabel('δ', label_font)

        # set the tick in Times New Roman and size
        self.ax.tick_params(labelsize=16)
        labels = self.ax.get_xticklabels() + self.ax.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]

        plt.grid() 
        plt.savefig('slack.png', format='png', dpi=300)
        plt.show()
        plt.close(self.fig)

    def show_cbf(self):
        """show the changes in cbf over time"""
        t = np.arange(0, self.terminal_time / 10, self.step_time)
        plt.plot(t, self.cbf_t[0][0: self.terminal_time].reshape(self.terminal_time, ), linewidth=3, color='red')

        # set the label in Times New Roman and size
        label_font = {'family': 'Times New Roman',
                      'weight': 'normal',
                      'size': 16,
                      }
        plt.title('Control Barrier Function', label_font)
        plt.xlabel('Time (s)', label_font)
        plt.ylabel('h(x)', label_font)

        # set the tick in Times New Roman and size
        self.ax.tick_params(labelsize=16)
        labels = self.ax.get_xticklabels() + self.ax.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]

        plt.grid() 
        # plt.savefig('cbf.png', format='png', dpi=300)
        plt.show()
        plt.close(self.fig)

    def show_control(self):
        """show the changes in control over time"""
        # set the label in Times New Roman and size
        label_font = {'family': 'Times New Roman',
                      'weight': 'normal',
                      'size': 16,
                      }
        
        # u_max and u_min
        u_max = self.parameter['u_max']
        t = np.arange(0, self.terminal_time / 10, self.step_time)

        plt.plot(t, self.ut[0][0: self.terminal_time].reshape(self.terminal_time, ), linewidth=3, color='b', label="v")        
        plt.plot(t, u_max[0] * np.ones(t.shape[0]), 'b--')
        plt.plot(t, -u_max[0] * np.ones(t.shape[0]), 'b--')

        plt.plot(t, self.ut[1][0: self.terminal_time].reshape(self.terminal_time, ), linewidth=3, color='r', label="w")
        plt.plot(t, u_max[1] * np.ones(t.shape[0]), 'r--')
        plt.plot(t, -u_max[1] * np.ones(t.shape[0]), 'r--')
        
        plt.title('Control Variables', label_font)
        plt.xlabel('Time (s)', label_font)
        plt.ylabel('v (m/s) / w (rad/s)', label_font)
       
        self.ax.tick_params(labelsize=16)
        labels = self.ax.get_xticklabels() + self.ax.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]
        
        legend_font = {'family': 'Times New Roman',  'weight': 'normal', 'size': 12}
        plt.legend(loc='upper right', prop=legend_font)
        plt.grid() 
        plt.savefig('control.png', format='png', dpi=300)
        plt.show()
        plt.close(self.fig)

    def show_control_v(self):
        """show the changes in vover time"""
        # set the label in Times New Roman and size
        label_font = {'family': 'Times New Roman',
                      'weight': 'normal',
                      'size': 16,
                      }
        
        # u_max and u_min
        u_max = self.parameter['u_max']
        t = np.arange(0, self.terminal_time / 10, self.step_time)

        plt.plot(t, self.ut[0][0: self.terminal_time].reshape(self.terminal_time, ), linewidth=3, color='b', label="v")        
        plt.plot(t, u_max[0] * np.ones(t.shape[0]), 'b--')
        plt.plot(t, -u_max[0] * np.ones(t.shape[0]), 'b--')
        
        plt.title('Control Variable (v) ', label_font)
        plt.xlabel('Time (s)', label_font)
        plt.ylabel('v (m/s)', label_font)
       
        self.ax.tick_params(labelsize=16)
        labels = self.ax.get_xticklabels() + self.ax.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]
        
        plt.grid() 
        plt.savefig('control_v.png', format='png', dpi=300)
        plt.show()
        plt.close(self.fig)

    def show_control_w(self):
        """show the changes in control over time"""
        # set the label in Times New Roman and size
        label_font = {'family': 'Times New Roman',
                      'weight': 'normal',
                      'size': 16,
                      }
        
        # u_max and u_min
        u_max = self.parameter['u_max']
        t = np.arange(0, self.terminal_time / 10, self.step_time)

        plt.plot(t, self.ut[1][0: self.terminal_time].reshape(self.terminal_time, ), linewidth=3, color='r', label="w")
        plt.plot(t, u_max[1] * np.ones(t.shape[0]), 'r--')
        plt.plot(t, -u_max[1] * np.ones(t.shape[0]), 'r--')
        
        plt.title('Control Variable (w)', label_font)
        plt.xlabel('Time (s)', label_font)
        plt.ylabel('w (rad/s)', label_font)
       
        self.ax.tick_params(labelsize=16)
        labels = self.ax.get_xticklabels() + self.ax.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]
        
        plt.grid() 
        plt.savefig('control_w.png', format='png', dpi=300)
        plt.show()
        plt.close(self.fig)

    def storage_data(self):
        np.savez('process_data', x=self.xt, slack=self.slackt, cbf=self.cbf_t, clf=self.clf_t, u=self.ut)

if __name__ == '__main__':
    test_target = Point_Stabilization_Execute()
    # test_target.qp_solve_clf()
    test_target.qp_solve_cbf_clf()
    
    test_target.render()
    # test_target.show_control_v()
    # test_target.show_control_w()
    # test_target.show_clf()
    # test_target.show_slack()
    # # test_target.show_cbf()
    # test_target.show_control()
    
