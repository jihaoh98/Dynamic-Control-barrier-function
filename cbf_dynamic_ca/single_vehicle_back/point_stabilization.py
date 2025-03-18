import numpy as np
import single_vehicle
import cbf_clf_qp
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
from matplotlib.ticker import FormatStrFormatter


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
        self.cbf_t = np.zeros((self.obstacle_num, self.time_steps))

        # plot
        self.fig, self.ax = plt.subplots()
        self.robot_body = None
        self.robot_arrow = None

        self.start_circle = None
        self.start_arrow = None
        self.target_circle = None
        self.target_arrow = None
        self.obs = [None for i in range(self.obstacle_num)]

        config = {
            "font.family": 'serif',
            "font.size": 12,
            "font.serif": ['Times New Roman'],
            "mathtext.fontset": 'stix',
        }
        plt.rcParams.update(config)

    def qp_solve_cbf_clf(self):
        """ solve the qp with cbf and clf"""
        u = np.zeros(2)
        t = 0
        process_time = []
        while np.linalg.norm(self.current_state - self.target_state) > 0.05 and t - self.time_steps < 0.0:
            if t % 100 == 0:
                print(f't = {t}')
            
            start_time = time.time()
            u, delta, cbf, clf, feas = self.qp.cbf_clf_qp(
                self.current_state, self.obstacle_state_list, 
                self.obstacle_dynamics_list, u_pre=u
            )
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
        label_font = {
            'family': 'Times New Roman',
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

        self.robot_arrow = mpatches.Arrow(
            position_x,
            position_y,
            self.robot_radius * np.cos(self.init_state[2]),
            self.robot_radius * np.sin(self.init_state[2]),
            width=0.15,
            color='k'
        )
        self.ax.add_patch(self.robot_arrow)

        for i in range(self.obstacle_num):
            self.obs[i] = mpatches.Circle(
                xy=(self.obstacle_init_state_list[i][0], self.obstacle_init_state_list[i][1]), 
                radius=self.obstacle_init_state_list[i][2], 
                color='k'
            )
            self.ax.add_patch(self.obs[i])

        self.ani = animation.FuncAnimation(
            self.fig, 
            func=self.animation_loop, 
            frames=self.terminal_time, 
            init_func=self.animation_init, 
            interval=20, 
            repeat=False
        )
        plt.grid()

        # writergif = animation.PillowWriter(fps=30) 
        # self.ani.save('.gif', writer=writergif)

        # writer = animation.PillowWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)
        # self.ani.save('static.gif', writer=writer)

        plt.show()
        
    def animation_init(self):

        # start position
        start_position_x = self.init_state[0] + self.l * np.cos(self.init_state[2])
        start_position_y = self.init_state[1] + self.l * np.sin(self.init_state[2])

        self.start_circle = mpatches.Circle(
            xy=(start_position_x, start_position_y), 
            radius=self.robot_radius, 
            color='silver'
        )
        self.ax.add_patch(self.start_circle)
        self.start_circle.set_zorder(0)

        self.start_arrow = mpatches.Arrow(
            start_position_x,
            start_position_y,
            self.robot_radius * np.cos(self.init_state[2]),
            self.robot_radius * np.sin(self.init_state[2]),
            width=0.15, color='k'
        )
        self.ax.add_patch(self.start_arrow)
        self.start_arrow.set_zorder(1)

        # goal position
        target_position_x = self.target_state[0] + self.l * np.cos(self.target_state[2])
        target_position_y = self.target_state[1] + self.l * np.sin(self.target_state[2])

        self.target_circle = mpatches.Circle(xy=(target_position_x, target_position_y), radius=self.robot_radius, color='silver')
        self.ax.add_patch(self.target_circle)
        self.target_circle.set_zorder(0)

        self.target_arrow = mpatches.Arrow(
            target_position_x,
            target_position_y,
            self.robot_radius * np.cos(self.target_state[2]),
            self.robot_radius * np.sin(self.target_state[2]),
            width=0.15, color='k'
        )
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

        self.robot_arrow = mpatches.Arrow(
            position_x,
            position_y,
            self.robot_radius * np.cos(orientation),
            self.robot_radius * np.sin(orientation),
            width=0.15,
            color='k'
        )
        self.ax.add_patch(self.robot_arrow)

        # add for obstacle
        for i in range(self.obstacle_num):
            self.obs[i] = mpatches.Circle(
                xy=(self.obstacle_list_t[i][:, indx][0], self.obstacle_list_t[i][:, indx][1]), 
                radius=self.obstacle_init_state_list[i][2], 
                color='k'
            )
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
    
    def show_static_obs(self, index_t):
        font_path = "/home/hjh/simfang.ttf"  
        fangsong_font = fm.FontProperties(fname=font_path, size=22)

        label_font = {
            'family': 'Times New Roman', 
            'weight': 'normal', 
            'size': 30,
        }

        figure, ax = plt.subplots(figsize=(10, 9))
        figure.set_dpi(300)
        ax.set_aspect('equal')
        ax.set_xlim(-1.0, 6.0)
        ax.set_ylim(-1.0, 6.0)

        plt.xlabel("x (m)", label_font)
        plt.ylabel("y (m)", label_font)
        plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

        ax.tick_params(labelsize=35)
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]

        start_color = ['#AAD2E6', '#90BEE0', '#4B74B2', '#3C6478']
        trajecotry_color = ['#4B74B2']

        # start position
        start_position_x = self.init_state[0] + self.l * np.cos(self.init_state[2])
        start_position_y = self.init_state[1] + self.l * np.sin(self.init_state[2])

        start_body = mpatches.Circle(
            xy=(start_position_x, start_position_y), 
            radius=self.robot_radius, 
            color=start_color[0]
        )
        start_body.set_zorder(2)
        ax.add_patch(start_body)

        start_body_add = mpatches.Circle(
            xy=(start_position_x, start_position_y), 
            radius=self.robot_radius + 0.05, 
            color=start_color[1]
        )
        start_body_add.set_zorder(1)
        ax.add_patch(start_body_add)

        start_arrow = mpatches.Arrow(
            start_position_x,
            start_position_y,
            self.robot_radius * np.cos(self.init_state[2]),
            self.robot_radius * np.sin(self.init_state[2]),
            width=0.2, 
            color='k'
        )
        start_arrow.set_zorder(3)
        ax.add_patch(start_arrow) 

        # target
        target_position_x = self.target_state[0] + self.l * np.cos(self.target_state[2])
        target_position_y = self.target_state[1] + self.l * np.sin(self.target_state[2])
        plt.plot(target_position_x, target_position_y, color='#8EB69C', marker="*", markersize=30, zorder=5)

        # trajectory
        tra_x = self.xt[0, 0: self.terminal_time + 1]
        tra_y = self.xt[1, 0: self.terminal_time + 1]
        plt.plot(tra_x, tra_y, color=trajecotry_color[0], linewidth=4, zorder=0)

        # final position
        final_position_x = self.xt[0, self.terminal_time - 1] + self.l * np.cos(self.xt[2, self.terminal_time - 1])
        final_position_y = self.xt[1, self.terminal_time - 1] + self.l * np.sin(self.xt[2, self.terminal_time - 1])
        robot_final = mpatches.Circle(
            xy=(final_position_x, final_position_y),
            radius=self.robot_radius,
            color=start_color[2],
            lw=4
        )
        robot_final.set_zorder(2)
        ax.add_patch(robot_final)

        robot_final_arrow = mpatches.Arrow(
            final_position_x,
            final_position_y,
            self.robot_radius * np.cos(self.xt[2, self.terminal_time - 1]),
            self.robot_radius * np.sin(self.xt[2, self.terminal_time - 1]),
            width=0.2,
            color='k'
        )
        robot_final_arrow.set_zorder(3)
        ax.add_patch(robot_final_arrow)

        # obstacle final
        obs_final = [None for i in range(self.obstacle_num)]
        for i in range(self.obstacle_num):
            obs_final[i] = mpatches.Circle(
                xy=(self.obstacle_list_t[i][0, self.terminal_time - 1], self.obstacle_list_t[i][1, self.terminal_time - 1]),
                radius=self.obstacle_init_state_list[i][2],
                fill=False,
                lw=4,
                linestyle='--',
                zorder=0
            )
            ax.add_patch(obs_final[i])

        # dynamic obstacle and robot
        robot = [None for i in range(len(index_t))]
        robot_arrow = [None for i in range(len(index_t))]
        obstacle = [[None for j in range(len(index_t))] for i in range(self.obstacle_num)]
        alpha_index = [0.3, 0.7]
        for i in range(len(index_t)):
            rx = self.xt[0, index_t[i]] + self.l * np.cos(self.xt[2, index_t[i]])
            ry = self.xt[1, index_t[i]] + self.l * np.sin(self.xt[2, index_t[i]])
            robot[i] = mpatches.Circle(
                xy=(rx, ry),
                radius=self.robot_radius,
                edgecolor=start_color[2], 
                fill=False,
                lw=4,
                alpha=alpha_index[i]
            )
            robot[i].set_zorder(2)
            ax.add_patch(robot[i])

            robot_arrow[i] = mpatches.Arrow(
                rx, ry,
                self.robot_radius * np.cos(self.xt[2, index_t[i]]),
                self.robot_radius * np.sin(self.xt[2, index_t[i]]),
                width=0.2,
                color='k',
                alpha=alpha_index[i]
            )
            robot_arrow[i].set_zorder(3)
            ax.add_patch(robot_arrow[i])

        start_proxy = plt.scatter([], [], s=500, edgecolor=start_color[1], facecolor=start_color[0], linewidths=2)
        tar_proxy = plt.scatter([], [], s=500, edgecolor='#8EB69C', facecolor='#8EB69C', marker='*')
        obs_proxy = plt.scatter([], [], s=500, edgecolor='k', linestyle='--', facecolor='none', linewidths=1.5)
        plt.legend(
            handles=[start_proxy, tar_proxy, obs_proxy], 
            labels=['起点', '目标点', '障碍物'], loc='upper left',
            prop=fangsong_font
        )

        plt.savefig('static.png', format='png', dpi=300, bbox_inches='tight')

    def show_static_fail(self):
        terminal_time = 24
        font_path = "/home/hjh/simfang.ttf"  
        fangsong_font = fm.FontProperties(fname=font_path, size=22)

        label_font = {
            'family': 'Times New Roman', 
            'weight': 'normal', 
            'size': 30,
        }

        figure, ax = plt.subplots(figsize=(10, 9))
        figure.set_dpi(300)
        ax.set_aspect('equal')
        ax.set_xlim(-1.0, 6.0)
        ax.set_ylim(-1.0, 6.0)

        plt.xlabel("x (m)", label_font)
        plt.ylabel("y (m)", label_font)
        plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

        ax.tick_params(labelsize=35)
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]

        start_color = ['#AAD2E6', '#90BEE0', '#4B74B2', '#3C6478']
        trajecotry_color = ['#4B74B2']

        # start position
        start_position_x = self.init_state[0] + self.l * np.cos(self.init_state[2])
        start_position_y = self.init_state[1] + self.l * np.sin(self.init_state[2])

        start_body = mpatches.Circle(
            xy=(start_position_x, start_position_y), 
            radius=self.robot_radius, 
            color=start_color[0]
        )
        start_body.set_zorder(2)
        ax.add_patch(start_body)

        start_body_add = mpatches.Circle(
            xy=(start_position_x, start_position_y), 
            radius=self.robot_radius + 0.05, 
            color=start_color[1]
        )
        start_body_add.set_zorder(1)
        ax.add_patch(start_body_add)

        start_arrow = mpatches.Arrow(
            start_position_x,
            start_position_y,
            self.robot_radius * np.cos(self.init_state[2]),
            self.robot_radius * np.sin(self.init_state[2]),
            width=0.2, 
            color='k'
        )
        start_arrow.set_zorder(3)
        ax.add_patch(start_arrow) 

        # target
        target_position_x = self.target_state[0] + self.l * np.cos(self.target_state[2])
        target_position_y = self.target_state[1] + self.l * np.sin(self.target_state[2])
        plt.plot(target_position_x, target_position_y, color='#8EB69C', marker="*", markersize=30, zorder=5)

        tra_x = self.xt[0, 0: terminal_time + 1]
        tra_y = self.xt[1, 0: terminal_time + 1]
        plt.plot(tra_x, tra_y, color=trajecotry_color[0], linewidth=4, zorder=0)

        final_position_x = self.xt[0, terminal_time] + self.l * np.cos(self.xt[2, terminal_time])
        final_position_y = self.xt[1, terminal_time] + self.l * np.sin(self.xt[2, terminal_time])
        robot_final = mpatches.Circle(
            xy=(final_position_x, final_position_y),
            radius=self.robot_radius,
            color=start_color[2],
            lw=4,
            fill=False
        )
        robot_final.set_zorder(2)
        ax.add_patch(robot_final)

        robot_final_arrow = mpatches.Arrow(
            final_position_x,
            final_position_y,
            self.robot_radius * np.cos(self.xt[2, self.terminal_time - 1]),
            self.robot_radius * np.sin(self.xt[2, self.terminal_time - 1]),
            width=0.2,
            color='k'
        )
        robot_final_arrow.set_zorder(3)
        ax.add_patch(robot_final_arrow)

        obs_final = [None for i in range(self.obstacle_num)]
        for i in range(self.obstacle_num):
            obs_final[i] = mpatches.Circle(
                xy=(self.obstacle_list_t[i][0, terminal_time - 1], self.obstacle_list_t[i][1, terminal_time - 1]),
                radius=self.obstacle_init_state_list[i][2],
                fill=False,
                lw=4,
                linestyle='--',
                zorder=0
            )
            ax.add_patch(obs_final[i])

        start_proxy = plt.scatter([], [], s=500, edgecolor=start_color[1], facecolor=start_color[0], linewidths=2)
        tar_proxy = plt.scatter([], [], s=500, edgecolor='#8EB69C', facecolor='#8EB69C', marker='*')
        obs_proxy = plt.scatter([], [], s=500, edgecolor='k', linestyle='--', facecolor='none', linewidths=1.5)
        plt.legend(
            handles=[start_proxy, tar_proxy, obs_proxy], 
            labels=['起点', '目标点', '障碍物'], loc='upper left',
            prop=fangsong_font
        )

        plt.savefig('static_fail.png', format='png', dpi=300, bbox_inches='tight')

    def show_dynamic_obs(self, index_t=[0, 0]):
        font_path = "/home/hjh/simfang.ttf"  
        fangsong_font = fm.FontProperties(fname=font_path, size=22)

        label_font = {
            'family': 'Times New Roman', 
            'weight': 'normal', 
            'size': 30,
        }

        figure, ax = plt.subplots(figsize=(10, 9))
        figure.set_dpi(300)
        ax.set_aspect('equal')
        ax.set_xlim(-1.0, 6.0)
        ax.set_ylim(-1.0, 6.0)

        plt.xlabel("x (m)", label_font)
        plt.ylabel("y (m)", label_font)
        plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

        ax.tick_params(labelsize=35)
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]

        start_color = ['#AAD2E6', '#90BEE0', '#4B74B2', '#3C6478']
        trajecotry_color = ['#4B74B2']

        # start position
        start_position_x = self.init_state[0] + self.l * np.cos(self.init_state[2])
        start_position_y = self.init_state[1] + self.l * np.sin(self.init_state[2])

        start_body = mpatches.Circle(
            xy=(start_position_x, start_position_y), 
            radius=self.robot_radius, 
            color=start_color[0]
        )
        start_body.set_zorder(2)
        ax.add_patch(start_body)

        start_body_add = mpatches.Circle(
            xy=(start_position_x, start_position_y), 
            radius=self.robot_radius + 0.05, 
            color=start_color[1]
        )
        start_body_add.set_zorder(1)
        ax.add_patch(start_body_add)

        start_arrow = mpatches.Arrow(
            start_position_x,
            start_position_y,
            self.robot_radius * np.cos(self.init_state[2]),
            self.robot_radius * np.sin(self.init_state[2]),
            width=0.2, 
            color='k'
        )
        start_arrow.set_zorder(3)
        ax.add_patch(start_arrow) 

        # target
        target_position_x = self.target_state[0] + self.l * np.cos(self.target_state[2])
        target_position_y = self.target_state[1] + self.l * np.sin(self.target_state[2])
        plt.plot(target_position_x, target_position_y, color='#8EB69C', marker="*", markersize=30, zorder=5)

        # trajectory
        tra_x = self.xt[0, 0: self.terminal_time + 1]
        tra_y = self.xt[1, 0: self.terminal_time + 1]
        plt.plot(tra_x, tra_y, color=trajecotry_color[0], linewidth=4, zorder=0)

        # obstacle
        for i in range(self.obstacle_num):
            plt.plot(self.obstacle_list_t[i][0, 0], self.obstacle_list_t[i][1, 0], color='k', marker="s", markersize=15, zorder=0)
            plt.plot(
                self.obstacle_list_t[i][0, :self.terminal_time + 1], 
                self.obstacle_list_t[i][1, :self.terminal_time + 1], 
                color='k', linewidth=4, linestyle='--', zorder=1
            )

        # final position
        final_position_x = self.xt[0, self.terminal_time - 1] + self.l * np.cos(self.xt[2, self.terminal_time - 1])
        final_position_y = self.xt[1, self.terminal_time - 1] + self.l * np.sin(self.xt[2, self.terminal_time - 1])
        robot_final = mpatches.Circle(
            xy=(final_position_x, final_position_y),
            radius=self.robot_radius,
            color=start_color[2],
            lw=4
        )
        robot_final.set_zorder(2)
        ax.add_patch(robot_final)

        robot_final_arrow = mpatches.Arrow(
            final_position_x,
            final_position_y,
            self.robot_radius * np.cos(self.xt[2, self.terminal_time - 1]),
            self.robot_radius * np.sin(self.xt[2, self.terminal_time - 1]),
            width=0.2,
            color='k'
        )
        robot_final_arrow.set_zorder(3)
        ax.add_patch(robot_final_arrow)

        # obstacle final
        obs_final = [None for i in range(self.obstacle_num)]
        for i in range(self.obstacle_num):
            obs_final[i] = mpatches.Circle(
                xy=(self.obstacle_list_t[i][0, self.terminal_time - 1], self.obstacle_list_t[i][1, self.terminal_time - 1]),
                radius=self.obstacle_init_state_list[i][2],
                fill=False,
                lw=4,
                linestyle='--',
                zorder=0
            )
            ax.add_patch(obs_final[i])

        # dynamic obstacle and robot
        robot = [None for i in range(len(index_t))]
        robot_arrow = [None for i in range(len(index_t))]
        obstacle = [[None for j in range(len(index_t))] for i in range(self.obstacle_num)]
        alpha_index = [0.3, 0.7]
        for i in range(len(index_t)):
            rx = self.xt[0, index_t[i]] + self.l * np.cos(self.xt[2, index_t[i]])
            ry = self.xt[1, index_t[i]] + self.l * np.sin(self.xt[2, index_t[i]])
            robot[i] = mpatches.Circle(
                xy=(rx, ry),
                radius=self.robot_radius,
                edgecolor=start_color[2], 
                fill=False,
                lw=4,
                alpha=alpha_index[i]
            )
            robot[i].set_zorder(2)
            ax.add_patch(robot[i])

            robot_arrow[i] = mpatches.Arrow(
                rx, ry,
                self.robot_radius * np.cos(self.xt[2, index_t[i]]),
                self.robot_radius * np.sin(self.xt[2, index_t[i]]),
                width=0.2,
                color='k',
                alpha=alpha_index[i]
            )
            robot_arrow[i].set_zorder(3)
            ax.add_patch(robot_arrow[i])

            for j in range(self.obstacle_num):
                ox = self.obstacle_list_t[j][0, index_t[i]]
                oy = self.obstacle_list_t[j][1, index_t[i]]
                obstacle[j][i] = mpatches.Circle(
                    xy=(ox, oy),
                    radius=self.obstacle_init_state_list[j][2],
                    edgecolor='k',
                    fill=False,
                    lw=4,
                    linestyle='--',
                    alpha=alpha_index[i],
                    zorder=0
                )
                ax.add_patch(obstacle[j][i])

        start_proxy = plt.scatter([], [], s=500, edgecolor=start_color[1], facecolor=start_color[0], linewidths=2)
        tar_proxy = plt.scatter([], [], s=500, edgecolor='#8EB69C', facecolor='#8EB69C', marker='*')
        obs_proxy = plt.scatter([], [], s=500, edgecolor='k', linestyle='--', facecolor='none', linewidths=1.5)
        obs_start_proxy = plt.scatter([], [], s=300, edgecolor='k', facecolor='k', marker='s')
        plt.legend(
            handles=[start_proxy, tar_proxy, obs_proxy, obs_start_proxy], 
            labels=['起点', '目标点', '障碍物', '障碍物起点'], loc='upper left',
            prop=fangsong_font
        )

        plt.savefig('dynamic.png', format='png', dpi=300, bbox_inches='tight')

    def show_clf(self, name='dynamic_clf.png'):
        """show the changes in clf over time"""
        if self.terminal_time > 200:
            self.terminal_time = 200
        figure, ax = plt.subplots(figsize=(16, 9))
        figure.set_dpi(200)

        font_path = "/home/hjh/simfang.ttf"  
        label_font = fm.FontProperties(fname=font_path, size=35)
        times_font = {
            'family': 'Times New Roman',
            'weight': 'normal',
            'size': 40,
        }

        t = np.arange(0, (self.terminal_time) * self.step_time, self.step_time)[0:self.terminal_time]
        plt.plot(
            t, self.clf_t[0][0: self.terminal_time].reshape(self.terminal_time, ), 
            linewidth=6, color='#219EBC'
        )
       
        plt.xlabel("时间" + r'$(s)$', fontproperties=label_font)
        plt.ylabel("V(x)", fontproperties=times_font)
        plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

        # set the tick in Times New Roman and size
        ax.tick_params(labelsize=40)
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]

        plt.grid() 
        plt.savefig(name, format='png', dpi=300, bbox_inches='tight')
        # plt.show()

    def show_slack(self, name='dynamic_slack.png'):
        """ show the changes in slack over time """
        if self.terminal_time > 200:
            self.terminal_time = 200
        figure, ax = plt.subplots(figsize=(16, 9))
        figure.set_dpi(200)

        font_path = "/home/hjh/simfang.ttf"  
        label_font = fm.FontProperties(fname=font_path, size=35)

        t = np.arange(0, (self.terminal_time) * self.step_time, self.step_time)[0:self.terminal_time]
        plt.plot(
            t, self.slackt[0][0: self.terminal_time].reshape(self.terminal_time, ), 
            linewidth=6, color='#90C9E7'
        )

        plt.xlabel("时间" + r'$(s)$', fontproperties=label_font)
        plt.ylabel("δ", fontproperties=label_font)
        plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

        # set the tick in Times New Roman and size
        ax.tick_params(labelsize=40)
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]

        plt.grid() 
        plt.savefig(name, format='png', dpi=300, bbox_inches='tight')
        # plt.show()

    def show_cbf(self, name='dynamic_cbf.png'):
        """show the changes in cbf over time"""
        if self.terminal_time > 200:
            self.terminal_time = 200
        figure, ax = plt.subplots(figsize=(16, 9))
        figure.set_dpi(200)

        font_path = "/home/hjh/simfang.ttf"  
        legend_font = fm.FontProperties(fname=font_path, size=35)
        label_font = fm.FontProperties(fname=font_path, size=35)
        color_list = ['#219EBC', '#FEB705']

        t = np.arange(0, (self.terminal_time) * self.step_time, self.step_time)[0:self.terminal_time]
        cbf1, = plt.plot(
            t, self.cbf_t[0][0: self.terminal_time].reshape(self.terminal_time, ), 
            linewidth=6, color=color_list[0]
        )
        cbf2, = plt.plot(
            t, self.cbf_t[1][0: self.terminal_time].reshape(self.terminal_time, ), 
            linewidth=6, color=color_list[1]
        )

        plt.xlabel("时间" + r'$(s)$', fontproperties=label_font)
        plt.ylabel("距离" + r'$(m)$', fontproperties=label_font)
        plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        plt.legend(
            handles=[cbf1, cbf2], 
            labels=['圆形障碍物1', '圆形障碍物2'], 
            loc='upper left', prop=legend_font
        )

        # set the tick in Times New Roman and size
        ax.tick_params(labelsize=40)
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]

        plt.grid() 
        plt.savefig(name, format='png', dpi=300, bbox_inches='tight')

    def show_control(self, name='controls_dynamic.png'):
        """show the changes in control over time"""
        if self.terminal_time > 200:
            self.terminal_time = 200

        figure, ax1 = plt.subplots(figsize=(16, 9))
        figure.set_dpi(200)
        font_path = "/home/hjh/simfang.ttf"  
        legend_font = {"family": "Times New Roman", "weight": "normal", "size": 25}
        label_font = fm.FontProperties(fname=font_path, size=35)

        v_color = ['#EDDDC3', '#8EB69C', '#4EAB90']
        w_color = ['#EEBF6D', '#D94F33', '#834026']

        t = np.arange(0, (self.terminal_time) * self.step_time, self.step_time)[0:self.terminal_time]
        v = self.ut[0][0:self.terminal_time].reshape(self.terminal_time,)
        w = self.ut[1][0:self.terminal_time].reshape(self.terminal_time,)

        window_size = 5
        v_smooth = np.convolve(v, np.ones(window_size) / window_size, mode='valid')
        w_smooth = np.convolve(w, np.ones(window_size) / window_size, mode='valid')

        ax1.set_xlabel("时间" + r'$(s)$', fontproperties=label_font)
        ax1.set_ylabel("线速度" + r'$v (m / s)$', fontproperties=label_font)
        ax1.tick_params(axis='y')

        u_max = self.parameter['u_max']
        vv, = ax1.plot(t[:v_smooth.size], v_smooth, linewidth=6, color=v_color[0])
        v_min, = ax1.plot(t, -u_max[0] * np.ones(t.shape[0]), linewidth=6, color=v_color[1], linestyle="--")
        v_max, = ax1.plot(t, u_max[0] * np.ones(t.shape[0]), linewidth=6, color=v_color[2], linestyle="--")

        ax1.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax1.grid(True, linestyle='--', alpha=0.6)  # 优化网格

        ax2 = ax1.twinx()
        ax2.set_ylim(ax1.get_ylim())  
        ax2.set_ylabel("角速度" + r'$w (rad / s)$', fontproperties=label_font)
        ax2.tick_params(axis='y')
        ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

        ww, = ax2.plot(t[:w_smooth.size], w_smooth, linewidth=6, color=w_color[0])
        w_min, = ax2.plot(t, -u_max[1] * np.ones(t.shape[0]), linewidth=6, color=w_color[1], linestyle="--")
        w_max, = ax2.plot(t, u_max[1] * np.ones(t.shape[0]), linewidth=6, color=w_color[2], linestyle="--")

        lines = [vv, ww, v_min, v_max, w_min, w_max]
        labels = [r'$v$', r'$w$', r'$v_{min}$', r'$v_{max}$', r'$w_{min}$', r'$w_{max}$']
        ax1.legend(lines, labels, loc='lower center', prop=legend_font, framealpha=0.5, ncol=6, bbox_to_anchor=(0.5, 0.2))

        ax1.tick_params(labelsize=45)
        ax2.tick_params(labelsize=45)

        plt.savefig(name, format='png', dpi=300, bbox_inches='tight')

    def show_control_v(self):
        """show the changes in v over time"""
        # set the label in Times New Roman and size
        label_font = {
            'family': 'Times New Roman',
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
        label_font = {
            'family': 'Times New Roman',
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
        np.savez('static', x=self.xt, slack=self.slackt, obs_t=self.obstacle_list_t, cbf=self.cbf_t, clf=self.clf_t, u=self.ut)

    def load_static_fail_data(self):
        data = np.load('static_fail.npz')
        self.xt = data['x']
        self.obstacle_list_t = data['obs_t']

        # self.render()
        self.show_static_fail()

    def load_static_data(self):
        data = np.load('static.npz')
        self.xt = data['x']
        self.slackt = data['slack']
        self.cbf_t = data['cbf']
        self.clf_t = data['clf']
        self.obstacle_list_t = data['obs_t']
        self.ut = data['u']

        # self.render()
        self.show_clf(name='static_clf.png')
        self.show_slack(name='static_slack.png')
        self.show_cbf(name='static_cbf.png')
        self.show_control(name='controls_static.png')
        # self.show_static_obs(index_t=[15, 30])

    def load_dynamic_data(self):
        data = np.load('dynamic.npz')
        self.xt = data['x']
        self.slackt = data['slack']
        self.cbf_t = data['cbf']
        self.clf_t = data['clf']
        self.obstacle_list_t = data['obs_t']
        self.ut = data['u']

        # self.render()
        self.show_clf()
        self.show_slack()
        self.show_cbf()
        self.show_control()
        # self.show_dynamic_obs([16, 38])

if __name__ == '__main__':
    test_target = Point_Stabilization_Execute()
    # test_target.qp_solve_clf()
    # test_target.qp_solve_cbf_clf()
    
    # test_target.storage_data()
    # test_target.load_static_fail_data()
    # test_target.load_static_data()
    test_target.load_dynamic_data()
    # test_target.render()
    # test_target.show_control_v()
    # test_target.show_control_w()
    # test_target.show_clf()
    # test_target.show_slack()
    # # test_target.show_cbf()
    # test_target.show_control()
    
