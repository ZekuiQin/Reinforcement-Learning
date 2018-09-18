"""
Path planning environment implemented by Zekui Qin
version 1
2018.07.30
"""
import math
import random
import numpy as np
import matplotlib.pyplot as plt

class Pathplanningenv(object):

    def __init__(self):
        self.step = 1    # 扩展步长
        self.width = 100   # 环境规模，宽度
        self.height = 100  # 环境规模，长度
        self.goal_distance_threshold = 10  # 目标范围阈值，认为距离目标点该距离以内就到达了目标点
        self.obstacle_distance_threshold = 0.5  # 距离阈值，判定是否安全是否到达目标点
        self.State = ['x','y']  # 状态为二维的连续空间
        self.Action = 'theta'   # 动作为朝向的角度，连续值，步长固定
        self.goal = [50,50]     # 目标点位置
        self.obstacle_num = 10  # 障碍物数目
        self.obstacle_pos = np.zeros((self.obstacle_num,2))     # 初始化障碍物位置
        self.obstacle_radius = np.zeros((self.obstacle_num,1))  # 初始化障碍物半径
        self.obstacle_radius_min = 1    # 障碍物最小半径
        self.obstacle_radius_max = 5    # 障碍物最大半径
        for i in range(self.obstacle_num):
            a = [random.randint(0,self.width),random.randint(0,self.height)]        # 随机初始化障碍物位置
            b = random.randint(self.obstacle_radius_min,self.obstacle_radius_max)   # 随机初始化障碍物半径
            dis = math.sqrt((a[0]-self.goal[0])**2+(a[1]-self.goal[1])**2)          # 障碍物到目标点的距离 
            while dis<=(self.obstacle_radius[i]+self.goal_distance_threshold):
                a = [random.randint(0,self.width),random.randint(0,self.height)]        # 保证目标点和障碍物不会靠的太近
                b = random.randint(self.obstacle_radius_min,self.obstacle_radius_max)
                dis = math.sqrt((a[0]-self.goal[0])**2+(a[1]-self.goal[1])**2)       
            self.obstacle_pos[i,:] = a
            self.obstacle_radius[i] = b

    def dynamic(self, action):   # initial_pos = reset(self),即为初始位置
        state = self.state
        next_state = [state[0]+self.step*math.cos(action),state[1]+self.step*math.sin(action)]   # 状态转移
        self.state = next_state
        is_terminal = False    # 是否到达目标点
        is_break = False       # 是否碰撞
        is_out = False         # 是否出了边界
        reward = 0             # 初始化奖励
        distance_goal = math.sqrt((next_state[0]-self.goal[0])**2+(next_state[1]-self.goal[1])**2)       # 转移后的状态到目标点的距离
        initial_pos = self.initial_state
        distance_initial = math.sqrt((initial_pos[0]-self.goal[0])**2+(initial_pos[1]-self.goal[1])**2)  # 起点到目标点的距离
        if distance_goal<self.goal_distance_threshold:   # 如果转移后的状态到目标点的距离小于阈值距离
            is_terminal = True                      # 判定agent到达目标点
            reward = reward+100                     # 大大增加奖励
        reward = reward+(1-distance_goal/distance_initial)*50   # 普通目标奖励，引导agent向目标移动
        distance_obs = np.zeros((self.obstacle_num,1))          # 初始化到障碍物的距离
        min_disobs = 10000                                      # 初始化到障碍物的最小距离
        for i in range(self.obstacle_num):
            distance_obs[i] = math.sqrt((next_state[0]-self.obstacle_pos[i,0])**2+(next_state[1]-self.obstacle_pos[i,1])**2)-self.obstacle_radius[i] 
            if distance_obs[i]<min_disobs:
                min_disobs = distance_obs[i]
        if min_disobs<=0:                           # 判定是否碰撞
            is_break = True
            reward = reward-50                     # 大大减小奖励
        elif min_disobs<=self.obstacle_distance_threshold:
            reward = reward+(1-min_disobs/self.obstacle_distance_threshold)*(-20)  # 普通障碍惩罚，驱使agent远离障碍物
        if (next_state[0]<0)|(next_state[0]>self.width)|(next_state[1]<0)|(next_state[1]>self.height):
            is_out = True
            reward = reward-50                     # 出边界奖励大大减小

        return next_state,reward,is_terminal,is_break,is_out



    def reset(self):
        self.state = [random.uniform(0,self.width),random.uniform(0,self.height)]
        is_tooclose = False
        for i in range(self.obstacle_num):
            dis = math.sqrt((self.state[0]-self.goal[0])**2+(self.state[1]-self.goal[1])**2)          # agent到障碍物的距离 
            if dis<=(self.obstacle_radius[i]+self.obstacle_distance_threshold):
                is_tooclose = True
        while is_tooclose:
            self.state = [random.uniform(0,self.width),random.uniform(0,self.height)]
            is_tooclose = False
            for i in range(self.obstacle_num):
                dis = math.sqrt((self.state[0]-self.goal[0])**2+(self.state[1]-self.goal[1])**2)      # 初始化agent位置不能离障碍物太近
                if dis<=(self.obstacle_radius[i]+self.obstacle_distance_threshold):
                    is_tooclose = True
        self.initial_state = self.state
        #initial_state = np.array(self.state)

        plt.close()

        #return self.state,initial_state
        return self.state

    def render(self):
        plt.cla()   # 清除上次绘制图像
        plt.plot(self.goal[0],self.goal[1],'go')    # 目标点
        plt.plot(self.state[0],self.state[1],'bo')  # agent位置
        plt.axis("equal")
        for i in range(self.obstacle_num):
            theta = np.arange(0,2*math.pi,0.01)
            x = self.obstacle_pos[i,0]+self.obstacle_radius[i]*np.cos(theta)
            y = self.obstacle_pos[i,1]+self.obstacle_radius[i]*np.sin(theta)
            plt.plot(x,y,'r')                                                 # 画障碍物圆
        plt.pause(0.01)
