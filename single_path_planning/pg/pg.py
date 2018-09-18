import math
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from single_env import Pathplanningenv 


class PolicyGradient(object):
    def __init__(self,learning_rate=0.01,reward_decay=0.95,output_graph=False):

        # 学习速率
        self.lr = learning_rate
        # 回报衰减率
        self.gamma = reward_decay
        # 一条轨迹的观测值，动作值，和回报值
        self.ep_obs, self.ep_as, self.ep_rs, self.ep_ps = [],[],[],[]
        # 创建策略网络
        self._build_net()
        # 启动一个默认的会话
        self.sess = tf.Session()
        if output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)
        # 初始化会话中的变量
        self.sess.run(tf.global_variables_initializer())


    # 创建策略网络的实现
    def _build_net(self):
        with tf.name_scope('input'):
            # 创建占位符作为输入
            self.tf_obs = tf.placeholder(tf.float32, [None, 2], name="observations")
            self.tf_acts = tf.placeholder(tf.float32, [None, 1], name="actions")
            self.tf_vt = tf.placeholder(tf.float32, [None, 1], name="actions_value")
            self.tf_logp = tf.placeholder(tf.float32, [1, None], name="probability")
        # 第一层
        layer = tf.layers.dense(
            inputs=self.tf_obs,
            units=20,
            activation=tf.nn.tanh,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc1')

        # 第二层
        layer1 = tf.layers.dense(
            inputs=layer,
            units=20,
            activation=tf.nn.tanh,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc2')
        # 第三层
        all_act = tf.layers.dense(
            inputs=layer1,
            units=1,
            activation=tf.nn.tanh,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc3')

        self.netout = all_act
        action = tf.truncated_normal(shape=[1,1],mean=all_act,stddev=1) # 采样距离中心偏差不超过两个标准差
        action = tf.mod(action,2*math.pi)  # 规范化到0～2*pi，采样的真实值
        a_mean = tf.mod(all_act,2*math.pi)  # 确定性策略规范后的中心值
        a_prob = 1/(tf.sqrt(2*math.pi)*1)*tf.exp((-(action-a_mean)**2)/(2*1**2))  # 该采样值的概率，服从高斯分布
        self.tf_logp = tf.log(a_prob)
        self.tf_logp = tf.transpose(self.tf_logp,perm=[1,0])  # 转置为行向量

        # 定义损失函数，很关键的难点，目标函数是标量还是向量，交叉商那块怎么变成了向量，片段多步更新
        with tf.name_scope('loss'):
            loss = -tf.reduce_mean(self.tf_logp*self.tf_vt)
        # 定义训练,更新参数
        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)
    # 定义如何选择行为，即状态ｓ处的行为采样.根据当前的行为概率分布进行采样
    def choose_action(self, observation):  # 训练时的采样规则，确定性策略+随机项

        # 确定性策略得到动作值之后叠加一些噪声，最后规范化到0～2*pi
        action_determinacy = self.sess.run(self.netout, feed_dict={self.tf_obs:np.array(observation)[np.newaxis,:]})
        # 高斯分布采样
        action = np.random.normal(action_determinacy,1) # 采样
        action = action%(2*math.pi)  # 规范化到0～2*pi，采样的真实值
        a_mean = action_determinacy%(2*math.pi)  # 确定性策略规范后的中心值
        a_prob = 1/(math.sqrt(2*math.pi)*1)*math.exp((-(action-a_mean)**2)/(2*1**2))  # 该采样值的概率，服从高斯分布
        a_prob = math.log(a_prob)

        return action,a_prob
    def greedy(self, observation):  # 测试时的采样规则，确定性策略
        action_determinacy = self.sess.run(self.netout, feed_dict={self.tf_obs: np.array(observation)[np.newaxis, :]})
        action = action_determinacy
        return action
    # 定义存储，将一个回合的状态，动作和回报都保存在一起
    def store_transition(self, s, a, r, p):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)
        self.ep_ps.append(p)
    # 学习，以便更新策略网络参数，一个episode之后学一回
    def learn(self):
        # 计算一个episode的折扣回报
        discounted_ep_rs_norm = self._discount_and_norm_rewards()
        # 调用训练函数更新参数
        self.sess.run(self.train_op, feed_dict={
            self.tf_obs: np.vstack(self.ep_obs),  # 垂直叠加
            self.tf_acts: np.vstack(self.ep_as),
            self.tf_vt: np.vstack(discounted_ep_rs_norm),
            self.tf_logp: np.array(self.ep_ps)[np.newaxis,:]
        })
        # 清空episode数据
        self.ep_obs, self.ep_as, self.ep_rs, self.ep_ps = [],[],[],[]
        return discounted_ep_rs_norm
    def _discount_and_norm_rewards(self):
        # 折扣回报和
        discounted_ep_rs =np.zeros_like(self.ep_rs)
        running_add = 0
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add
        # 归一化
        discounted_ep_rs-= np.mean(discounted_ep_rs)
        #discounted_ep_rs /= np.std(discounted_ep_rs)

        return discounted_ep_rs



def training(max_episode,RENDER,env,RL):
    #a,initial_state = np.array(env.reset())
    #print("initial position:",initial_state)
    # 学习过程
    for i_episode in range(max_episode):
        
        observation = env.reset() # 每次开始位置不一样
        #observation = initial_state  # 每次开始位置一样
        # plt.close()                  # 关掉上次绘图
        while True:
            if RENDER: env.render()
            # 采样动作，探索环境
            action,prob = RL.choose_action(observation)
    
            observation_next, reward, is_terminal, is_break, is_out = env.dynamic(action)
    
            # 将观测，动作和回报存储起来
            RL.store_transition(observation, action, reward, prob)
            if is_terminal:
                print("episode:",'{:<5}'.format(i_episode),'{:<15}'.format("result:arrived"), "rewards:",'{:.2f}'.format(RL.ep_rs[-1]))
                if i_episode==(max_episode-1):
                    print("max_episode!")
                # 每个episode学习一次
                vt = RL.learn()
                if i_episode == 0:
                    print(vt)
                    plt.figure()
                    plt.plot(vt)
                    plt.xlabel('episode steps')
                    plt.ylabel('normalized state-action value')
                    plt.show()
                break
            
            if is_break:
                print("episode:",'{:<5}'.format(i_episode),'{:<15}'.format("reselt:break"), "rewards:",'{:.2f}'.format(RL.ep_rs[-1]))
                if i_episode==(max_episode-1):
                    print("max_episode!")
    
                # 每个episode学习一次
                vt = RL.learn()
                break
    
            if is_out:
                print("episode:",'{:<5}'.format(i_episode),'{:<15}'.format("result:out"), "rewards:",'{:.2f}'.format(RL.ep_rs[-1]))
                if i_episode==(max_episode-1):
                    print("max_episode!")
    
                # 每个episode学习一次
                vt = RL.learn()
                break
    
    
            # 智能体探索一步
            observation = observation_next



def test(env,RL,max_test_episode):
    
    # #测试过程
    done = False
    print("----------------------------------------------------")
    print("test begin!")
    for i_episode in range(max_test_episode):
        observation = env.reset()
        if done:
            print("goal! perfect!! you are a genius!!!")
            print("steps:",count)
            
            #break
        count = 0
        while True:
            # 采样动作，探索环境
            env.render()
            action = RL.greedy(observation)
            observation_next, reward, is_terminal, is_break, is_out = env.dynamic(action)
            if is_terminal:
                print("episode:", i_episode,"result:arrived")
                done = True
    
                break
    
            if is_break:
                print("episode:", i_episode,"reselt:break")
    
                break
    
            if is_out:
                print("episode:", i_episode,"result:out")
    
                break
    
            observation = observation_next
            count+=1
            # time.sleep(0.001)


env = Pathplanningenv()
RL = PolicyGradient(learning_rate=0.02,reward_decay=0.99)
RENDER = False  # 是否显示环境
max_episode = 50
max_test_episode = 3
training(max_episode,RENDER,env,RL)
#saver = tf.train.Saver()
#with tf.Session() as sess:
#    saver.save(sess,"model/model.ckpt")
test(env,RL,max_test_episode)

