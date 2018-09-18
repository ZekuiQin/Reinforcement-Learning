import time
from ddpg import DDPG
import numpy as np
import tensorflow as tf


def train(env, max_episode, render, nb_train_steps, actor, critic, memory, observation_shape, action_shape, gamma=0.99,observation_range=(0,100),
        tau=0.001, normalize_returns=False, normalize_observations=False, batch_size=128, action_noise=None, param_noise=True,
        action_range=(-np.pi,np.pi),critic_l2_reg=0., actor_lr=1e-4, critic_lr=1e-3, popart=False, clip_norm=None, param_noise_adaption_interval=50):
    

    agent = DDPG(actor, critic, memory, observation_shape, action_shape,action_range=action_range,observation_range=observation_range,
        gamma=gamma, tau=tau, normalize_returns=normalize_returns, normalize_observations=normalize_observations,
        batch_size=batch_size, action_noise=action_noise, param_noise=param_noise, critic_l2_reg=critic_l2_reg,
        actor_lr=actor_lr, critic_lr=critic_lr, enable_popart=popart, clip_norm=clip_norm)

    saver = tf.train.Saver()
        
    with tf.Session() as sess:  # tf.Session()
        # Prepare everything.
        agent.initialize(sess)  # 初始化所有参数
        sess.graph.finalize()

        agent.reset()
        obs = env.reset()
        RENDER = False

        start_time = time.time()
        for i_episode in range(max_episode):
            step = 0
            while True:
                # Predict next action.
                step += 1
                if RENDER: env.render()

                action, q = agent.pi(obs, apply_noise=True, compute_Q=True)

                # Execute next action.
                if render:
                    env.render()
                new_obs, r, is_terminal, is_break, is_out = env.dynamic(action)  
                done = is_break or is_out
                agent.store_transition(obs, action, r, new_obs, done)
                obs = new_obs
                agent.train()
                
                if (step%20)==0:
                    agent.update_target_net()  # 每隔一定步数才更新目标网络，训练更稳定
                if done or is_terminal:
                    if (i_episode%10)==0:
                        if is_terminal:
                            print("episode:",'{:<5}'.format(i_episode),'{:<15}'.format("result:arrived"), "rewards:",'{:.2f}'.format(r))
                        if is_break:
                            print("episode:",'{:<5}'.format(i_episode),'{:<15}'.format("result:break"), "rewards:",'{:.2f}'.format(r))
                        if is_out:
                            print("episode:",'{:<5}'.format(i_episode),'{:<15}'.format("result:out"), "rewards:",'{:.2f}'.format(r))
                    agent.reset()
                    obs = env.reset()
                    break
