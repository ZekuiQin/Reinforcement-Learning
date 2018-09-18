import training as training
from models import Actor, Critic
from memory import Memory
from noise import *
from single_env import Pathplanningenv
import tensorflow as tf

def run():
    # Configure things.

    # Create envs.
    print("创建路径规划环境")
    env = Pathplanningenv()
    print("初始化状态")
    observaiton = env.reset()

    # Parse noise_type
    print("设置参数")
    action_noise = None
    param_noise = None
    nb_actions = 1
    max_episode = 10000
    render = False
    nb_train_steps = 50
    observation_shape = (2,)
    action_shape = (1,)

    # Configure components.
    print("创建memory")
    memory = Memory(limit=int(1e6), action_shape=(1,), observation_shape=(2,))
    print("创建critic")
    critic = Critic()
    print("创建actor")
    actor = Actor(nb_actions)

    # Seed everything to make things reproducible.
    tf.reset_default_graph()

    # Disable logging for rank != 0 to avoid noise.
    print("begin training")
    training.train(env, max_episode, render, nb_train_steps, actor, critic, memory, observation_shape, action_shape)
    #env.close()



if __name__ == '__main__':
    # Run actual script.
    print("开始运行")
    run()
