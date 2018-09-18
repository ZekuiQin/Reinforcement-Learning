import numpy as np


class RingBuffer(object):
    def __init__(self, maxlen, shape, dtype='float32'):  # 循环缓冲区，maxlen最大容量(行)，shape列数
        self.maxlen = maxlen                             # 可以容纳的最大的行数                        
        self.start = 0                                   # 起始点，一般为0，存满之后用它标记覆盖存储
        self.length = 0                                  # 非空数据量，初始化为0的不算数据
        self.data = np.zeros((maxlen,) + shape).astype(dtype)   # tuple一个元素的话加个逗号，区分整型，两个tuple相加扩展维数，astype指定数据类型
                                                                # shape必须也是tuple
    def __len__(self):  # 返回数据长度
        return self.length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:  # 取数索引[0,length-1]
            raise KeyError()
        return self.data[(self.start + idx) % self.maxlen]

    def get_batch(self, idxs):
        return self.data[(self.start + idxs) % self.maxlen]  # 取数，idxs小表示数据存的早，并非data下标小就早，start可能不为0

    def append(self, v):
        if self.length < self.maxlen:
            # We have space, simply increase the length.
            self.length += 1
        elif self.length == self.maxlen:
            # No space, "remove" the first item.
            self.start = (self.start + 1) % self.maxlen
        else:
            # This should never happen.
            raise RuntimeError()
        self.data[(self.start + self.length - 1) % self.maxlen] = v  # 加入新的数据


def array_min2d(x):
    x = np.array(x)  # 列表转换为数组，值传递，不会影响原来的列表
    if x.ndim >= 2:  # .ndim广义维数，2*2为2，2*3*4为3
        return x
    return x.reshape(-1, 1)  # 变为一列


class Memory(object):
    def __init__(self, limit, action_shape, observation_shape):  # limit为最大容量
        
        print("class Memory begins to create ")
        self.limit = limit
        self.observations0 = RingBuffer(limit, shape=observation_shape)  # observation_shape这里为(2,)
        self.actions = RingBuffer(limit, shape=action_shape)             # action_shape这里为(1,) 
        self.rewards = RingBuffer(limit, shape=(1,))
        self.terminals1 = RingBuffer(limit, shape=(1,))
        self.observations1 = RingBuffer(limit, shape=observation_shape)
        print("class Memory finishs creating")

    def sample(self, batch_size):                             
        # Draw such that we always have a proceeding element.
        batch_idxs = np.random.random_integers(self.nb_entries, size=batch_size)  # 在[1,前项]之间取整数，规模为size(如(2,3),),返回array
        
        '''                                                                           # size也可以是一个数字，返回一行数字列array,这里最好输入数字
        obs0_batch = self.observations0.get_batch(batch_idxs)  # 这里取数输入的batch_idxs是一个数组，可能有问题！！！
        obs1_batch = self.observations1.get_batch(batch_idxs)  # 用列表生成式，obs0_batch = [self..get_batch(x) for x in batch_idxs]
        action_batch = self.actions.get_batch(batch_idxs)
        reward_batch = self.rewards.get_batch(batch_idxs)
        terminal1_batch = self.terminals1.get_batch(batch_idxs)
        '''

        obs0_batch = [self.observations0.get_batch(x) for x in batch_idxs]
        obs1_batch = [self.observations1.get_batch(x) for x in batch_idxs]
        action_batch = [self.actions.get_batch(x) for x in batch_idxs]
        reward_batch = [self.rewards.get_batch(x) for x in batch_idxs]
        terminal1_batch = [self.terminals1.get_batch(x) for x in batch_idxs]


        result = {
            'obs0': array_min2d(obs0_batch),
            'obs1': array_min2d(obs1_batch),
            'rewards': array_min2d(reward_batch),
            'actions': array_min2d(action_batch),
            'terminals1': array_min2d(terminal1_batch),
        }
        return result

    def append(self, obs0, action, reward, obs1, terminal1, training=True):
        if not training:
            return
        
        self.observations0.append(obs0)
        self.actions.append(action)
        self.rewards.append(reward)
        self.observations1.append(obs1)
        self.terminals1.append(terminal1)

    @property
    def nb_entries(self):
        return len(self.observations0)  # 已经存的数据的个数
