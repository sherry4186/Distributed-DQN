import chainer.links as L
import chainer.functions as F
from chainer import Chain, optimizers, Variable, serializers, initializers
from collections import deque
import copy
import gym
import numpy as np
import sys
import pickle
import os
import glob
from time import sleep
import timeit


class NN(Chain):
    def __init__(self, n_in, n_out):
        super(NN, self).__init__(
            L1=L.Linear(n_in, 100),
            L2=L.Linear(100, 100),
            L3=L.Linear(100, 100),
            Q_value=L.Linear(100, n_out, initialW=initializers.Normal(scale=0.05))
        )

    def Q_func(self, x):
        h1 = F.leaky_relu(self.L1(x))
        h2 = F.leaky_relu(self.L2(h1))
        h3 = F.leaky_relu(self.L3(h2))
        return F.identity(self.Q_value(h3))


class DQN(object):
    def __init__(self, n_st, n_act, seed=0):
        super(DQN, self).__init__()
        np.random.seed(seed)
        sys.setrecursionlimit(10000)
        self.n_st = n_st
        self.n_act = n_act
        self.model = NN(n_st, n_act)
        self.target_model = copy.deepcopy(self.model)
        self.optimizer = optimizers.Adam()
        self.optimizer.setup(self.model)
        self.memory = deque()
        self.loss = 0
        self.step = 0
        self.gamma = 0.99
        self.memory_size = 10000
        self.batch_size = 100
        self.epsilon = 1
        self.epsilon_decay = 0.001
        self.epsilon_min = 0
        self.exploration = 1000
        self.train_freq = 10
        self.target_update_freq = 30

    def stock_experience(self, st, act, r, st_dash, ep_end):
        self.memory.append((st, act, r, st_dash, ep_end))
        if len(self.memory) > self.memory_size:
            self.memory.popleft()

    def forward(self, st, act, r, st_dash, ep_end):
        s = Variable(st)
        s_dash = Variable(st_dash)
        Q = self.model.Q_func(s)
        Q_dash = self.target_model.Q_func(s_dash)
        max_Q_dash = np.asanyarray(list(map(np.max, Q_dash.data)))
        target = np.asanyarray(copy.deepcopy(Q.data), dtype=np.float32)
        for i in range(self.batch_size):
            target[i, act[i]] = r[i] + (self.gamma * max_Q_dash[i]) * (not ep_end[i])
        loss = F.mean_squared_error(Q, Variable(target))
        self.loss = loss.data
        return loss

    def shuffle_memory(self):
        mem = np.array(self.memory)
        return np.random.permutation(mem)

    def parse_batch(self, batch):
        st, act, r, st_dash, ep_end = [], [], [], [], []
        for i in range(self.batch_size):
            st.append(batch[i][0])
            act.append(batch[i][1])
            r.append(batch[i][2])
            st_dash.append(batch[i][3])
            ep_end.append(batch[i][4])
        st = np.array(st, dtype=np.float32)
        act = np.array(act, dtype=np.int8)
        r = np.array(r, dtype=np.float32)
        st_dash = np.array(st_dash, dtype=np.float32)
        ep_end = np.array(ep_end, dtype=np.bool)
        return st, act, r, st_dash, ep_end

    def experience_replay(self):
        mem = self.shuffle_memory()
        perm = np.array(range(len(mem)))
        index = perm[0:self.batch_size]
        batch = mem[index]
        st, act, r, st_dash, ep_end = self.parse_batch(batch)
        self.model.cleargrads()
        loss = self.forward(st, act, r, st_dash, ep_end)
        loss.backward()
        self.optimizer.update()

    def get_action(self, st):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.n_act), 0
        else:
            s = Variable(st)
            Q = self.model.Q_func(s)
            Q = Q.data[0]
            a = np.argmax(Q)
            return np.asarray(a, dtype=np.int8), max(Q)

    def reduce_epsilon(self):
        if self.epsilon > self.epsilon_min and self.exploration < self.step:
            self.epsilon -= self.epsilon_decay

    def train(self):
        if len(self.memory) >= self.memory_size:
            if self.step % self.train_freq == 0:
                self.experience_replay()
                self.reduce_epsilon()
            if self.step % self.target_update_freq == 0:
                self.target_model = copy.deepcopy(self.model)
        self.step += 1

    def save_model(self, outputfile):
        serializers.save_npz(outputfile, self.model)

    def load_model(self, inputfile):
        serializers.load_npz(inputfile, self.model)


class DQNMaster(DQN):
    def __init__(self, n_st, n_act, seed=0):
        super(DQNMaster, self).__init__(n_st, n_act, seed=0)

    def load_experience(self, files, share_model_update_flag_file):
        if os.path.exists(share_model_update_flag_file):
            os.remove(share_model_update_flag_file)
        for file in files:
            memory = pickle.load(open(file, 'rb'))
            self.memory.extend(memory)
            if len(self.memory) > 500:
                print('beyond 500 ! time:', timeit.default_timer())
            while len(self.memory) > self.memory_size:
                self.memory.popleft()
            os.remove(file)
        print('load_experience !')

    def save_model_share(self, share_model_file, share_model_file_bk, share_model_update_flag_file):
        if os.path.exists(share_model_file):
            if os.path.exists(share_model_file_bk):
                os.remove(share_model_file_bk)
            os.rename(share_model_file, share_model_file_bk)
        serializers.save_npz(share_model_file, self.model)
        sleep(2)
        update_flag = True
        pickle.dump(update_flag, open(share_model_update_flag_file, 'wb'))
        print('save_model_share !')

    def save_epsilon(self, share_epsilon_file):
        pickle.dump(self.epsilon, open(share_epsilon_file, 'wb'))
        print('save_epsilon !')


if __name__ == "__main__":
    env_name = "CartPole-v0"
    seed = 0
    env = gym.make(env_name)
    view_path = 'video/' + env_name

    n_st = env.observation_space.shape[0]

    if type(env.action_space) == gym.spaces.discrete.Discrete:
        # CartPole-v0, Acrobot-v0, MountainCar-v0
        n_act = env.action_space.n
        action_list = np.arange(0, n_act)
    elif type(env.action_space) == gym.spaces.box.Box:
        # Pendulum-v0
        action_list = [np.array([a]) for a in [-2.0, 2.0]]
        n_act = len(action_list)

    print('time:', timeit.default_timer())

    # new
    agent = DQNMaster(n_st, n_act, seed)
    share_folder = '//ALB0218/Users/xinzhu_ye/PythonScript/share/'
    slave_number = 3
    share_memory_folder = share_folder + 'memory' + os.sep
    share_model_folder = share_folder + 'DQNmodel' + os.sep
    share_model_file = share_model_folder + 'DQNmodel.model'
    share_model_file_bk = share_model_folder + 'DQNmodel.model.bk'
    share_model_update_flag_file = share_model_folder + 'DQNmodel_update_flag.txt'
    end_flag_folder = share_folder + 'end_flag' + os.sep
    share_epsilon_folder = share_folder + 'epsilon' + os.sep
    share_epsilon_file = share_epsilon_folder + 'epsilon.txt'

    share_memory_files = glob.glob(share_memory_folder + '*')
    for file in share_memory_files:
        os.remove(file)

    if os.path.exists(share_model_update_flag_file):
        os.remove(share_model_update_flag_file)

    end_flag_files = glob.glob(end_flag_folder + '*')
    for file in end_flag_files:
        os.remove(file)

    if os.path.exists(share_epsilon_file):
        os.remove(share_epsilon_file)

    end_flag_count = len(glob.glob(end_flag_folder + '*'))
    # while not (end_flag_file1 in end_flag_files and end_flag_file2 in end_flag_files):
    while end_flag_count < slave_number:
        share_memory_files = glob.glob(share_memory_folder + '*')
        sleep(5)
        if len(share_memory_files) > 0:
            agent.load_experience(share_memory_files, share_model_update_flag_file)
            agent.train()
            agent.save_model_share(share_model_file, share_model_file_bk, share_model_update_flag_file)
            agent.save_epsilon(share_epsilon_file)
        end_flag_count = len(glob.glob(end_flag_folder + '*'))

    print('yes!')
