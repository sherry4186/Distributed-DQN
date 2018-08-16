import chainer.links as L
import chainer.functions as F
from chainer import Chain, optimizers, Variable, serializers, initializers
from collections import deque
import copy
import gym
import numpy as np
import socket
import sys
import matplotlib.pyplot as plt
import pickle
from time import sleep
import os


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


class DQNSlave(DQN):
    def __init__(self, n_st, n_act, seed=0):
        super(DQNSlave, self).__init__(n_st, n_act, seed)

    def load_model_share(self, inputfile):
        while 1:
            if os.path.exists(inputfile):
                sleep(5)
                serializers.load_npz(inputfile, self.model)
                break
        print('load_model_share !')

    def load_epsilon(self, share_epsilon_file):
        while 1:
            if os.path.exists(share_epsilon_file):
                try:
                    self.epsilon = pickle.load(open(share_epsilon_file, 'rb'))
                except Exception:
                    continue
                break
        print('load_epsilon !')

    def update_experience_and_model(self, share_memory_file, share_model_file, share_model_update_flag_file):
        if len(self.memory) >= self.train_freq:
            pickle.dump(self.memory, open(share_memory_file, 'wb'))
            while os.path.exists(share_memory_file):
                pass
            while not os.path.exists(share_model_update_flag_file):
                pass
            self.load_model_share(share_model_file)
            self.load_epsilon(share_epsilon_file)
            self.memory.clear()
            print('update_experience_and_model !')


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

    # new
    agent = DQNSlave(n_st, n_act, seed)
    share_folder = '//ALB0218/Users/xinzhu_ye/PythonScript/share/'
    slave_number = 3
    share_memory_folder = share_folder + 'memory/'
    share_memory_file = share_memory_folder + 'memory' + socket.gethostname() + '.txt'
    share_model_folder = share_folder + 'DQNmodel/'
    share_model_file = share_model_folder + 'DQNmodel.model'
    share_model_update_flag_file = share_model_folder + 'DQNmodel_update_flag.txt'
    end_flag_folder = share_folder + 'end_flag/'
    end_flag_file = end_flag_folder + 'end_flag' + socket.gethostname() + '.txt'
    share_epsilon_folder = share_folder + 'epsilon' + os.sep
    share_epsilon_file = share_epsilon_folder + 'epsilon.txt'
    # env.Monitor.start(view_path, video_callable=None, force=True, seed=seed)

    list_t = []
    list_loss = []
    for i_episode in range(1000):
        print("episode_num" + str(i_episode))
        observation = env.reset()
        for t in range(400):

            env.render()
            state = observation.astype(np.float32).reshape((1, n_st))
            act_i = agent.get_action(state)[0]
            action = action_list[act_i]
            observation, reward, ep_end, _ = env.step(action)
            state_dash = observation.astype(np.float32).reshape((1, n_st))

            reward_true = t / 200
            sleep(10)

            # new: 增加异常处理
            agent.stock_experience(state, act_i, reward_true, state_dash, ep_end)
            agent.update_experience_and_model(share_memory_file, share_model_file, share_model_update_flag_file)

            if ep_end:
                print('max t:', t)
                print('loss:', agent.loss)
                list_t.append(t)
                list_loss.append(agent.loss)
                break

    end_flag = True
    pickle.dump(end_flag, open(end_flag_file, 'wb'))
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(list_t)
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(list_loss)
    plt.show()
    # env.Monitor.close()

    agent.save_model('DQN.model')
