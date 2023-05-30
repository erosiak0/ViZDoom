#!/usr/bin/env python3

# M. Kempka, T.Sternal, M.Wydmuch, Z.Boztoprak
# January 2021

import itertools as it
import os
from collections import deque
from random import sample
from time import sleep, time

import numpy as np
import skimage.color
import skimage.transform
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, Dense, Flatten, ReLU
from tensorflow.keras.optimizers import SGD
from tqdm import trange

import vizdoom as vzd


tf.compat.v1.enable_eager_execution()
tf.executing_eagerly()

# Q-learning settings
episodes = 10
learning_rate = 0.00025
discount_factor = 0.99
replay_memory_size = 1000
num_train_epochs = 5
learning_steps_per_epoch = 200
target_net_update_steps = 100

# NN learning settings
batch_size = 16

# Training regime
test_episodes_per_epoch = 10

# Other parameters
frames_per_action = 5
resolution = (120, 90)
episodes_to_watch = 20
player_skip = frames_per_action
save_model = True
load = False
skip_learning = False
watch = False

# Configuration file path
model_savefolder = "F:/SIiUM3/ViZDoom/model4.1"

if len(tf.config.experimental.list_physical_devices("GPU")) > 0:
    print("GPU available")
    DEVICE = "/gpu:0"
else:
    print("No GPU available")
    DEVICE = "/cpu:0"


def preprocess(img):
    img = np.mean(img, axis = 0)
    # print(img.shape)
    # img = skimage.color.rgb2gray(img)
    img = skimage.transform.resize(img, resolution)
    img = img.astype(np.float32)
    # img = np.expand_dims(img, axis=-1)

    return tf.stack(img)



class DQN(Model):
    def __init__(self, num_actions):
        super().__init__()
        self.conv1 = Sequential(
            [
                Conv2D(8, kernel_size=6, strides=3, input_shape=(120, 90, 1)),
                BatchNormalization(),
                ReLU(),
            ]
        )

        self.conv2 = Sequential(
            [
                Conv2D(12, kernel_size=5, strides=2, input_shape=(39, 29, 8)),
                BatchNormalization(),
                ReLU(),
            ]
        )

        self.conv3 = Sequential(
            [
                Conv2D(16, kernel_size=3, strides=2, input_shape=(18, 13, 12)),
                BatchNormalization(),
                ReLU(),
            ]
        )

        self.conv4 = Sequential(
            [
                Conv2D(24, kernel_size=3, strides=1, input_shape=(8, 6, 16)),
                BatchNormalization(),
                ReLU(),
            ]
        )
        self.conv5 = Sequential(
            [
                Conv2D(32, kernel_size=3, strides=2, input_shape=(6, 4, 24)),
                BatchNormalization(),
                ReLU(),
            ]
        )
        self.flatten = Flatten()

        self.state_value = Dense(1)
        self.advantage = Dense(num_actions)

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)        
        x = self.flatten(x)
        x1 = x[:, :x.shape[1]//2]
        x2 = x[:, x.shape[1]//2:]
        x1 = self.state_value(x1)
        x2 = self.advantage(x2)

        x = x1 + (x2 - tf.reshape(tf.math.reduce_mean(x2, axis=1), shape=(-1, 1)))
        return x
    

class DQNAgent:
    def __init__(
        self, num_actions:int=8,
        epsilon:float=1,
        epsilon_min:float=0.1,
        epsilon_decay=0.9995,
        load=load
        ):

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.discount_factor = discount_factor
        self.num_actions = num_actions
        self.optimizer = SGD(learning_rate)

        if load:
            print("Loading model from: ", model_savefolder)
            self.dqn = tf.keras.models.load_model(model_savefolder, compile=False)
            self.dqn.compile(loss = 'mse', optimizer = self.optimizer, metrics=['accuracy'])
            self.target_net = tf.keras.models.load_model(model_savefolder, compile=False)
            self.target_net.compile(loss = 'mse', optimizer = self.optimizer, metrics=['accuracy'])

        else:
            self.dqn = DQN(self.num_actions)
            self.target_net = DQN(self.num_actions)

        # self.dqn = DQN(self.num_actions)
        # self.target_net = DQN(self.num_actions)
        
        # if load:
        #     print("Loading model from: ", model_savefolder)

        #     self.dqn.load_weights(model_savefolder)
        #     self.target_net.load_weights(model_savefolder)

    def update_target_net(self):
        self.target_net.set_weights(self.dqn.get_weights())

    def choose_action(self, state):
        if self.epsilon < np.random.uniform(0, 1):
            action = int(tf.argmax(self.dqn(tf.reshape(state, (1, 120, 90, 1))), axis=1))
        else:
            action = np.random.choice(range(self.num_actions), 1)[0]

        return action

    def train_dqn(self, samples):
        screen_buf, actions, rewards, next_screen_buf, dones = split_tuple(samples)

        row_ids = list(range(screen_buf.shape[0]))

        ids = extractDigits(row_ids, actions)
        done_ids = extractDigits(np.where(dones)[0])

        with tf.GradientTape() as tape:
            tape.watch(self.dqn.trainable_variables)
            Q_prev = tf.gather_nd(self.dqn(screen_buf), ids)

            Q_next = self.target_net(next_screen_buf)
            Q_next = tf.gather_nd(
                Q_next,
                extractDigits(row_ids, tf.argmax(self.dqn(next_screen_buf), axis=1)),
            )

            q_target = rewards + self.discount_factor * Q_next

            if len(done_ids) > 0:
                done_rewards = tf.gather_nd(rewards, done_ids)
                q_target = tf.tensor_scatter_nd_update(
                    tensor=q_target, indices=done_ids, updates=done_rewards
                )

            td_error = tf.keras.losses.MSE(q_target, Q_prev)

        gradients = tape.gradient(td_error, self.dqn.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.dqn.trainable_variables))

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min

def extractDigits(*argv):
    if len(argv) == 1:
        return list(map(lambda x: [x], argv[0]))

    return list(map(lambda x, y: [x, y], argv[0], argv[1]))

def split_tuple(samples):
    samples = np.array(samples, dtype=object)
    screen_buf = tf.stack(samples[:, 0])
    actions = samples[:, 1]
    rewards = tf.stack(samples[:, 2])
    next_screen_buf = tf.stack(samples[:, 3])
    dones = tf.stack(samples[:, 4])
    return screen_buf, actions, rewards, next_screen_buf, dones

def get_samples(memory):
    if len(memory) < batch_size:
        sample_size = len(memory)
    else:
        sample_size = batch_size

    return sample(memory, sample_size)

def run(agent, game, replay_memory, actions, train_scores):
    time_start = time()
    
    state = game.get_state()
    screen_buf = preprocess(state.screen_buffer)
    action = agent.choose_action(screen_buf)
    reward = game.make_action(actions[action], frames_per_action)
    done = game.is_episode_finished()

    if not done:
        next_screen_buf = preprocess(game.get_state().screen_buffer)
    else:
        next_screen_buf = tf.zeros(shape=screen_buf.shape)

    # if game.is_player_dead() or done:
    train_scores.append(game.get_total_reward())

    replay_memory.append((screen_buf, action, reward, next_screen_buf, done))

    return replay_memory, train_scores, action, reward

