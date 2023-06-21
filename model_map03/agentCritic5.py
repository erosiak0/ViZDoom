#!/usr/bin/env python3

# M. Kempka, T.Sternal, M.Wydmuch, Z.Boztoprak
# January 2021

import itertools as it
import os
from collections import deque
from random import sample
from time import sleep, time
# import json
import pickle
import numpy as np
import skimage.color
import skimage.transform
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, Dense, Flatten, ReLU, Softmax
from tensorflow.keras.optimizers import SGD, Adam
from tqdm import trange

import vizdoom as vzd

import matplotlib.pyplot as plt
tf.compat.v1.enable_eager_execution()
tf.executing_eagerly()

# Q-learning settings
learning_rate_actor = 0.000001
learning_rate_critic = 0.000005

gamma = 0.90
replay_memory_size = 50000
num_train_epochs = 5
batch_size = 200

# Other parameters
frames_per_action = 3
resolution = (100, 75)
resolution_buffer = (100, 75*3)

save_model = True
load = True
player_skip = frames_per_action
# Configuration file path

actions = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0],
    [1, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 1, 0, 0],
    [1, 0, 0, 1, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 1, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 1, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 1, 0, 1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 1, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 1, 0, 0],
    [1, 0, 1, 1, 0, 0, 0, 0, 0],
    [1, 0, 1, 0, 1, 0, 0, 0, 0],
    [1, 0, 1, 0, 0, 1, 0, 0, 0],
    [1, 0, 1, 0, 0, 0, 1, 0, 0],
    [0, 1, 1, 1, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 1, 0, 0, 0, 0],
    [0, 1, 1, 0, 0, 1, 0, 0, 0],
    [0, 1, 1, 0, 0, 0, 1, 0, 0],
    [0, 0, 1, 1, 0, 1, 0, 0, 0],
    [0, 0, 1, 1, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 1, 1, 0, 0, 0],
    [0, 0, 1, 0, 1, 0, 1, 0, 0],    
]

if len(tf.config.experimental.list_physical_devices("GPU")) > 0:
    print("GPU available")
    DEVICE = "/gpu:0"
else:
    print("No GPU available")
    DEVICE = "/cpu:0"


def preprocess(img):
    img = np.mean(img, axis = 0)
    img = skimage.transform.resize(img, resolution)
    img = img.astype(np.float32)
    # img = np.expand_dims(img, axis=-1)

    return tf.stack(img)


class AgentCriticActor(Model):
    def __init__(self, num_actions):
        super().__init__()
        self.conv1 = Sequential(
            [
                Conv2D(8, kernel_size=3, strides=2, input_shape=(100, 225, 1)),
                BatchNormalization(),
                ReLU(),
            ]
        )

        self.conv2 = Sequential(
            [
                Conv2D(12, kernel_size=3, strides=1, input_shape=(49, 112, 8)),
                BatchNormalization(),
                ReLU(),
            ]
        )

        self.conv3 = Sequential(
            [
                Conv2D(20, kernel_size=3, strides=1, input_shape=(47, 110, 12)),
                BatchNormalization(),
                ReLU(),
            ]
        )

        self.conv4 = Sequential(
            [
                Conv2D(28, kernel_size=3, strides=2, input_shape=(45, 108, 20)),
                BatchNormalization(),
                ReLU(),
            ]
        )

        self.conv5 = Sequential(
            [
                Conv2D(36, kernel_size=3, strides=2, input_shape=(22, 53, 28)),
                BatchNormalization(),
                ReLU(),
            ]
        )

        self.flatten = Flatten()
        self.advantage = Dense(num_actions, activation = 'softmax')
        self.state_value = Dense(1)


    def call(self, x):
        x1 = self.conv1(x)
        x1 = self.conv2(x1)
        x1 = self.conv3(x1)
        x1 = self.conv4(x1)
        x1 = self.conv5(x1)
        x = self.flatten(x1)
        critic = self.state_value(x)

        actor  = self.advantage(x)

        return actor, critic
    
    
class AgentCritic:
    def __init__(self, num_actions, epsilon=1, epsilon_min=0.1, epsilon_decay=0.9995):
        self.num_actions = num_actions
        self.gamma = gamma    # discount rate
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.episode = 40

        self.model_savefolder = f"F:/SIiUM3/ViZDoom/model_actor5_a38_map03/model_backup/actor{self.episode}"

        self.model_savefolder_backup = f"F:/SIiUM3/ViZDoom/model_actor5_a38_map03/model_backup/actor{self.episode-1}"

        if load:
            print("Loading model actor from: ", self.model_savefolder)
            self.actor_critic = tf.keras.models.load_model("F:/SIiUM3/ViZDoom/model_actor5_a38_map03/model_backup/actor42",  compile=False)

        else:
            self.actor_critic = AgentCriticActor(self.num_actions)

        self.actor_critic.compile(loss = 'mse', optimizer = Adam(learning_rate_actor),  metrics=['accuracy'])

    def choose_action(self, state):
        state = np.array([state], dtype=np.float32)
        action_probability, _ = self.actor_critic.predict(state, verbose = 0)

        ids = np.arange(len(action_probability[0]))
        try:
            chosen_action = np.random.choice(ids, p = action_probability[0])
        except:
            print("nan in action_probability. Load last checkpoint")
            chosen_action = np.random.choice(ids)
            self.actor_critic = tf.keras.models.load_model(f"F:/SIiUM3/ViZDoom/model_actor5_a38_map03/model_backup/actor{self.episode-2}",  compile=False)
            self.actor_critic.compile(loss = 'mse', optimizer = Adam(learning_rate_actor),  metrics=['accuracy'])
            
            if self.episode > 2:
                self.episode -= 2

        return chosen_action


    def train(self, state, actions, rewards, next_state, dones):

        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            tape1.watch(self.actor_critic.trainable_variables)
            state = tf.convert_to_tensor(state, dtype=tf.float32)
            next_state = tf.convert_to_tensor(next_state, dtype=tf.float32)
            actions = tf.convert_to_tensor(actions, dtype=tf.int32)
            actions = tf.expand_dims(actions, axis=1)
            probs, value = self.actor_critic(state)

            _, next_value = self.actor_critic(next_state)

            t3 = self.gamma * next_value
            value_target = tf.expand_dims(rewards, axis=1) + t3 - value

            prob_actions = tf.gather_nd(probs, actions, batch_dims=1)

            log_prob = tf.math.log(prob_actions)
            loss_actor = -log_prob*value_target
            loss_critic = value_target**2
            loss = (loss_actor + loss_critic)/2


        loss = tf.convert_to_tensor(loss, dtype=tf.float32)
        gradient_actor_critic = tape1.gradient(loss, self.actor_critic.trainable_variables)
        self.actor_critic.optimizer.apply_gradients(zip(gradient_actor_critic, self.actor_critic.trainable_variables))


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

def update_buffer(img, framebuffer):
    axis = -1
    framebuffer = np.concatenate([img, framebuffer], axis = axis)
    return framebuffer


def get_reward(action_reward, game_variables, game_variables_next):
    reward = 0
    if game_variables_next[1] - game_variables[1] > 0: # hitcount change
        reward += 20.0

    if game_variables_next[1] > 15:  #hitcount number
        reward += 5.0


    # if game_variables_next[1] - game_variables[1] == 0 and \   #shot without hit
    #     game_variables_next[14] - game_variables[14] < 0 or \
    #     game_variables_next[13] - game_variables[13] < 0 or \
    #     game_variables_next[12] - game_variables[12] < 0 or \
    #     game_variables_next[11] - game_variables[11] < 0 or \
    #     game_variables_next[10] - game_variables[10] < 0 or \
    #     game_variables_next[9] - game_variables[9] < 0 :
    #     reward -= 10.0

#     if game_variables_next[14] - game_variables[14] > 0: # collect ammo
#         reward += 1.0
#     if game_variables_next[13] - game_variables[13] > 0:
#         reward += 1.0
#     if game_variables_next[12] - game_variables[12] > 0:
#         reward += 1.0
#     if game_variables_next[11] - game_variables[11] > 0:
#         reward += 1.0
#     if game_variables_next[10] - game_variables[10] > 0:
#         reward += 1.0
#     if game_variables_next[9] - game_variables[9] < 0 :
#         reward += 1.0

    if game_variables_next[2] - game_variables[2] > 0:  # killcount
        reward += 100.0
    if game_variables_next[3] - game_variables[3] > 0: # hits_taken
        reward -= 1.0
    if game_variables_next[4] - game_variables[4] > 0:  # dead
        reward -= 5.0   
    if game_variables_next[15] - game_variables[15] > 0:  #fragcount
        reward += 500.0
    if game_variables_next[15] - game_variables[15] < 0: # self murder
        reward -= 100.0
    return reward + action_reward


def loadall(filename):
    with open(filename, "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break

def load_pickle(filepath): 
    replay_memory = deque(maxlen=replay_memory_size)
    saved_memory = loadall(filepath)

    for memories in saved_memory:
        for memory in memories:
            framebuffer = memory["framebuffer"] 
            action = memory["action"] 
            reward = memory["reward"]
            framebuffer_next = memory["framebuffer_next"]

            replay_memory.append((np.array(framebuffer, dtype=np.float32), np.array(action, dtype=np.int32), 
                                float(reward), np.array(framebuffer_next, dtype=np.float32), 0))
    print(f"len(replay_memory): {len(replay_memory)}")
    return replay_memory