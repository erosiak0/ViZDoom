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
episodes = 5
learning_rate_actor = 0.000001
learning_rate_critic = 0.000005

gamma = 0.90
replay_memory_size = 1000
num_train_epochs = 5
learning_steps_per_epoch = 200
target_net_update_steps = 100
batch_size = 32
# Training regime
test_episodes_per_epoch = 10

# Other parameters
frames_per_action = 1
resolution = (120, 90)
resolution_buffer = (120, 90*3)

episodes_to_watch = 20

save_model = True
load = False
skip_learning = False
watch = False
player_skip = frames_per_action
# Configuration file path
model_savefolder_actor = "F:/SIiUM3/ViZDoom/model_actor1_all/model1_2/actor"
model_savefolder_critic = "F:/SIiUM3/ViZDoom/model_actor1_all/model1_2/critic"

model_savefolder_actor_backup = "F:/SIiUM3/ViZDoom/model_actor1_all/model1_1/actor"
model_savefolder_critic_backup = "F:/SIiUM3/ViZDoom/model_actor1_all/model1_1/critic"
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
                Conv2D(8, kernel_size=6, strides=3, input_shape=(120, 270, 1)),
                BatchNormalization(),
                ReLU(),
            ]
        )

        self.conv2 = Sequential(
            [
                Conv2D(12, kernel_size=5, strides=2, input_shape=(39, 89, 8)),
                BatchNormalization(),
                ReLU(),
            ]
        )

        self.conv3 = Sequential(
            [
                Conv2D(16, kernel_size=3, strides=2, input_shape=(18, 43, 12)),
                BatchNormalization(),
                ReLU(),
            ]
        )

        self.conv4 = Sequential(
            [
                Conv2D(24, kernel_size=3, strides=2, input_shape=(8, 21, 16)),
                BatchNormalization(),
                ReLU(),
            ]
        )

        self.flatten = Flatten()

        self.advantage = Dense(num_actions, activation = 'softmax')

    def call(self, x):
        x1 = self.conv1(x)
        x1 = self.conv2(x1)
        x1 = self.conv3(x1)
        x1 = self.conv4(x1)
        x = self.flatten(x1)
        x = self.advantage(x)

        return x
    
class AgentCriticCritic(Model):
    def __init__(self, num_actions):
        super().__init__()
        self.conv1 = Sequential(
            [
                Conv2D(8, kernel_size=6, strides=4, input_shape=(120, 270, 1)),
                BatchNormalization(),
                ReLU(),
            ]
        )

        self.conv2 = Sequential(
            [
                Conv2D(12, kernel_size=5, strides=2, input_shape=(29, 67, 8)),
                BatchNormalization(),
                ReLU(),
            ]
        )

        self.conv3 = Sequential(
            [
                Conv2D(16, kernel_size=3, strides=2, input_shape=(13, 32, 12)),
                BatchNormalization(),
                ReLU(),
            ]
        )

        self.flatten = Flatten()

        self.state_value = Dense(1)

    def call(self, x):
        x1 = self.conv1(x)
        x1 = self.conv2(x1)
        x1 = self.conv3(x1)
    
        x = self.flatten(x1)
        x = self.state_value(x)

        return x
    
class AgentCritic:
    def __init__(self, num_actions, epsilon=1, epsilon_min=0.1, epsilon_decay=0.9995):
        self.num_actions = num_actions
        self.gamma = gamma    # discount rate
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
    
        if load:
            print("Loading model actor from: ", model_savefolder_actor, 
                  "\n Loading model critic from: ",model_savefolder_critic)
            self.actor = tf.keras.models.load_model(model_savefolder_actor,  compile=False)
            self.critic = tf.keras.models.load_model(model_savefolder_critic,  compile=False)

        else:
            self.actor = AgentCriticActor(self.num_actions)
            self.critic = AgentCriticCritic(self.num_actions)

        self.actor.compile(loss = 'mse', optimizer = Adam(learning_rate_actor),  metrics=['accuracy'])
        self.critic.compile(loss = 'mse', optimizer = Adam(learning_rate_critic), metrics=['accuracy'])



        # self.actor.optimizer = Adam(learning_rate =0.00001)
        # self.critic.optimizer = Adam(learning_rate =0.00005)

    def choose_action(self, state, variables = None):
        """
        Compute the action to take in the current state, basing on policy returned by the network.

        Note: To pick action according to the probability generated by the network
        """
        # state = tf.convert_to_tensor([state], dtype=tf.float32)
        state = np.array([state], dtype=np.float32)

        if self.epsilon < np.random.uniform(0, 1):
            action_probability = self.actor.predict(state, verbose = 0)
        
            # print(f"action_probability: {action_probability}")
            ids = np.arange(len(action_probability[0]))
            try:
                chosen_action = np.random.choice(ids, p = action_probability[0])
            except:
                print("nan in action_probability. Load last checkpoint")
                chosen_action = np.random.choice(ids)
                self.actor = tf.keras.models.load_model(model_savefolder_actor_backup,  compile=False)
                self.critic = tf.keras.models.load_model(model_savefolder_critic_backup,  compile=False)
                self.actor.compile(loss = 'mse', optimizer = Adam(learning_rate_actor),  metrics=['accuracy'])
                self.critic.compile(loss = 'mse', optimizer = Adam(learning_rate_critic), metrics=['accuracy'])



        else:
            ids = np.arange(self.num_actions)
            chosen_action = np.random.choice(ids)
        return chosen_action


    def train(self, state, actions, rewards, next_state, dones):

        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            tape1.watch(self.actor.trainable_variables)
            tape2.watch(self.critic.trainable_variables)
            state = tf.convert_to_tensor(state, dtype=tf.float32)
            next_state = tf.convert_to_tensor(next_state, dtype=tf.float32)
            actions = tf.convert_to_tensor(actions, dtype=tf.int32)
            actions = tf.expand_dims(actions, axis=1)
            probs = self.actor(state)
            value = self.critic(state)
            next_value = self.critic(next_state)


            t3 = self.gamma * next_value
            value_target = tf.expand_dims(rewards, axis=1) + t3 - value

            prob_actions = tf.gather_nd(probs, actions, batch_dims=1)

            log_prob = tf.math.log(prob_actions)
            loss_actor = -log_prob*value_target
            loss_critic = value_target**2

        loss_actor = tf.convert_to_tensor(loss_actor, dtype=tf.float32)
        loss_critic = tf.convert_to_tensor(loss_critic, dtype=tf.float32)
        gradient_actor = tape1.gradient(loss_actor, self.actor.trainable_variables)
        gradient_critic = tape2.gradient(loss_critic, self.critic.trainable_variables)

        self.actor.optimizer.apply_gradients(zip(gradient_actor, self.actor.trainable_variables))
        self.critic.optimizer.apply_gradients(zip(gradient_critic, self.critic.trainable_variables))

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

def update_buffer(img, framebuffer):
    axis = -1
    framebuffer = np.concatenate([img, framebuffer], axis = axis)
    return framebuffer

def get_reward(action_reward, game_variables, game_variables_next):
    reward = 0
    if game_variables_next[1] - game_variables[1] > 0:
        reward += 0.5
    elif game_variables_next[2] - game_variables[2] > 0:
        reward += 0.2
    elif game_variables_next[4] - game_variables[4] > 0:
        reward -= 0.5   
    elif game_variables_next[15] - game_variables[15] > 0:
        reward += 0.1
    elif game_variables_next[15] - game_variables[15] < 0:
        reward -= 0.5

    return reward + action_reward
    

def save_memory2file(filepath, framebuffer, action, reward, framebuffer_next, done):
    saved_memory = {
    "framebuffer":framebuffer.tolist(),
    "action":action, #.tolist(),
    "reward":reward,
    "framebuffer_next":framebuffer_next.tolist()
    }

    # Check if the file already exists
    if os.path.exists(filepath):
        # Load existing data from the file
        with open(filepath, "rb") as file:
            existing_data = pickle.load(file)

        # Append the new data to the existing data
        existing_data.append(saved_memory)

        # Write the updated data to the file
        with open(filepath, "wb") as file:
            pickle.dump(existing_data, file)

    else:
        # Create a new file and write the new data to it
        with open(filepath, "wb") as file:
            pickle.dump([saved_memory], file)

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
    # with open(filepath, "rb") as file:
    #     saved_memory = pickle.load(file)

    for memories in saved_memory:
        for memory in memories:
            framebuffer = memory["framebuffer"] 
            action = memory["action"] 
            reward = memory["reward"]
            framebuffer_next = memory["framebuffer_next"]

            if action != 0:
                replay_memory.append((np.array(framebuffer, dtype=np.float32), np.array(action, dtype=np.int32), 
                                    float(reward), np.array(framebuffer_next, dtype=np.float32), 0))

    return replay_memory

    
    

# def run(agent, game, actions, train_scores, framebuffer, framebuffer_next, replay_memory):
#     reward = 0
#     state = game.get_state()
#     screen_buf = preprocess(state.screen_buffer)
#     framebuffer = update_buffer(screen_buf, framebuffer[:,:-90])
#     action = agent.choose_action(framebuffer)
#     game_variables = state.game_variables
#     action_reward = game.make_action(actions[action], frames_per_action)
#     if not game.is_episode_finished():


#         next_state = game.get_state()
#         game_variables_next = next_state.game_variables
#         reward = get_reward(action_reward, game_variables, game_variables_next)

#         # if not done:
#         next_screen_buf = preprocess(next_state.screen_buffer)
#         framebuffer_next = update_buffer(next_screen_buf, framebuffer_next[:,:-90])
#         replay_memory.append((framebuffer, action, reward, framebuffer_next, 0))

#         # else:
#         #     next_screen_buf = tf.zeros(shape=screen_buf.shape)
#         #     framebuffer_next = update_buffer(next_screen_buf, framebuffer_next[:,:-90])


#         if len(replay_memory) >= batch_size:    
#             screen_buf, actions, rewards, next_screen_buf, dones = split_tuple(get_samples(replay_memory))

#             agent.train(screen_buf, actions, rewards, next_screen_buf, dones)

#     if game.is_player_dead():
#         train_scores.append(game.get_total_reward())

#     return train_scores, action, reward, framebuffer, framebuffer_next

