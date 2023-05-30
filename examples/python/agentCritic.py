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
from tensorflow.keras.layers import BatchNormalization, Conv2D, Dense, Flatten, ReLU, Softmax
from tensorflow.keras.optimizers import SGD, Adam
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

# Training regime
test_episodes_per_epoch = 10

# Other parameters
frames_per_action = 5
resolution = (120, 90)
episodes_to_watch = 20

save_model = True
load = False
skip_learning = False
watch = False
player_skip = frames_per_action
# Configuration file path
model_savefolder_actor = "F:/SIiUM3/ViZDoom/model4.1/actor"
model_savefolder_critic = "F:/SIiUM3/ViZDoom/model4.1/critic"


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
                Conv2D(24, kernel_size=3, strides=2, input_shape=(8, 6, 16)),
                BatchNormalization(),
                ReLU(),
            ]
        )
        # self.conv5 = Sequential(
        #     [
        #         Conv2D(32, kernel_size=3, strides=2, input_shape=(3, 6, 24)),
        #         BatchNormalization(),
        #         ReLU(),
        #     ]
        # )
        self.flatten = Flatten()

        # self.state_value = Dense(1)
        self.advantage = Dense(num_actions, activation = 'softmax')
        # self.softmax = Softmax()

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # x = self.conv5(x)        
        x = self.flatten(x)
        # x1 = x[:, :96]
        # x2 = x[:, 96:]
        # x1 = self.state_value(x1)
        x2 = self.advantage(x)

        # x = x1 + (x2 - tf.reshape(tf.math.reduce_mean(x2, axis=1), shape=(-1, 1)))
        # x = self.softmax(x)
        return x2
    
class AgentCriticCritic(Model):
    def __init__(self, num_actions):
        super().__init__()
        self.conv1 = Sequential(
            [
                Conv2D(8, kernel_size=6, strides=4, input_shape=(120, 90, 1)),
                BatchNormalization(),
                ReLU(),
            ]
        )

        self.conv2 = Sequential(
            [
                Conv2D(12, kernel_size=5, strides=2, input_shape=(29, 22, 8)),
                BatchNormalization(),
                ReLU(),
            ]
        )

        self.conv3 = Sequential(
            [
                Conv2D(16, kernel_size=3, strides=2, input_shape=(13, 9, 12)),
                BatchNormalization(),
                ReLU(),
            ]
        )

        self.flatten = Flatten()

        self.state_value = Dense(1)
        # self.advantage = Dense(num_actions)

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
    
        x = self.flatten(x)
        # x1 = x[:, :96]
        # x2 = x[:, 96:]
        x1 = self.state_value(x)
        # x2 = self.advantage(x)

        # x = x1 + (x2 - tf.reshape(tf.math.reduce_mean(x2, axis=1), shape=(-1, 1)))
        return x1
    
class AgentCritic:
    def __init__(self, num_actions):
        self.num_actions = num_actions
        self.gamma = 0.99    # discount rate
        self.learning_rate = 0.001

        if load:
            print("Loading model actor from: ", model_savefolder_actor, 
                  "\n Loading model critic from: ",model_savefolder_critic)
            self.actor = tf.keras.models.load_model(model_savefolder_actor,  compile=False)
            self.critic = tf.keras.models.load_model(model_savefolder_critic,  compile=False)
            self.actor.compile(loss = 'mse', metrics=['accuracy'])
            self.critic.compile(loss = 'mse', metrics=['accuracy'])

        else:
            self.actor = AgentCriticActor(self.num_actions)
            self.critic = AgentCriticCritic(self.num_actions)

        
        # self.actor = AgentCriticActor(self.num_actions)
        # self.critic = AgentCriticCritic(self.num_actions)
        
        # if load:
        #     print("Loading model actor from: ", model_savefolder_actor, 
        #           "\n Loading model critic from: ",model_savefolder_critic)

        #     self.actor.load_weights(model_savefolder_actor)
        #     self.critic.load_weights(model_savefolder_critic)

        self.actor.optimizer = Adam(learning_rate =0.0001)
        self.critic.optimizer = Adam(learning_rate =0.0005)

        # self.actor.compile(loss = 'mse', metrics=['accuracy'])
        # self.critic.compile(loss = 'mse',  metrics=['accuracy'])
    def choose_action(self, state):
        """
        Compute the action to take in the current state, basing on policy returned by the network.

        Note: To pick action according to the probability generated by the network
        """
        state = tf.convert_to_tensor([state], dtype=tf.float32)

        action_probability = self.actor.predict(state, verbose = 0)[0]
        ids = np.arange(len(action_probability))
        chosen_action = np.random.choice(ids, p = action_probability)
        return chosen_action

  

    def train(self, state, action, reward, next_state, done):
        
        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            state = tf.convert_to_tensor([state], dtype=tf.float32)
            next_state = tf.convert_to_tensor([next_state], dtype=tf.float32)

            probs = self.actor(state)[0]
            value = self.critic(state)[0]
            next_value = self.critic(next_state)[0]
            value_target = reward + (1 - np.array(done))* self.gamma * next_value - value

            log_prob = tf.math.log(probs[action])
            loss_actor = -log_prob*value_target
            loss_critic = value_target**2


        gradient_actor = tape1.gradient(loss_actor, self.actor.trainable_variables)
        gradient_critic = tape2.gradient(loss_critic, self.critic.trainable_variables)

        self.actor.optimizer.apply_gradients(zip(gradient_actor, self.actor.trainable_variables))
        self.critic.optimizer.apply_gradients(zip(gradient_critic, self.critic.trainable_variables))



def run(agent, game, actions, train_scores):
    state = game.get_state()
    screen_buf = preprocess(state.screen_buffer)
    action = agent.choose_action(screen_buf)
    reward = game.make_action(actions[action], frames_per_action)
    done = game.is_episode_finished()

    if not done:
        next_screen_buf = preprocess(game.get_state().screen_buffer)
    else:
        next_screen_buf = tf.zeros(shape=screen_buf.shape)
    
    agent.train(screen_buf, action, reward, next_screen_buf, done)
    if game.is_player_dead() or done:
        train_scores.append(game.get_total_reward())


    return train_scores, action, reward

