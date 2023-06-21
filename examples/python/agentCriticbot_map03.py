#!/usr/bin/env python3

# M. Kempka, T.Sternal, M.Wydmuch, Z.Boztoprak
# January 2021

import numpy as np
import skimage.color
import skimage.transform
import tensorflow as tf
# from tensorflow.keras import Model, Sequential
# from tensorflow.keras.layers import BatchNormalization, Conv2D, Dense, Flatten, ReLU, Softmax
# from tensorflow.keras.optimizers import SGD, Adam



# # Q-learning settings
# learning_rate_actor = 0.000001
# learning_rate_critic = 0.000005

# gamma = 0.90
# replay_memory_size = 1000000
# num_train_epochs = 5
# batch_size = 96

# # Other parameters
frames_per_action = 3
resolution = (100, 75)
resolution_buffer = (100, 75*3)


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
 
def update_buffer(img, framebuffer):
    axis = -1
    framebuffer = np.concatenate([img, framebuffer], axis = axis)
    return framebuffer

class AgentCritic:
    def __init__(self, num_actions, epsilon=1, epsilon_min=0.1, epsilon_decay=0.9995, gamma = 0.9):
        self.num_actions = num_actions
        self.gamma = gamma    # discount rate
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.episode = 2


    def choose_action(self, state):
        state = np.array([state], dtype=np.float32)
        action_probability, _ = self.actor_critic.predict(state, verbose = 0)

        ids = np.arange(len(action_probability[0]))
        try:
            chosen_action = np.random.choice(ids, p = action_probability[0])
        except:
            print("nan in action_probability. Load last checkpoint")
            chosen_action = np.random.choice(ids)

        return chosen_action



