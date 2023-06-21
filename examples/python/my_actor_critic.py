#!/usr/bin/env python3

# M. Kempka, T.Sternal, M.Wydmuch, Z.Boztoprak
# January 2021

import itertools as it
import os
from collections import deque
from random import sample
import pickle
import numpy as np
import skimage.color
import skimage.transform
import tensorflow as tf
from tqdm import trange
import skimage.color
import skimage.transform
from tensorflow.keras.models import Sequential, Model, load_model, model_from_json
from tensorflow.keras.layers import Dense, Dropout, Input, Activation, Conv2D, Conv1D, MaxPooling2D, Flatten
import absl.logging
import vizdoom as vzd

absl.logging.set_verbosity(absl.logging.ERROR)
tf.compat.v1.enable_eager_execution()
tf.executing_eagerly()

replay_memory_size = 6000
num_train_epochs = 40
learning_steps_per_epoch = 500

# NN learning settings
batch_size = 32

# Other parameters
frames_per_action = 12
resolution = (100, 100, 3)
n_images_to_stack = 3
resolution_for_buffer = (resolution[0], resolution[1] * n_images_to_stack, resolution[2])
episodes_to_watch = 20

save_model = True
load = False
# skip_learning = False
watch = True
window_visible = False
FILEPATH_FOR_SCORE = "F:\SIiUM3\ViZDoom\JK_map03\scores.pickle"

# Configuration file path
# config_file_path = os.path.join(vzd.scenarios_path, "multi.cfg")
model_savefolder = "F:/SIiUM3/ViZDoom/JK_map03/JK_map03"

DEVICE = "/gpu:0"
def yield_pickle(filename):
    with open(filename, "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break


def get_memory_with_gameplay(filepath):
    replay_memory = deque(maxlen=replay_memory_size)
    saved_memory = yield_pickle(filepath)

    for memories in saved_memory:
        for memory in memories:
            framebuffer = memory["framebuffer"]
            action = memory["action"]
            reward = memory["reward"]
            next_framebuffer = memory["next_framebuffer"]

            # if action != 0:
            replay_memory.append((np.array(framebuffer, dtype=np.float32), np.array(action, dtype=np.int32),
                                  reward, np.array(next_framebuffer, dtype=np.float32), 0))

    print(f'Length of replay_memory is: {len(replay_memory)}')
    return replay_memory


def preprocess(img):
    img = skimage.transform.resize(img, resolution)
    img = img.astype(np.float32)
    return img


def get_reward(state_variables, next_state_variables):
    reward = 0

    if next_state_variables[0] - state_variables[0] > 0:
        reward += 1

    if next_state_variables[0] - state_variables[0] < 0:
        reward += -1

    if next_state_variables[1] - state_variables[1] > 0:
        reward += 5

    if next_state_variables[2] - state_variables[2] > 0:
        reward += 100

    if next_state_variables[3] - state_variables[3] > 0:
        reward += -1

    if next_state_variables[4] or state_variables[4]:
        reward += -100

    if next_state_variables[5] - state_variables[5] > 0:
        reward += 1

    if next_state_variables[5] - state_variables[5] < 0:
        reward += -1

    if next_state_variables[9] - state_variables[9] > 0:
        reward += 5

    if next_state_variables[10] - state_variables[10] > 0:
        reward += 5

    if next_state_variables[11] - state_variables[11] > 0:
        reward += 5

    if next_state_variables[15] - state_variables[15] > 0:
        reward += 100

    return reward


def initialize_game():
    print("Initializing doom...")
    game = vzd.DoomGame()
    game.add_game_args(
        "-host 1 "
        # This machine will function as a host for a multiplayer game with this many players (including this machine).
        # It will wait for other machines to connect using the -join parameter and then start the game when everyone is connected.
        # "-port 5029 "  # Specifies the port (default is 5029).
        # "+viz_connect_timeout 60 "  # Specifies the time (in seconds), that the host will wait for other players (default is 60).
        "-deathmatch "  # Deathmatch rules are used for the game.
        "+timelimit 20.0 "  # The game (episode) will end after this many minutes have elapsed.
        "+sv_forcerespawn 1 "  # Players will respawn automatically after they die.
        "+sv_noautoaim 1 "  # Autoaim is disabled for all players.
        "+sv_respawnprotect 1 "  # Players will be invulnerable for two second after spawning.
        "+sv_spawnfarthest 1 "  # Players will be spawned as far as possible from any other players.
        "+sv_nocrouch 1 "  # Disables crouching.
        "+viz_respawn_delay 0 "  # Sets delay between respawns (in seconds, default is 0).
        "+viz_nocheat 1"
    )
    game.load_config("F:\SIiUM3\ViZDoom\scenarios\multi.cfg")
    game.set_window_visible(window_visible)
    game.set_mode(vzd.Mode.PLAYER)
    game.set_doom_map('map03')
    # game.set_screen_format(vzd.ScreenFormat.GRAY8)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.init()
    print("Doom initialized.")

    return game


class ResNetAgent:
    def __init__(
            self, num_actions=8, epsilon=1, epsilon_min=0.1, epsilon_decay=0.9995, load=load
    ):
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.num_actions = num_actions
        self.gamma = 0.95
        self.alpha_learning_rate = 0.000001
        self.beta_learning_rate = 0.000005

        if load:
            print("Loading model from: ", model_savefolder)
            self.actor = tf.keras.models.load_model(model_savefolder + '_actor')
            self.critic = tf.keras.models.load_model(model_savefolder + '_critic')
        else:
            self.actor = MyResNet(self.num_actions)
            self.critic = CriticNet()
            self.actor.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.alpha_learning_rate))
            self.critic.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.beta_learning_rate))

    def choose_action(self, state):
        if self.epsilon < np.random.uniform(0, 1):
            action = int(tf.argmax(self.actor(tf.reshape(state, resolution_for_buffer)), axis=1))
        else:
            action = np.random.choice(range(self.num_actions), 1)[0]

        return action

    def train(self, samples):
        screen_buf, actions, rewards, next_screen_buf, _ = split_tuple(samples)

        with tf.GradientTape(persistent=True) as tape:
            state_tensors = tf.convert_to_tensor(screen_buf, dtype=tf.float32)
            action_tensors = tf.convert_to_tensor(actions[:, np.newaxis], dtype=tf.int32)#             tf.expand_dims(tf.convert_to_tensor(actions, dtype=tf.int32), axis=1)
            rewards_tensors = tf.convert_to_tensor(tf.cast(rewards[:, np.newaxis], tf.float32), dtype=tf.float32)
            next_state_tensors = tf.convert_to_tensor(next_screen_buf, dtype=tf.float32)

            actor_state_prediction = self.actor(state_tensors, training=True)
            critic_state_prediction = self.critic(state_tensors, training=True)
            critic_next_state_prediction = self.critic(next_state_tensors, training=True)
            critic_loss = rewards_tensors + self.gamma * critic_next_state_prediction - critic_state_prediction
            indexed_actions = tf.gather_nd(actor_state_prediction, action_tensors, batch_dims=1)
            log_prob = tf.math.log(indexed_actions)
            actor_loss = -log_prob * critic_loss
            critic_loss = critic_loss ** 2

        grads_actor = tape.gradient(actor_loss, self.actor.trainable_variables)
        grads_critic = tape.gradient(critic_loss, self.critic.trainable_variables)

        self.actor.optimizer.apply_gradients(zip(grads_actor, self.actor.trainable_variables))
        self.critic.optimizer.apply_gradients(zip(grads_critic, self.critic.trainable_variables))

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min


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


def run(agent, game, replay_memory):

    framebuffer = np.zeros(resolution_for_buffer, 'float32')
    next_framebuffer = np.zeros(resolution_for_buffer, 'float32')

    for episode in range(num_train_epochs):
        # train_rewards = []
        print("\nEpoch %d\n-------" % (episode + 1))

        game.new_episode()
        game.send_game_command("removebots")
        for i in range(10):
            game.send_game_command("addbot")

        game.send_game_command("pukename change_difficulty 2")

        total_reward = 0
        total_kills = 0
        total_frags = 0
        total_hit_count = 0
        for i in trange(learning_steps_per_epoch, leave=False):
            state = game.get_state()
            game_variables = state.game_variables
            screen_buf = preprocess(state.screen_buffer)
            framebuffer = np.concatenate([screen_buf, framebuffer[:, :-resolution[1]]], axis=1)
            action = agent.choose_action(framebuffer)
            _ = game.make_action(actions[action], frames_per_action)
            next_state = game.get_state()
            game_variables_next = next_state.game_variables
            reward = get_reward(game_variables, game_variables_next)

            total_reward += reward
            total_hit_count += game_variables[1]
            total_kills += game_variables[2]
            total_frags += game_variables[-1]

            next_screen_buf = preprocess(game.get_state().screen_buffer)
            next_framebuffer = np.concatenate([next_screen_buf, next_framebuffer[:, :-resolution[1]]], axis=1)

            replay_memory.append((framebuffer, action, reward, next_framebuffer, 0))

            # if i >= batch_size:
            agent.train(get_samples(replay_memory))

        if save_model:
            agent.actor.save(model_savefolder + '_actor')
            agent.critic.save(model_savefolder + '_critic')
            try:
                with open(FILEPATH_FOR_SCORE, "ab") as file:
                    pickle.dump((total_reward, total_hit_count, total_kills, total_frags), file)
            except:
                with open(FILEPATH_FOR_SCORE, "wb") as file:
                    pickle.dump((total_reward, total_hit_count, total_kills, total_frags), file)
            print(f'Models saved')

        # train_rewards.append(total_reward)
        print((total_reward, total_hit_count, total_kills, total_frags))

        # game.new_episode()



        # if not len(train_scores):
        #     train_scores.append(0)
        #
        # train_scores = np.array(train_scores)
        # print(
        #     "Results: mean: {:.1f}±{:.1f},".format(
        #         train_scores.mean(), train_scores.std()
        #     ),
        #     "min: %.1f," % train_scores.min(),
        #     "max: %.1f," % train_scores.max(),
        # )


class MyResNet(Model):
    def __init__(self, action_size):
        super().__init__()
        self.pretrained_model_for_demo = tf.keras.applications.ResNet50(include_top=False,
                                                                        input_shape=resolution_for_buffer,
                                                                        pooling='avg',
                                                                        classes=action_size,
                                                                        weights='imagenet')

        for each_layer in self.pretrained_model_for_demo.layers:
            each_layer.trainable = False

        self.flatten = Flatten()
        self.dense1 = Dense(512, activation='relu')
        self.dense2 = Dense(action_size, activation='softmax')

    def call(self, x):
        # print(x.shape)
        if len(x.shape) == 3:
            #     print('bad')
            x = tf.stack(x[np.newaxis, ...])
        # x = tf.reshape(x, shape=[None, 100, 100, 3])
        x = self.pretrained_model_for_demo(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x


class CriticNet(Model):
    def __init__(self):
        super().__init__()

        self.input1 = tf.keras.layers.InputLayer(input_shape=resolution_for_buffer)

        self.conv1 = tf.keras.layers.Conv2D(8, kernel_size=6, strides=3, activation='relu')
        self.bn1 = tf.keras.layers.BatchNormalization()

        self.conv2 = tf.keras.layers.Conv2D(10, kernel_size=3, strides=2, activation='relu')
        self.bn2 = tf.keras.layers.BatchNormalization()

        self.conv3 = tf.keras.layers.Conv2D(12, kernel_size=3, strides=2, activation='relu')
        self.bn3 = tf.keras.layers.BatchNormalization()

        self.dense1 = tf.keras.layers.Dense(500, activation='relu')
        self.output1 = tf.keras.layers.Dense(1, activation='linear')

        self.flatten = Flatten()

    def call(self, x):
        # print(x.shape)
        if len(x.shape) == 3:
            #     print('bad')
            x = tf.stack(x[np.newaxis, ...])
        # x = tf.reshape(x, shape=[None, 100, 100, 3])
        x = self.input1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.output1(x)
        return x


if __name__ == "__main__":
    game = initialize_game()
    # replay_memory = deque(maxlen=replay_memory_size)
    replay_memory = get_memory_with_gameplay('F:\SIiUM3\ViZDoom\JK_map03\my_gameplay_buffer_smol.pickle')
    actions = [
        # [0, 0, 0, 0, 0, 0, 0, 0, 0],
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
    ]

    print(f'number of possible actions = {len(actions)}')
    agent = ResNetAgent(num_actions=len(actions))

    with tf.device(DEVICE):
        # if not skip_learning:
        print("Starting the training!")

        run(agent, game, replay_memory)

        game.close()
        print("======================================")
        print("Training is finished.")

        game.close()

        # if watch:
        #     game.set_window_visible(True)
        #     game.set_mode(vzd.Mode.ASYNC_PLAYER)
        #     game.init()
        #
        #     for _ in range(episodes_to_watch):
        #         game.new_episode()
        #         while not game.is_episode_finished():
        #             state = preprocess(game.get_state().screen_buffer)
        #             best_action_index = agent.choose_action(state)
        #
        #             # Instead of make_action(a, frame_repeat) in order to make the animation smooth
        #             game.set_action(np.concatenate((actions[best_action_index], (0,0)), axis=0))
        #             for _ in range(frames_per_action):
        #                 game.advance_action()
        #
        #         # Sleep between episodes
        #         sleep(1.0)
        #         score = game.get_total_reward()
        #         print("Total score: ", score)
