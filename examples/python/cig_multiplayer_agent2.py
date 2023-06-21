#!/usr/bin/env python3

#####################################################################
# This script presents how to join and play a deathmatch game,
# that can be hosted using cig_multiplayer_host.py script.
#####################################################################

import os

from agentCriticbot import *

import vizdoom as vzd

load = False
game = vzd.DoomGame()

# Use CIG example config or your own.
game.load_config(os.path.join(vzd.scenarios_path, "F:/SIiUM3/ViZDoom/scenarios/multi.cfg"))

# # Join existing game.
game.add_game_args(
    "-join 127.0.0.1 -port 5029"
)  # Connect to a host for a multiplayer game.

# Name your agent and select color
# colors: 0 - green, 1 - gray, 2 - brown, 3 - red, 4 - light gray, 5 - light brown, 6 - light red, 7 - light blue
game.add_game_args("+name Agent2 +colorset 2")

# During the competition, async mode will be forced for all agents.
# game.set_mode(vzd.Mode.PLAYER)
game.set_mode(vzd.Mode.PLAYER)
# game.set_window_visible(True)
game.set_window_visible(True)

game.init()

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

# Three example sample actions
agent = AgentCritic(num_actions=len(actions))

agent.actor = tf.keras.models.load_model("F:\\SIiUM3\\ViZDoom\\Models\\map01\\actor89",  compile=False)
agent.critic = tf.keras.models.load_model("F:\\SIiUM3\\ViZDoom\\Models\\map01\\critic89",  compile=False)

resolution = (100, 75)
resolution_buffer = (100, 75*3)
episodes = 25_000
# Get player's number
player_number = int(game.get_game_variable(vzd.GameVariable.PLAYER_NUMBER))
last_frags = 0

framebuffer = np.zeros(resolution_buffer, 'float32')

# Play until the game (episode) is over.
for episode in range(episodes):

    print(f"Episode #  {episode + 1} Player "+ str(player_number))

    while not game.is_episode_finished():
        state = game.get_state()

        # Analyze the state.

        screen_buf = preprocess(state.screen_buffer)
        framebuffer = update_buffer(screen_buf, framebuffer[:,:-resolution[1]])
        action = agent.choose_action(framebuffer)
        action_reward = game.make_action(actions[action], frames_per_action)
        if game.is_episode_finished():
            break

        frags = game.get_game_variable(vzd.GameVariable.FRAGCOUNT)

        # Check if player is dead
        if game.is_player_dead():
            print("Player " + str(player_number) + " died.")
            # Use this to respawn immediately after death, new state will be available.
            game.respawn_player()


    print("Episode finished.")
    print("************************")
    print("Results:")
    server_state = game.get_server_state()
    for i in range(len(server_state.players_in_game)):
        if server_state.players_in_game[i]:
            print(
                server_state.players_names[i]
                + ": "
                + str(server_state.players_frags[i])
            )
    print("************************")    
    game.new_episode()
    
game.close()
