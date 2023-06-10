#!/usr/bin/env python3

#####################################################################
# This script presents how to play a deathmatch game with built-in bots.
#####################################################################

import os
from random import choice
import time
# from agent_example import Agent
import vizdoom as vzd
from agentCritic3 import *

game = vzd.DoomGame()
# Use CIG example config or your own.
game.load_config("F:/SIiUM3/ViZDoom/scenarios/multi.cfg")

import csv
# Start multiplayer game only with your AI
# (with options that will be used in the competition, details in cig_mutliplayer_host.py example).
game.add_game_args(
    "-host 1 "
    # This machine will function as a host for a multiplayer game with this many players (including this machine).
    # It will wait for other machines to connect using the -join parameter and then start the game when everyone is connected.
    # "-port 5029 "  # Specifies the port (default is 5029).
    # "+viz_connect_timeout 60 "  # Specifies the time (in seconds), that the host will wait for other players (default is 60).
    "-deathmatch "  # Deathmatch rules are used for the game.
    "+timelimit 3.0 "  # The game (episode) will end after this many minutes have elapsed.
    "+sv_forcerespawn 1 "  # Players will respawn automatically after they die.
    "+sv_noautoaim 1 "  # Autoaim is disabled for all players.
    "+sv_respawnprotect 1 "  # Players will be invulnerable for two second after spawning.
    "+sv_spawnfarthest 1 "  # Players will be spawned as far as possible from any other players.
    "+sv_nocrouch 1 "  # Disables crouching.
    "+viz_respawn_delay 0 "  # Sets delay between respawns (in seconds, default is 0).
    "+viz_nocheat 1"
)
game.set_doom_map("map03")  # Limited deathmatch.

# Bots are loaded from file, that by default is bots.cfg located in the same dir as ViZDoom exe
# Other location of bots configuration can be specified by passing this argument
game.add_game_args("+viz_bots_path F:/SIiUM3/ViZDoom/scenarios/bots.cfg")


# Name your agent and select color
# colors: 0 - green, 1 - gray, 2 - brown, 3 - red, 4 - light gray, 5 - light brown, 6 - light red, 7 - light blue
game.add_game_args("+name AI +colorset 0")

### Change game mode 
# game.set_mode(vzd.Mode.ASYNC_PLAYER)
# game.set_mode(vzd.Mode.ASYNC_SPECTATOR)
game.set_mode(vzd.Mode.SPECTATOR)
# game.set_mode(vzd.Mode.PLAYER)

game.set_ticrate(35)
# game.set_console_enabled(True)
game.init()

# Three example sample actions
# n = game.get_available_buttons_size() -2
# actions = [list([*a, 0,0]) for a in it.product([0, 1], repeat=n)]

dict_actions = {}
for i, a in enumerate(actions):
    dict_actions[i] = a
last_frags = 0

# Play with this many bots
bots = 10

# Run this many episodes
episodes = 3 # 25_000

### DEFINE YOUR AGENT HERE (or init)

agent = AgentCritic(num_actions=len(actions))
replay_memory = deque(maxlen=replay_memory_size)

framebuffer = np.zeros(resolution_buffer, 'float32')
framebuffer_next = np.zeros(resolution_buffer, 'float32')
filepath = 'F:/SIiUM3/ViZDoom/buffer/framebuffer_agent3_map03.pickle'
print((vzd.scenarios_path))
with open(filepath, "wb") as file:
            

    iteration = 0
    for i in range(episodes):
        print("Episode #" + str(i + 1))

        ### Add specific number of bots
        # edit this file to adjust bots).
        game.send_game_command("removebots")
        for i in range(bots):
            game.send_game_command("addbot")

        game.send_game_command("pukename change_difficulty 3")

        while not game.is_episode_finished():

            iteration += 1
            # Get the state.
            state = game.get_state()

            screen_buf = preprocess(state.screen_buffer)
            framebuffer = update_buffer(screen_buf, framebuffer[:,:-90])
            game.advance_action()
            act = game.get_last_action()
            action = [k for k, v in dict_actions.items() if v == act]
            print(f"action: {action}, act: {act}")
            game_variables = state.game_variables
            action_reward = 0
            if game.is_episode_finished():
                break

            next_state = game.get_state()
            game_variables_next = next_state.game_variables
            reward = get_reward(action_reward, game_variables, game_variables_next)
            next_screen_buf = preprocess(next_state.screen_buffer)
            framebuffer_next = update_buffer(next_screen_buf, framebuffer_next[:,:-90])
            replay_memory.append((framebuffer, action, reward, framebuffer_next, 0))

            if action != [] and action != 0:
                saved_memory = {
                "framebuffer":framebuffer.tolist(),
                "action":action, 
                "reward":reward,
                "framebuffer_next":framebuffer_next.tolist()
                }
                pickle.dump([saved_memory], file)

            # Check if player is dead
            if game.is_player_dead():

                print("Player died.")
                # Use this to respawn immediately after death, new state will be available.
                game.respawn_player()

        game.new_episode()

game.close()

