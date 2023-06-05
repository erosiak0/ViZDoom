#!/usr/bin/env python3

#####################################################################
# This script presents how to host a deathmatch game.
#####################################################################

import os
from random import choice
import itertools as it

import vizdoom as vzd
# from qLearning import *
from agentCritic import *

game = vzd.DoomGame()

# Use CIG example config or your own.
game.load_config(os.path.join(vzd.scenarios_path, "multi.cfg"))

# game.set_doom_map("map02")  # Full deathmatch.

# Host game with options that will be used in the competition.
game.add_game_args(
    "-host 2 "
    # This machine will function as a host for a multiplayer game with this many players (including this machine).
    # It will wait for other machines to connect using the -join parameter and then start the game when everyone is connected.
    "-port 5029 "  # Specifies the port (default is 5029).
    "+viz_connect_timeout 60 "  # Specifies the time (in seconds), that the host will wait for other players (default is 60).
    "-deathmatch "  # Deathmatch rules are used for the game.
    "+timelimit 1.0 "  # The game (episode) will end after this many minutes have elapsed.
    "+sv_forcerespawn 1 "  # Players will respawn automatically after they die.
    "+sv_noautoaim 1 "  # Autoaim is disabled for all players.
    "+sv_respawnprotect 1 "  # Players will be invulnerable for two second after spawning.
    "+sv_spawnfarthest 1 "  # Players will be spawned as far as possible from any other players.
    "+sv_nocrouch 1 "  # Disables crouching.
    "+viz_respawn_delay 2 "  # Sets delay between respawns (in seconds, default is 0).
    "+viz_nocheat 1"
)  # Disables depth and labels buffer and the ability to use commands that could interfere with multiplayer game.

# This can be used to host game without taking part in it (can be simply added as argument of vizdoom executable).
# game.add_game_args("+viz_spectator 1")

# Name your agent and select color
# colors: 0 - green, 1 - gray, 2 - brown, 3 - red, 4 - light gray, 5 - light brown, 6 - light red, 7 - light blue
game.add_game_args("+name Host +colorset 4")
# game.add_game_args("+viz_bots_path ../../src/perfect_bots.cfg")
game.add_game_args("+viz_bots_path F:/SIiUM3/ViZDoom/scenarios/bots.cfg")

# During the competition, async mode will be forced for all agents.
# game.set_mode(vzd.Mode.PLAYER)
game.set_mode(vzd.Mode.ASYNC_PLAYER)

# game.set_window_visible(False)
game.set_window_visible(True)

game.init()

# Three example sample actions
n = game.get_available_buttons_size()
actions = [list(a) for a in it.product([0, 1], repeat=n)]
for i in range(episodes):
    # game.new_episode()
    game.send_game_command("removebots")

    for j in range(10):
        game.send_game_command('addbot')
    
    print(f"Episode #  {i + 1} Player host")
# Play until the game (episode) is over.
    while not game.is_episode_finished():
        sleep(0.1)

        # game.send_game_command('addbot')

        # Get the state.
        s = game.get_state()

        # Analyze the state.

        # Make your action.
        game.make_action(choice(actions), player_skip)
        # Check if player is dead
        if game.is_player_dead():
            print("Player host died.")

            # Use this to respawn immediately after death, new state will be available.
            game.respawn_player()
    print(
        "Player host frags:",
        game.get_game_variable(vzd.GameVariable.FRAGCOUNT),
    )

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
    # game.respawn_player()
game.close()
