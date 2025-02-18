#!/usr/bin/env python3

#####################################################################
# This script presents how to play a deathmatch game with built-in bots.
#####################################################################

import os
from random import choice
import time
# from agent_example import Agent
import vizdoom as vzd
# from agentCritic1 import *
from agentCritic5 import *

game = vzd.DoomGame()
# Use CIG example config or your own.
game.load_config("F:/SIiUM3/ViZDoom/scenarios/multi.cfg")

# Start multiplayer game only with your AI
# (with options that will be used in the competition, details in cig_mutliplayer_host.py example).
game.add_game_args(
    "-host 1 "
    # This machine will function as a host for a multiplayer game with this many players (including this machine).
    # It will wait for other machines to connect using the -join parameter and then start the game when everyone is connected.
    "-port 5029 "  # Specifies the port (default is 5029).
    "+viz_connect_timeout 60 "  # Specifies the time (in seconds), that the host will wait for other players (default is 60).
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

# Bots are loaded from file, that by default is bots.cfg located in the same dir as ViZDoom exe
# Other location of bots configuration can be specified by passing this argument
game.add_game_args("+viz_bots_path F:/SIiUM3/ViZDoom/scenarios/bots.cfg")

game.set_window_visible(False)

# Name your agent and select color
# colors: 0 - green, 1 - gray, 2 - brown, 3 - red, 4 - light gray, 5 - light brown, 6 - light red, 7 - light blue
game.add_game_args("+name ActorCritic3 +colorset 0")
game.set_doom_map("map01")  # Limited deathmatch.

### Change game mode 
# game.set_mode(vzd.Mode.ASYNC_PLAYER)
# game.set_mode(vzd.Mode.ASYNC_SPECTATOR)
# game.set_mode(vzd.Mode.SPECTATOR)
game.set_mode(vzd.Mode.PLAYER)
# game.set_window_visible(True)

game.set_ticrate(35)
# game.set_console_enabled(True)
game.init()

# Three example sample actions
last_frags = 0

# Play with this many bots
bots = 10# 7

# Run this many episodes
episodes = 25_000

### DEFINE YOUR AGENT HERE (or init)

agent = AgentCritic(num_actions=len(actions))
replay_memory = deque(maxlen=replay_memory_size)

summary = []
summary_rewards = []
summary_frags = []
framebuffer = np.zeros(resolution_buffer, 'float32')
framebuffer_next = np.zeros(resolution_buffer, 'float32')

# try:
#     replay_memory = load_pickle('F:/SIiUM3/ViZDoom/buffer/framebuffer_agent5_map03.pickle')
# except:
#     print("pickle file not exist or is empty")

iteration = 0
for i in range(episodes):
    agent.episode += 1
    print("Episode #" + str(i + 1) + " ::" +str(agent.episode))
    train_scores = []
    rewards = 0
    ### Add specific number of bots
    # edit this file to adjust bots).
    game.send_game_command("removebots")
    for i in range(bots):
        game.send_game_command("addbot")

    ### Change the bots difficulty
    # Valid args: 1, 2, 3, 4, 5 (1 - easy, 5 - very hard)
    game.send_game_command("pukename change_difficulty 1")

    ### Change number of monster to spawn 
    # Valid args: >= 0
    # game.send_game_command(f"pukename change_num_of_monster_to_spawn 0")
    counter = 0
    # Play until the game (episode) is over.
    while not game.is_episode_finished():
        counter += 1
        iteration += 1
        # Get the state.
        state = game.get_state()

        screen_buf = preprocess(state.screen_buffer)
        
        framebuffer = update_buffer(screen_buf, framebuffer[:,:-resolution[1]])
        action = agent.choose_action(framebuffer)
        game_variables = state.game_variables
        # print(f"game_variables: {game_variables}")
        action_reward = game.make_action(actions[action], frames_per_action)

        if game.is_episode_finished():
            break

        next_state = game.get_state()
        game_variables_next = next_state.game_variables
        reward = get_reward(action_reward, game_variables, game_variables_next)
        next_screen_buf = preprocess(next_state.screen_buffer)
        framebuffer_next = update_buffer(next_screen_buf, framebuffer_next[:,:-resolution[1]])

        replay_memory.append((framebuffer, action, reward, framebuffer_next, 0))

        ### TRAIN YOUR AGENT HERE
        if len(replay_memory) >= batch_size:    
            screen_buf, act, rewa, next_screen_buf, dones = split_tuple(get_samples(replay_memory))

            agent.train(screen_buf, act, rewa, next_screen_buf, dones)

        rewards += reward
        # print(f"rewards: {rewards}")

        # Check if player is dead
        if game.is_player_dead():
            train_scores.append(game.get_total_reward())

            print("Player died.")
            # Use this to respawn immediately after death, new state will be available.
            game.respawn_player()

    print(f"Results: {rewards}, counter: {counter}")



    print("Episode finished.")
    print("************************")
    if save_model:
        print("save model")
        agent.actor_critic.save(f"F:/SIiUM3/ViZDoom/model_actor5_a38_map03/model_backup/actor{agent.episode}")
        # agent.critic.save(f"F:/SIiUM3/ViZDoom/model_actor5_a38_map03/model_backup/critic{agent.episode}")

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

    # Starts a new episode. All players have to call new_episode() in multiplayer mode.
    game.new_episode()
    summary.append(train_scores)
    summary_rewards.append(rewards)
    summary_frags.append(server_state.players_frags)
    np.save("F:\\SIiUM3\\ViZDoom\\log\\rewards_summary_agentcritic_map03.npy", np.array(summary_rewards))
    np.save("F:\\SIiUM3\\ViZDoom\\log\\summary_frags_agentcritic_map03.npy", np.array(summary_frags))

game.close()

