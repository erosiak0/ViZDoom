#!/usr/bin/env python3

#####################################################################
# This script presents how to join and play a deathmatch game,
# that can be hosted using cig_multiplayer_host.py script.
#####################################################################

import itertools as it
from collections import deque
import os
from random import choice
# from agentCritic import *
# from agentCritic1 import *
from agentCritic2 import *

import vizdoom as vzd


game = vzd.DoomGame()

# Use CIG example config or your own.
game.load_config(os.path.join(vzd.scenarios_path, "F:/SIiUM3/ViZDoom/scenarios/multi.cfg"))

game.set_doom_map("map04")  # Limited deathmatch.
# game.set_doom_map("map02")  # Full deathmatch.

# # Join existing game.
game.add_game_args(
    "-join 127.0.0.1 -port 5029"
)  # Connect to a host for a multiplayer game.

# Name your agent and select color
# colors: 0 - green, 1 - gray, 2 - brown, 3 - red, 4 - light gray, 5 - light brown, 6 - light red, 7 - light blue
game.add_game_args("+name ActorCritic +colorset 3")

# game.set_screen_format(vzd.ScreenFormat.GRAY8)
# game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
# During the competition, async mode will be forced for all agents.
# game.set_mode(Mode.PLAYER)
game.set_mode(vzd.Mode.ASYNC_PLAYER)
game.set_window_visible(True)
# game.set_window_visible(False)

game.init()

replay_memory = deque(maxlen=replay_memory_size)

# Three example sample actions
n = game.get_available_buttons_size() -2
actions = [list(a) for a in it.product([0, 1], repeat=n)]
# actions = [
#     [1, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 1, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 1, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 1, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 1, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 1, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 1, 0, 0],
# ]
agent = AgentCritic(num_actions=len(actions))

# Get player's number
player_number = int(game.get_game_variable(vzd.GameVariable.PLAYER_NUMBER))
last_frags = 0
summary = []
summary_rewards = []
framebuffer = np.zeros(resolution_buffer, 'float32')
framebuffer_next = np.zeros(resolution_buffer, 'float32')
# print(f"len(game.get_game_variable): {game.get_game_variable}")
# Play until the game (episode) is over.
for episode in range(episodes):
    # game.new_episode()
    # for j in range(5):
    #     game.send_game_command('addbot')

    train_scores = []
    counter = 0
    print(f"Episode #  {episode + 1} Player "+ str(player_number))
    time_start = time()
    rewards = 0
    while not game.is_episode_finished():
        sleep(0.1)

        # for j in range(5):
        #     game.send_game_command('addbot')
        counter += 1
        # Get the state.
        state = game.get_state()


        # Analyze the state.

        screen_buf = preprocess(state.screen_buffer)
        framebuffer = update_buffer(screen_buf, framebuffer[:,:-90])
        action = agent.choose_action(framebuffer)
        game_variables = state.game_variables
        action_reward = game.make_action(actions[action], frames_per_action)
        if game.is_episode_finished():
            break

        next_state = game.get_state()
        game_variables_next = next_state.game_variables
        reward = get_reward(action_reward, game_variables, game_variables_next)
        next_screen_buf = preprocess(next_state.screen_buffer)
        framebuffer_next = update_buffer(next_screen_buf, framebuffer_next[:,:-90])
        replay_memory.append((framebuffer, action, reward, framebuffer_next, 0))




        if len(replay_memory) >= batch_size:    
            screen_buf, act, rewards, next_screen_buf, dones = split_tuple(get_samples(replay_memory))

            agent.train(screen_buf, act, rewards, next_screen_buf, dones)

        rewards += reward

        frags = game.get_game_variable(vzd.GameVariable.FRAGCOUNT)

        # Check if player is dead
        if game.is_player_dead():
            train_scores.append(game.get_total_reward())

            print("Player " + str(player_number) + " died.")
            # Use this to respawn immediately after death, new state will be available.
            game.respawn_player()

    if save_model:
        print("save model")
        agent.actor.save(model_savefolder_actor)
        agent.critic.save(model_savefolder_critic)

    try:
        train_scores = np.array(train_scores)
        print(
        "Results: mean: {:.1f}±{:.1f},".format(
            train_scores.mean(), train_scores.std()
        ),
        "min: %.1f," % train_scores.min(),
        "max: %.1f," % train_scores.max(),
        
        )
    except:
        print("Empty train_scores")

    print(f"Results: {rewards}")


    print(f"counter: {counter}","Total elapsed time: %.2f minutes" % ((time() - time_start) / 60.0))
    print("Episode finished.")
    print("************************")
    if save_model:
        print("save model")
        agent.actor.save(model_savefolder_actor)
        agent.critic.save(model_savefolder_critic)

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
    
    summary.append(train_scores)
    summary_rewards.append(rewards)

game.close()
