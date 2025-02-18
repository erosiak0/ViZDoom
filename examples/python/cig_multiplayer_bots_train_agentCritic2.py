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
from agentCritic2 import *

game = vzd.DoomGame()
# Use CIG example config or your own.
game.load_config("F:/SIiUM3/ViZDoom/scenarios/multi.cfg")
game.add_game_args(
    "-join 127.0.0.1 -port 5029"
)  # Connect to a host for a multiplayer game.
# Name your agent and select color
# colors: 0 - green, 1 - gray, 2 - brown, 3 - red, 4 - light gray, 5 - light brown, 6 - light red, 7 - light blue
game.add_game_args("+name AgentCritic2 +colorset 6")
game.set_doom_map("map01")  # Limited deathmatch.

### Change game mode 
game.set_mode(vzd.Mode.ASYNC_PLAYER)
# game.set_mode(vzd.Mode.ASYNC_SPECTATOR)
# game.set_mode(vzd.Mode.SPECTATOR)
# game.set_mode(vzd.Mode.PLAYER)

game.set_ticrate(35)
# game.set_console_enabled(True)
game.init()

# Three example sample actions
last_frags = 0

# Run this many episodes
episodes = 25_000

### DEFINE YOUR AGENT HERE (or init)
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

agent = AgentCritic(num_actions=len(actions))
replay_memory = deque(maxlen=replay_memory_size)

summary = []
summary_rewards = []
summary_frags = []
framebuffer = np.zeros(resolution_buffer, 'float32')
framebuffer_next = np.zeros(resolution_buffer, 'float32')

try:
    replay_memory = load_pickle('F:/SIiUM3/ViZDoom/buffer/framebuffer_agent3.pickle')
except:
    print("pickle file not exist or is empty")

iteration = 0
for i in range(episodes):
    agent.episode += 1
    print("Episode #" + str(i + 1) + " ::" +str(agent.episode))
    train_scores = []
    rewards = 0

    # Play until the game (episode) is over.
    while not game.is_episode_finished():

        iteration += 1
        # Get the state.
        state = game.get_state()

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
        if i > 4:
            replay_memory.append((framebuffer, action, reward, framebuffer_next, 0))

        ### TRAIN YOUR AGENT HERE
        if len(replay_memory) >= batch_size:    
            screen_buf, act, rewa, next_screen_buf, dones = split_tuple(get_samples(replay_memory))

            agent.train(screen_buf, act, rewa, next_screen_buf, dones)

        rewards += reward

        # Check if player is dead
        if game.is_player_dead():
            train_scores.append(game.get_total_reward())

            print("Player died.")
            # Use this to respawn immediately after death, new state will be available.
            game.respawn_player()

    print(f"Results: {rewards}")



    print("Episode finished.")
    print("************************")
    if save_model:
        print("save model")
        agent.actor.save(f"F:/SIiUM3/ViZDoom/model_actor2_a38/model_backup/actor{agent.episode}")
        agent.critic.save(f"F:/SIiUM3/ViZDoom/model_actor2_a38/model_backup/critic{agent.episode}")

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
    np.save("F:\\SIiUM3\\ViZDoom\\log\\rewards_summary_agentcritic2.npy", np.array(summary_rewards))
    np.save("F:\\SIiUM3\\ViZDoom\\log\\summary_frags_agentcritic2.npy", np.array(summary_frags))

game.close()

