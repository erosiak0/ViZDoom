#!/usr/bin/env python3

#####################################################################
# This script presents how to join and play a deathmatch game,
# that can be hosted using cig_multiplayer_host.py script.
#####################################################################

import itertools as it
from collections import deque
import os
from random import choice
from qLearning import *
import vizdoom as vzd


game = vzd.DoomGame()

# Use CIG example config or your own.
game.load_config(os.path.join(vzd.scenarios_path, "multi.cfg"))

game.set_doom_map("map01")  # Limited deathmatch.
# game.set_doom_map("map02")  # Full deathmatch.

# Join existing game.
game.add_game_args(
    "-join 127.0.0.1 -port 5029"
)  # Connect to a host for a multiplayer game.

# Name your agent and select color
# colors: 0 - green, 1 - gray, 2 - brown, 3 - red, 4 - light gray, 5 - light brown, 6 - light red, 7 - light blue
game.add_game_args("+name AI +colorset 0")
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
n = game.get_available_buttons_size() 
actions = [list(a) for a in it.product([0, 1], repeat=n)]
agent = DQNAgent(num_actions=len(actions))

# Get player's number
player_number = int(game.get_game_variable(vzd.GameVariable.PLAYER_NUMBER))
last_frags = 0
summary = []
summary_rewards = []
# Play until the game (episode) is over.
for episode in range(episodes):
    
    game.new_episode()

    train_scores = []
    rewards = []
    counter = 0
    print(f"Episode #  {episode + 1} Player 1")
    # game.send_game_command('addbot')

    time_start = time()
    while not game.is_episode_finished():
        counter += 1

        # Get the state.
        s = game.get_state()

        # Analyze the state.
        replay_memory, train_scores, action, reward = run(agent, game, replay_memory, actions, train_scores)
        rewards.append(reward)
        if len(replay_memory) > batch_size:
            agent.train_dqn(get_samples(replay_memory))

        if len(replay_memory)%target_net_update_steps == 0:
            agent.update_target_net()

        # Make your action.

        game.make_action(actions[action], player_skip)

        frags = game.get_game_variable(vzd.GameVariable.FRAGCOUNT)
        if frags != last_frags:
            last_frags = frags
            print("Player " + str(player_number) + " has " + str(frags) + " frags.")

        # Check if player is dead
        if game.is_player_dead():
            print("Player " + str(player_number) + " died.")
            # Use this to respawn immediately after death, new state will be available.
            game.respawn_player()

    train_scores = np.array(train_scores)
    if save_model:
        print("save model")
        agent.dqn.save(model_savefolder)
        # agent.dqn.save_weights(model_savefolder.format(epoch=episode))

    try:
        print(
        "Results: mean: {:.1f}±{:.1f},".format(
            train_scores.mean(), train_scores.std()
        ),
        "min: %.1f," % train_scores.min(),
        "max: %.1f," % train_scores.max(),
        )
    except:
        print("Empty train_scores")
    try:
        print(
        "Results: mean: {:.1f}±{:.1f},".format(
            rewards.mean(), rewards.std()
        ),
        "min: %.1f," % rewards.min(),
        "max: %.1f," % rewards.max(),
        )
    except:
        print("Empty rewards")
    print(f"counter: {counter}","Total elapsed time: %.2f minutes" % ((time() - time_start) / 60.0))

    # game.new_episode()
    
    summary.append(train_scores)
    summary_rewards.append(rewards)
    game.respawn_player()

print(summary)
print(summary_rewards)
# with open("F:/SIiUM3/ViZDoom/log/train001_"+ str(player_number), 'w') as f:
#     f.write(summary)
game.close()
