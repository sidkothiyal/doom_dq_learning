from collections import deque
import numpy as np
from preprocess import Preprocess
from game_env import Doom
import tensorflow as tf
from model import DeepQNetwork
from memory import Memory
import random


stack_size = 4
frame_size = (64, 128)

stacked_frames = deque([np.zeros(frame_size) for i in range(stack_size)], maxlen=stack_size)


def frame_stacking(frame, new_episode=False):
    global stacked_frames

    if new_episode:
        for i in range(stack_size):
            stacked_frames.append(frame)

    else:
        stacked_frames.append(frame)

    return np.stack(stacked_frames, axis=len(frame_size))

def testing_trained_agent():
    """Testing Trained Agent"""
    global frame_size, stack_size

    with tf.Session() as sess:
        game = Doom()

        state_size = list(frame_size)
        state_size.append(stack_size)

        no_actions = len(game.actions)

        learning_rate = 0.0002

        deep_Q_network = DeepQNetwork(state_size, no_actions, learning_rate)

        totalScore = 0

        saver = tf.train.Saver()

        saver.restore(sess, "./models/model.ckpt")

        game.start_game()

        for i in range(1):
            done = False

            game.restart_episode()

            img, game_vars = game.get_environment_state()
            state = frame_stacking(img, True)

            while not game.is_episode_finished():
                Qs = sess.run(deep_Q_network.output, feed_dict={deep_Q_network.inputs: state.reshape((1, *state.shape))})

                choice = np.argmax(Qs)
                action = game.actions[int(choice)]

                game.take_action(action)

                done = game.is_episode_finished()

                score = game.game_environment.get_total_reward()

                if done:
                    break

                else:
                    print("else ")
                    next_img, next_game_vars = game.get_environment_state()
                    next_state = frame_stacking(next_img, False)
                    state = next_state

            score = game.game_environment.get_total_reward()
            print("Score: ", score)
        game.close_environment()



def main():
    global frame_size, stack_size

    state_size = list(frame_size)
    state_size.append(stack_size)

    game = Doom()
    no_actions = len(game.actions)

    learning_rate = 0.002
    no_episodes = 500
    max_steps = 100
    batch_size = 32

    explore_max = 1.
    explore_min = 0.01
    decay_rate = 0.00001

    gamma = 0.95

    pretrain_length  = batch_size
    memory_size = 1000000

    training = True

    episode_render = True

    tf.reset_default_graph()

    deep_Q_network = DeepQNetwork(state_size, no_actions, learning_rate)

    memory = Memory(max_size=memory_size)
    game.start_game()
    game.restart_episode()

    for i in range(pretrain_length):
        if i == 0:
            img, game_vars = game.get_environment_state()
            state = frame_stacking(img, True)

        action = random.choice(game.actions)

        reward = game.take_action(action)

        done = game.is_episode_finished()

        if done:
            next_state = np.zeros(state.shape)
            memory.add((state, action, reward, next_state, done))

            game.restart_episode()
            img, game_vars = game.get_environment_state()
            state = frame_stacking(img, True)

        else:
            next_img, next_game_vars = game.get_environment_state()
            next_state = frame_stacking(img, False)

            memory.add((state, action, reward, next_state, done))

            state = next_state

    writer = tf.summary.FileWriter("./tensorboard/dqn/1")

    tf.summary.scalar("Loss", deep_Q_network.loss)

    write_op = tf.summary.merge_all()

    """Prediction """

    def predict_action(curr_decay_step, curr_state):
        exp_exp_tradeoff = np.random.rand()

        curr_explore_prob = explore_min + ((explore_max - explore_min) * np.exp(-decay_rate * curr_decay_step))

        if curr_explore_prob > exp_exp_tradeoff:
            curr_action = random.choice(game.actions)

        else:
            Qs = sess.run(deep_Q_network.output, feed_dict={deep_Q_network.inputs: curr_state.reshape((1, *curr_state.shape))})

            choice = np.argmax(Qs)
            curr_action = game.actions[choice]

        return curr_action, curr_explore_prob

    """Training Agent"""
    saver = tf.train.Saver()

    if training:
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            decay_step = 0

            game.start_game()

            for episode in range(no_episodes):
                step = 0

                episode_rewards = []

                game.restart_episode()
                img, game_vars = game.get_environment_state()

                state = frame_stacking(img, True)

                while step < max_steps:
                    step += 1

                    decay_step += 1

                    action, explore_prob = predict_action(decay_step, state)

                    reward = game.take_action(action)

                    done = game.is_episode_finished()

                    episode_rewards.append(reward)

                    if done:
                        next_img = np.zeros(frame_size, dtype=np.int)
                        next_state = frame_stacking(next_img, False)

                        step = max_steps

                        total_rewards = np.sum(episode_rewards)

                        print("Episode No. {}".format(episode),
                              "Total reward: {}".format(total_rewards),
                              "Training Loss: {:.4f}".format(loss_val),
                              "Explore Prob: {:.4f}".format(explore_prob))

                        memory.add((state, action, reward, next_state, done))

                    else:
                        next_img, next_game_vars = game.get_environment_state()
                        next_state = frame_stacking(next_img, False)

                        memory.add((state, action, reward, next_state, done))

                        state = next_state

                    """Learning Part """
                    """Get mini-batches from memory and train"""
                    batch = memory.sample(batch_size)

                    states_mb = []
                    actions_mb = []
                    rewards_mb = []
                    next_states_mb = []
                    dones_mb = []

                    for each in batch:
                        states_mb.append(each[0])
                        actions_mb.append(each[1])
                        rewards_mb.append(each[2])
                        next_states_mb.append(each[3])
                        dones_mb.append(each[4])

                    states_mb = np.array(states_mb)
                    actions_mb = np.array(actions_mb)
                    rewards_mb = np.array(rewards_mb)
                    next_states_mb = np.array(next_states_mb)
                    dones_mb = np.array(dones_mb)

                    target_Qs_batch = []

                    Qs_next_state = sess.run(deep_Q_network.output, feed_dict={deep_Q_network.inputs: next_states_mb})

                    for i in range(0, len(batch)):
                        terminal = dones_mb[i]

                        if terminal:
                            target_Qs_batch.append(rewards_mb[i])

                        else:
                            target = rewards_mb[i] + (gamma * np.max(Qs_next_state[i]))
                            target_Qs_batch.append(target)

                    targets_mb = np.array(target_Qs_batch)

                    loss_val, _ = sess.run([deep_Q_network.loss, deep_Q_network.optimizer],
                                           feed_dict={deep_Q_network.inputs: states_mb,
                                                      deep_Q_network.target_Q: targets_mb,
                                                      deep_Q_network.actions: actions_mb})

                    summary = sess.run(write_op, feed_dict={deep_Q_network.inputs: states_mb,
                                                            deep_Q_network.target_Q: targets_mb,
                                                            deep_Q_network.actions: actions_mb})

                    writer.add_summary(summary, episode)
                    writer.flush()

                if episode % 5 == 0:
                    save_path = saver.save(sess, "./models/model.ckpt")
                    print("Model Saved")


if __name__ == "__main__":
    main()
    testing_trained_agent()