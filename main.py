from collections import deque
import numpy as np
from preprocess import Preprocess
from game_env import Doom


stack_size = 4
frame_size = (175, 350, 3)

stacked_frames = deque([np.zeros(frame_size) for i in range(stack_size)], maxlen=stack_size)


def frame_stacking(frame, new_episode=False):
    global stacked_frames

    if new_episode:
        for i in range(stack_size):
            stacked_frames.append(frame)

    else:
        stacked_frames.append(frame)

    return np.stack(stacked_frames, axis=len(frame_size))


def main():
    global frame_size, stack_size

    state_size = list(frame_size)
    state_size.append(stack_size)

    game = Doom()
    no_actions = len(game.actions)

    learning_rate = 0.0002
    no_episodes = 500
    max_steps = 100
    batch_size = 64

    explore_max = 1.
    explore_min = 0.01
    decay_rate = 0.00001

    gamma = 0.95

    pretrain_length  = batch_size
    memory_size = 1000000

    training = True

    episode_render = True



if __name__ == "__main__":
    main()
