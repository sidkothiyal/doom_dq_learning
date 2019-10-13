from vizdoom import *
import random
import time


class Doom:

    def __init__(self, test=False):
        self.game_environment = None
        self.actions = None
        self.action_desc = None

        self.config = "basic.cfg"
        self.scenario_path = "basic.wad"

        self.init_game()

        if test:
            self.test_environment()

    def init_game(self):
        self.game_environment = DoomGame()

        self.game_environment.load_config(self.config)
        self.game_environment.set_doom_scenario_path(self.scenario_path)

        self.init_actions()

    def start_game(self):
        self.game_environment.init()

    def init_actions(self):
        self.action_desc = ['move_left', 'move_right', 'shoot']

        self.actions = []

        for i in range(len(self.action_desc)):
            self.actions.append([0] * len(self.action_desc))
            self.actions[-1][i] = 1

    def restart_episode(self):
        self.game_environment.new_episode()

    def is_episode_finished(self):
        return self.game_environment.is_episode_finished()

    def get_environment_state(self):
        state = self.game_environment.get_state()
        img = state.screen_buffer
        game_vars = state.game_variables

        return img, game_vars

    def close_environment(self):
        self.game_environment.close()

    def take_action(self, action):
        return self.game_environment.make_action(action)

    def test_environment(self):
        episodes = 10

        for i in range(episodes):
            self.restart_episode()

            while not self.is_episode_finished():
                img, game_vars = self.get_environment_state()

                action = random.choice(self.actions)
                print(action)
                reward = self.take_action(action)
                print("Reward: ", reward)
                time.sleep(0.02)

            print("Result ", self.game_environment.get_total_reward())
            time.sleep(2)
        self.close_environment()


if __name__ == "__main__":
    d = Doom(test=True)
