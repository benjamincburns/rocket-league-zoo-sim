"""
The Match object.
"""

from rlzoo_sim.envs.environment import Environment
from rlzoo_sim.utils.gamestates import GameState
from rlzoo_sim.utils import common_values
from gymnasium import spaces
import numpy as np
from typing import Dict, List, Union, Any


class Match(Environment):
    MAX_TEAM_SIZE = 4

    def __init__(self,
                 reward_function,
                 terminal_conditions,
                 obs_builder,
                 action_parser,
                 state_setter,
                 team_size=1,
                 spawn_opponents=False):
        super().__init__()

        self.team_size = team_size
        self.spawn_opponents = spawn_opponents
        self._reward_fn = reward_function
        self._terminal_conditions = terminal_conditions
        self._obs_builder = obs_builder
        self._action_parser = action_parser
        self._state_setter = state_setter

        if type(terminal_conditions) not in (tuple, list):
            self._terminal_conditions = [terminal_conditions, ]

        self.agents = self.team_size * 2 if self.spawn_opponents else self.team_size

        self.observation_space = None
        self._auto_detect_obs_space()
        self.action_space = self._action_parser.get_action_space()

        self._prev_actions = np.zeros((Match.MAX_TEAM_SIZE*2, 8), dtype=float)
        self._spectator_ids = None

        self.last_touch = None
        self._initial_score = 0

    def episode_reset(self, initial_state: GameState):
        self._spectator_ids = [p.car_id for p in initial_state.players]
        self._prev_actions.fill(0)
        for condition in self._terminal_conditions:
            condition.reset(initial_state)
        self._reward_fn.reset(initial_state)
        self._obs_builder.reset(initial_state)
        self.last_touch = None
        self._initial_score = initial_state.blue_score - initial_state.orange_score

    def build_observations(self, state) -> Dict[str, Any]:
        observations = {}

        self._obs_builder.pre_step(state)

        for i, player in enumerate(state.players):
            obs = self._obs_builder.build_obs(player, state, self._prev_actions[i])
            team = "blue" if player.car_id < 5 else "orange"
            observations[f"{team}_{player.car_id}"] = obs

        if state.last_touch is None:
            state.last_touch = self.last_touch
        else:
            self.last_touch = state.last_touch

        return observations

    def get_rewards(self, state, terminated) -> Dict[str, float]:
        rewards = {}

        self._reward_fn.pre_step(state)
        for i, player in enumerate(state.players):
            if terminated:
                reward = self._reward_fn.get_final_reward(player, state, self._prev_actions[i])
            else:
                reward = self._reward_fn.get_reward(player, state, self._prev_actions[i])

            team = "blue" if player.car_id < 5 else "orange"
            rewards[f"{team}_{player.car_id}"] = reward

        return rewards

    def is_terminated(self, state):
        for condition in self._terminal_conditions:
            if condition.is_terminal(state) and "timeout" not in condition.__class__.__name__.lower():
                return True
        return False

    def is_truncated(self, state):
        for condition in self._terminal_conditions:
            # HACK: this is a hack to get around the fact that there isn't
            # a concrete timeout signal in the terminal condition interface
            if condition.is_terminal(state) and "timeout" in condition.__class__.__name__.lower():
                return True
        return False

    def get_result(self, state: GameState):
        current_score = state.blue_score - state.orange_score
        return current_score - self._initial_score

    def parse_actions(self, actions: Any, state: GameState) -> np.ndarray:
        # Prevent people from accidentally modifying numpy arrays inside the ActionParser
        if isinstance(actions, np.ndarray):
            actions = np.copy(actions)
        return self._action_parser.parse_actions(actions, state)

    def format_actions(self, actions: Dict[str, np.ndarray]):
        self._prev_actions[:len(actions)] = actions[:]
        acts = []
        for i in range(len(actions)):
            acts.append(float(self._spectator_ids[i]))
            for act in actions[i]:
                acts.append(float(act))

        return acts

    def get_reset_state(self) -> list:
        new_state = self._state_setter.build_wrapper(self.team_size, self.spawn_opponents)
        self._state_setter.reset(new_state)
        return new_state.format_state()

    def _auto_detect_obs_space(self):
        from rlzoo_sim.utils.gamestates.player_data import PlayerData

        num_cars = self.team_size * 2 if self.spawn_opponents else self.team_size
        empty_player_packets = []
        for i in range(num_cars):
            player_packet = PlayerData()
            player_packet.car_id = i
            empty_player_packets.append(player_packet)

        empty_game_state = GameState()
        prev_inputs = np.zeros(common_values.NUM_ACTIONS)

        empty_game_state.players = empty_player_packets

        self.observation_space = self._obs_builder.get_obs_space()
        if self.observation_space is None:
            self._obs_builder.reset(empty_game_state)
            self._obs_builder.pre_step(empty_game_state)
            obs_shape = np.shape(self._obs_builder.build_obs(empty_player_packets[0], empty_game_state, prev_inputs))

            self.observation_space = spaces.Box(-np.inf, np.inf, shape=obs_shape)
