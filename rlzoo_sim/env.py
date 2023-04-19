"""
    The Rocket League gymnasium environment.
"""
from typing import List, Union, Tuple, Dict, Any

from gymnasium import Space

from rlzoo_sim.utils import common_values
from rlzoo_sim.utils.terminal_conditions import common_conditions
from rlzoo_sim.utils.action_parsers.default_act import DefaultAction
from rlzoo_sim.utils.obs_builders.default_obs import DefaultObs
from rlzoo_sim.utils.reward_functions.default_reward import DefaultReward
from rlzoo_sim.utils.state_setters.default_state import DefaultState
from rlzoo_sim.envs import Match
from gymnasium.spaces import Dict as DictSpace
from pettingzoo import ParallelEnv
from rlzoo_sim.simulator import RocketSimGame
import numpy as np
import RocketSim as rsim


class RocketSimEnv(ParallelEnv):
    def __init__(self,
                 tick_skip: int = 8,
                 spawn_opponents: bool = False,
                 team_size: int = 1,
                 gravity: float = 1,
                 boost_consumption: float = 1,
                 copy_gamestate_every_step = True,
                 dodge_deadzone = 0.8,
                 terminal_conditions: List[object] = (common_conditions.TimeoutCondition(225), common_conditions.GoalScoredCondition()),
                 reward_fn: object = DefaultReward(),
                 obs_builder: object = DefaultObs(),
                 action_parser: object = DefaultAction(),
                 state_setter: object = DefaultState()):

        super().__init__()

        self.gravity = gravity
        self.boost_consumption = boost_consumption
        self.tick_skip = tick_skip

        match = Match(reward_function=reward_fn,
                      terminal_conditions=terminal_conditions,
                      obs_builder=obs_builder,
                      action_parser=action_parser,
                      state_setter=state_setter,
                      team_size=team_size,
                      spawn_opponents=spawn_opponents)

        self._match = match
        self._agents = []
        self._terminated = {}
        self._truncated = {}

        self._prev_state = None

        self._game = RocketSimGame(match,
                                   copy_gamestate=copy_gamestate_every_step,
                                   dodge_deadzone=dodge_deadzone,
                                   tick_skip=tick_skip)
        
        self.update_settings(gravity=gravity, boost_consumption=boost_consumption, tick_skip=tick_skip)

        self.observation_spaces = DictSpace({agent: match.observation_space for agent in self._agents})
        self.action_spaces = DictSpace({agent: match.action_space for agent in self._agents})


        
    @property
    def num_agents(self):
        return self._game.n_agents
    
    @property
    def agents(self):
        return [agent for agent in self._agents if not (self._terminated.get(agent, False) or self._truncated.get(agent, False))]

    @property
    def possible_agents(self):
        return list(self._agents)
    
    def observation_space(self, agent: str) -> Space:
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> Space:
        return self.action_spaces[agent]

    def reset(self, seed: int | None = None, return_info: bool = False, options: Dict[str, Any] | None = None) -> Tuple[List, Dict[str, Any]]:
        """
        The environment reset function. When called, this will reset the state
        of the environment and objects in the game. This should be called once
        when the environment is initialized, then every time the `done` flag
        from the `step()` function is `True`.
        
        :param seed: The seed to use for the environment. If `None`, the
         environment will use whatever seed it was last given. This must be
         called with a seed before the environment can be used.
        
        :param options: A dictionary of options to be passed to the game. Can
         be used to update gravity, boost consumption, or tick_skip. This
         feature is currently broken, however.
        
        :return: A tuple containing (obs, info)
        """
        
        if options is not None and \
            (options.get("gravity", self.gravity) != self.gravity or \
            options.get("boost_consumption", self.boost_consumption) != self.boost_consumption or \
            options.get("tick_skip", self.tick_skip) != self.tick_skip):
            self.update_settings(**options)

        state_str = self._match.get_reset_state()
        state = self._game.reset(state_str)

        self._match.episode_reset(state)
        self._prev_state = state
        
        self._terminated.clear()
        self._truncated.clear()

        obs = self._match.build_observations(state)
        if not return_info:
            return obs

        info = {
            'state': state,
            'result': self._match.get_result(state)
        }

        info = {agent: info for agent in self._agents}

        return obs, info

    def step(self, actions: Dict[str, Any]) -> Tuple[Dict[str, List], Dict[str, List], Dict[str, bool], Dict[str, bool], Dict[str, Dict[str, Any]]]:
        """
        The step function will send the dict of provided actions to the game,
        then advance the game forward by `tick_skip` physics ticks using that
        action. The game is then paused, and the current state is sent back to
        rlzoo_sim. This is decoded into a `GameState` object, which gets
        passed to the configuration objects to determine the rewards, next
        observation, terminated and truncated signals, and info dict.

        :param actions: An object containing actions, in the format specified
         by the `ActionParser`.

        :return: A tuple containing (obs, rewards, terminated, truncated, info), where each is a dict that is keyed by agent ID.
        """

        actions = np.array([actions[id] for id in self._agents])
        actions = self._match.format_actions(self._match.parse_actions(actions, self._prev_state))

        state = self._game.step(actions)
        result = self._match.get_result(state)
        self._prev_state = state

        terminated = self._match.is_terminated(state)
        truncated = False if terminated else self._match.is_truncated(state)

        obs = self._match.build_observations(state)
        reward = self._match.get_rewards(state, terminated)

        self._terminated = {agent: terminated for agent in self._agents}
        self._truncated = {agent: truncated for agent in self._agents}
        info = { 'state': state, 'result': result }
        info = {agent: info for agent in self._agents}

        return obs, reward, self._terminated, self._truncated, info

    def render(self):
        pass

    def close(self):
        pass

    def update_settings(self, gravity=None, boost_consumption=None, tick_skip=None, **kwargs):
        """
        Updates the specified RocketSim instance settings

        :param gravity: Scalar to be multiplied by in-game gravity. Default 1.

        :param boost_consumption: Scalar to be multiplied by default boost
        consumption rate. Default 1.

        :param tick_skip: Number of physics ticks the simulator will be
         advanced with the current controls before a `GameState` is returned at
         each call to `step()`.
        """

        self.gravity = gravity
        self.boost_consumption = boost_consumption
        self.tick_skip = tick_skip

        mutator_cfg = self._game.arena.get_mutator_config()
        if gravity is not None:
            mutator_cfg.gravity = rsim.Vec(0, 0, common_values.GRAVITY_Z*gravity)

        if boost_consumption is not None:
            mutator_cfg.boost_used_per_second = common_values.BOOST_CONSUMED_PER_SECOND*boost_consumption

        if tick_skip is not None:
            self._game.tick_skip = tick_skip
            self._match.tick_skip = tick_skip

        self._game.arena.set_mutator_config(mutator_cfg)
        self._agents = [f"{'blue' if car_id < 5 else 'orange'}_{car_id}" for car_id in self._game.players]
