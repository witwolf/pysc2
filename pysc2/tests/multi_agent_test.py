from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from future.builtins import range  # pylint: disable=redefined-builtin
import six

from pysc2.agents import random_agent
from pysc2.env import sc2_env
from pysc2.tests import utils
from s2clientprotocol import raw_pb2 as sc_raw
from s2clientprotocol import sc2api_pb2 as sc_pb
from s2clientprotocol import common_pb2 as sc_common
from pysc2.lib.actions import FUNCTIONS, FunctionCall
from pysc2.lib import actions, features, units
import math
import numpy as np

UNIT_MINERAL = 1680


class MultiAgent():

    def __init__(self, *args, **kwargs):
        pass

    def step(self, obs):
        marines = []
        minerals = []
        for unit in obs.feature_units:
            if unit.unit_type == units.Terran.Marine:
                marines.append(unit)
            elif unit.unit_type == UNIT_MINERAL:
                minerals.append(unit)
        return self.get_actions(marines, minerals)

    def get_actions(self, marines, minerals):
        groups = self._group_by_distance(marines, minerals)
        actions = []
        for marine, mineral_group in zip(marines, groups):
            if not mineral_group:
                continue
            distance = self._get_distances(marine, mineral_group)
            idx = np.argmin(distance)
            action = self.move_marine_to_minarel(marine, mineral_group[idx])
            actions.extend(action)

        return actions

    def move_marine_to_minarel(self, marine, mineral):
        select_action = FUNCTIONS.select_point("select", (marine.x, marine.y))
        spatial_action = FUNCTIONS.Move_screen("now", (mineral.x, mineral.y))
        return [select_action, spatial_action]

    def _group_by_distance(self, marines, minerals):
        groups = [[]] * len(marines)
        distances = np.ndarray(shape=[len(minerals), len(marines)], dtype=np.float32)
        for i, mineral in enumerate(minerals):
            distances[i] = self._get_distances(mineral, marines)
        min_distance_idxs = np.argmin(distances, axis=1)
        for i, idx in enumerate(min_distance_idxs):
            groups[idx].append(minerals[i])
        return groups

    def _get_distances(self, unit, units):
        distances = np.ndarray(shape=[len(units)], dtype=np.float32)
        for i, m in enumerate(units):
            distances[i] = self._get_distance(unit, m)
        return distances

    def _get_distance(self, unit0, unit1):
        return math.sqrt(math.pow(unit1.x - unit0.x, 2)
                         + math.pow(unit1.y - unit0.y, 2))


class TestMultiAgent(utils.TestCase):

    def test_multi_agent(self):
        with sc2_env.SC2Env(
                map_name="CollectMineralShards",
                # step_mul=8,
                agent_interface_format=sc2_env.AgentInterfaceFormat(
                    use_unit_counts=True,
                    use_feature_units=True,
                    feature_dimensions=sc2_env.Dimensions(
                        screen=(64, 64),
                        minimap=(64, 64)))) as env:
            multiplayer_obs_spec = env.observation_spec()
            obs_spec = multiplayer_obs_spec[0]
            multiplayer_action_spec = env.action_spec()
            action_spec = multiplayer_action_spec[0]

            agent = MultiAgent(obs_spec, action_spec)
            multiplayer_obs = env.reset()
            episode = 0
            for _ in range(10000):
                raw_obs = multiplayer_obs[0]
                obs = raw_obs.observation
                act = agent.step(obs)
                multiplayer_act = (act,)
                multiplayer_obs = env.step(multiplayer_act)
                if raw_obs.last():
                    print('Episode:%d, score:%d' % (episode, obs['score_cumulative'][0]))
                    episode += 1


if __name__ == "__main__":
    absltest.main()
