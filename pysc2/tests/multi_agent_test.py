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
        min_distance_mineral, _ = self.get_closet_farthest_minerals(marines[0], minerals)
        _, max_distance_mineral = self.get_closet_farthest_minerals(marines[1], minerals)

        marine0_actions = self.move_marines_to_minarel(marines[0], min_distance_mineral)
        marine1_actions = self.move_marines_to_minarel(marines[1], max_distance_mineral)

        return marine0_actions + marine1_actions

    def move_marines_to_minarel(self, marine, mineral):
        select_action = FUNCTIONS.select_point("select", (marine.x, marine.y))
        spatial_action = FUNCTIONS.Move_screen("now", (mineral.x, mineral.y))
        return [select_action, spatial_action]

    def get_closet_farthest_minerals(self, marine, minerals):
        min_distance = (2 << 16 - 1)
        max_distance = 0
        min_distance_mineral = minerals[0]
        max_distance_mineral = minerals[0]
        for m in minerals:
            distance = math.sqrt(math.pow(m.x - marine.x, 2)
                                 + math.pow(m.y - marine.y, 2))
            if distance < min_distance:
                min_distance = distance
                min_distance_mineral = m

            if distance > max_distance:
                max_distance = distance
                max_distance_mineral = m
        return (min_distance_mineral, max_distance_mineral)


class TestMultiAgent(utils.TestCase):

    def test_multi_agent(self):
        with sc2_env.SC2Env(
                map_name="CollectMineralShards",
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

            for _ in range(10000):
                raw_obs = multiplayer_obs[0]
                obs = raw_obs.observation
                act = agent.step(obs)
                multiplayer_act = (act,)
                multiplayer_obs = env.step(multiplayer_act)


if __name__ == "__main__":
    absltest.main()
