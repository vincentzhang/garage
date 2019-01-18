# flake8: noqa
import math

import numpy as np

from garage.core import Serializable
from garage.envs import Step
from garage.envs.mujoco import MujocoEnv
from garage.envs.mujoco.mujoco_env import q_inv, q_mult
from garage.misc import tabular
from garage.misc.overrides import overrides


class AntEnv(MujocoEnv, Serializable):

    FILE = 'ant.xml'
    ORI_IND = 3

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Always call Serializable constructor last
        Serializable.quick_init(self, locals())

    def get_current_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
            np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
            self.get_body_xmat("torso").flat,
            self.get_body_com("torso"),
        ]).reshape(-1)

    def step(self, action):
        self.forward_dynamics(action)
        comvel = self.get_body_comvel("torso")
        forward_reward = comvel[0]
        lb, ub = self.action_bounds
        scaling = (ub - lb) * 0.5
        ctrl_cost = 0.5 * 1e-2 * np.sum(np.square(action / scaling))
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1))),
        survive_reward = 0.05
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self._state
        notdone = np.isfinite(state).all() \
            and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self.get_current_obs()
        return Step(ob, float(reward), done)

    @overrides
    def get_ori(self):
        ori = [0, 1, 0, 0]
        rot = self.sim.data.qpos[self.__class__.ORI_IND:self.__class__.ORI_IND
                                 + 4]  # take the quaternion
        ori = q_mult(q_mult(rot, ori),
                     q_inv(rot))[1:3]  # project onto x-y plane
        ori = math.atan2(ori[1], ori[0])
        return ori

    @overrides
    def log_diagnostics(self, paths):
        progs = [
            path["observations"][-1][-3] - path["observations"][0][-3]
            for path in paths
        ]
        tabular.record_tabular('AverageForwardProgress', np.mean(progs))
        tabular.record_tabular('MaxForwardProgress', np.max(progs))
        tabular.record_tabular('MinForwardProgress', np.min(progs))
        tabular.record_tabular('StdForwardProgress', np.std(progs))
