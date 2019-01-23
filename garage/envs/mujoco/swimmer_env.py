import numpy as np

from garage.core import Serializable
from garage.envs import Step
from garage.envs.mujoco import MujocoEnv
from garage.logger import tabular
from garage.misc import autoargs
from garage.misc.overrides import overrides


class SwimmerEnv(MujocoEnv, Serializable):

    FILE = 'swimmer.xml'
    ORI_IND = 2

    @autoargs.arg(
        'ctrl_cost_coeff', type=float, help='cost coefficient for controls')
    def __init__(self, ctrl_cost_coeff=1e-2, *args, **kwargs):
        self.ctrl_cost_coeff = ctrl_cost_coeff
        super().__init__(*args, **kwargs)

        # Always call Serializable constructor last
        Serializable.quick_init(self, locals())

    def get_current_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
            self.get_body_com("torso").flat,
        ]).reshape(-1)

    def get_ori(self):
        return self.sim.data.qpos[self.__class__.ORI_IND]

    def step(self, action):
        self.forward_dynamics(action)
        next_obs = self.get_current_obs()
        lb, ub = self.action_bounds
        scaling = (ub - lb) * 0.5
        ctrl_cost = 0.5 * self.ctrl_cost_coeff * np.sum(
            np.square(action / scaling))
        forward_reward = self.get_body_comvel("torso")[0]
        reward = forward_reward - ctrl_cost
        done = False
        return Step(next_obs, reward, done)

    @overrides
    def log_diagnostics(self, paths):
        if paths:
            progs = [
                path["observations"][-1][-3] - path["observations"][0][-3]
                for path in paths
            ]
            tabular.record('AverageForwardProgress', np.mean(progs))
            tabular.record('MaxForwardProgress', np.max(progs))
            tabular.record('MinForwardProgress', np.min(progs))
            tabular.record('StdForwardProgress', np.std(progs))
        else:
            tabular.record('AverageForwardProgress', np.nan)
            tabular.record('MaxForwardProgress', np.nan)
            tabular.record('MinForwardProgress', np.nan)
            tabular.record('StdForwardProgress', np.nan)
