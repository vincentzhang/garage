# flake8: noqa
import numpy as np
import tensorflow as tf

from garage.core import Serializable
from garage.misc import tabular
from garage.misc.overrides import overrides
from garage.tf.core import LayersPowered
import garage.tf.core.layers as L
from garage.tf.core.network import GRUNetwork
from garage.tf.distributions import RecurrentDiagonalGaussian
from garage.tf.misc import tensor_utils
from garage.tf.policies import StochasticPolicy
from garage.tf.spaces import Box


class GaussianGRUPolicy(StochasticPolicy, LayersPowered, Serializable):
    def __init__(
            self,
            env_spec,
            name="GaussianGRUPolicy",
            hidden_dim=32,
            feature_network=None,
            state_include_action=True,
            hidden_nonlinearity=tf.tanh,
            gru_layer_cls=L.GRULayer,
            learn_std=True,
            init_std=1.0,
            output_nonlinearity=None,
            std_share_network=False,
    ):
        """
        :param env_spec: A spec for the env.
        :param hidden_dim: dimension of hidden layer
        :param hidden_nonlinearity: nonlinearity used for each hidden layer
        :return:
        """
        assert isinstance(env_spec.action_space, Box)

        self._mean_network_name = "mean_network"
        self._std_network_name = "std_network"

        with tf.variable_scope(name, "GaussianGRUPolicy"):
            Serializable.quick_init(self, locals())
            super(GaussianGRUPolicy, self).__init__(env_spec)

            obs_dim = env_spec.observation_space.flat_dim
            action_dim = env_spec.action_space.flat_dim

            if state_include_action:
                input_dim = obs_dim + action_dim
            else:
                input_dim = obs_dim

            l_input = L.InputLayer(shape=(None, None, input_dim), name="input")

            if feature_network is None:
                feature_dim = input_dim
                l_flat_feature = None
                l_feature = l_input
            else:
                feature_dim = feature_network.output_layer.output_shape[-1]
                l_flat_feature = feature_network.output_layer
                l_feature = L.OpLayer(
                    l_flat_feature,
                    extras=[l_input],
                    name="reshape_feature",
                    op=lambda flat_feature, input: tf.reshape(
                        flat_feature,
                        tf.stack([
                            tf.shape(input)[0],
                            tf.shape(input)[1], feature_dim
                        ])),
                    shape_op=lambda _, input_shape: (
                        input_shape[0], input_shape[1], feature_dim))

            if std_share_network:
                mean_network = GRUNetwork(
                    input_shape=(feature_dim, ),
                    input_layer=l_feature,
                    output_dim=2 * action_dim,
                    hidden_dim=hidden_dim,
                    hidden_nonlinearity=hidden_nonlinearity,
                    output_nonlinearity=output_nonlinearity,
                    gru_layer_cls=gru_layer_cls,
                    name="gru_mean_network")

                l_mean = L.SliceLayer(
                    mean_network.output_layer,
                    slice(action_dim),
                    name="mean_slice")

                l_step_mean = L.SliceLayer(
                    mean_network.step_output_layer,
                    slice(action_dim),
                    name="step_mean_slice")

                l_log_std = L.SliceLayer(
                    mean_network.output_layer,
                    slice(action_dim, 2 * action_dim),
                    name="log_std_slice")

                l_step_log_std = L.SliceLayer(
                    mean_network.step_output_layer,
                    slice(action_dim, 2 * action_dim),
                    name="step_log_std_slice")
            else:
                mean_network = GRUNetwork(
                    input_shape=(feature_dim, ),
                    input_layer=l_feature,
                    output_dim=action_dim,
                    hidden_dim=hidden_dim,
                    hidden_nonlinearity=hidden_nonlinearity,
                    output_nonlinearity=output_nonlinearity,
                    gru_layer_cls=gru_layer_cls,
                    name="gru_mean_network")

                l_mean = mean_network.output_layer

                l_step_mean = mean_network.step_output_layer

                l_log_std = L.ParamLayer(
                    mean_network.input_layer,
                    num_units=action_dim,
                    param=tf.constant_initializer(np.log(init_std)),
                    name="output_log_std",
                    trainable=learn_std,
                )

                l_step_log_std = L.ParamLayer(
                    mean_network.step_input_layer,
                    num_units=action_dim,
                    param=l_log_std.param,
                    name="step_output_log_std",
                    trainable=learn_std,
                )

            self.mean_network = mean_network
            self.feature_network = feature_network
            self.l_input = l_input
            self.state_include_action = state_include_action

            flat_input_var = tf.placeholder(
                dtype=tf.float32, shape=(None, input_dim), name="flat_input")
            if feature_network is None:
                feature_var = flat_input_var
            else:
                feature_var = L.get_output(
                    l_flat_feature,
                    {feature_network.input_layer: flat_input_var})

            with tf.name_scope(self._mean_network_name):
                out_step_mean, out_step_hidden_mean = L.get_output(
                    [l_step_mean, mean_network.step_hidden_layer],
                    {mean_network.step_input_layer: feature_var})
                out_step_mean = tf.identity(out_step_mean, "step_mean")
                out_step_hidden_mean = tf.identity(out_step_hidden_mean,
                                                   "step_hidden_mean")

            with tf.name_scope(self._std_network_name):
                out_step_log_std = L.get_output(
                    l_step_log_std,
                    {mean_network.step_input_layer: feature_var})
                out_step_log_std = tf.identity(out_step_log_std,
                                               "step_log_std")

            self.f_step_mean_std = tensor_utils.compile_function([
                flat_input_var,
                mean_network.step_prev_state_layer.input_var,
            ], [out_step_mean, out_step_log_std, out_step_hidden_mean])

            self.l_mean = l_mean
            self.l_log_std = l_log_std

            self.input_dim = input_dim
            self.action_dim = action_dim
            self.hidden_dim = hidden_dim

            self.prev_actions = None
            self.prev_hiddens = None
            self.dist = RecurrentDiagonalGaussian(action_dim)
            self.name = name

            out_layers = [l_mean, l_log_std, l_step_log_std]
            if feature_network is not None:
                out_layers.append(feature_network.output_layer)

            LayersPowered.__init__(self, out_layers)

    @overrides
    def dist_info_sym(self, obs_var, state_info_vars, name=None):
        with tf.name_scope(name, "dist_info_sym", [obs_var, state_info_vars]):
            n_batches = tf.shape(obs_var)[0]
            n_steps = tf.shape(obs_var)[1]
            obs_var = tf.reshape(obs_var, tf.stack([n_batches, n_steps, -1]))
            if self.state_include_action:
                prev_action_var = state_info_vars["prev_action"]
                all_input_var = tf.concat(
                    axis=2, values=[obs_var, prev_action_var])
            else:
                all_input_var = obs_var
            if self.feature_network is None:
                with tf.name_scope(
                        self._mean_network_name, values=[all_input_var]):
                    means = L.get_output(self.mean_network.output_layer,
                                         {self.l_input: all_input_var})
                with tf.name_scope(
                        self._std_network_name, values=[all_input_var]):
                    log_stds = L.get_output(self.l_log_std,
                                            {self.l_input: all_input_var})
            else:
                flat_input_var = tf.reshape(all_input_var,
                                            (-1, self.input_dim))
                with tf.name_scope(
                        self._mean_network_name,
                        values=[all_input_var, flat_input_var]):
                    means = L.get_output(
                        self.mean_network.output_layer, {
                            self.l_input: all_input_var,
                            self.feature_network.input_layer: flat_input_var
                        })
                with tf.name_scope(
                        self._mean_network_name,
                        values=[all_input_var, flat_input_var]):
                    log_stds = L.get_output(
                        self.l_log_std, {
                            self.l_input: all_input_var,
                            self.feature_network.input_layer: flat_input_var
                        })
            return dict(mean=means, log_std=log_stds)

    @property
    def vectorized(self):
        return True

    def reset(self, dones=None):
        if dones is None:
            dones = [True]
        dones = np.asarray(dones)
        if self.prev_actions is None or len(dones) != len(self.prev_actions):
            self.prev_actions = np.zeros((len(dones),
                                          self.action_space.flat_dim))
            self.prev_hiddens = np.zeros((len(dones), self.hidden_dim))

        self.prev_actions[dones] = 0.
        self.prev_hiddens[dones] = self.mean_network.hid_init_param.eval()

    # The return value is a pair. The first item is a matrix (N, A), where each
    # entry corresponds to the action value taken. The second item is a vector
    # of length N, where each entry is the density value for that action, under
    # the current policy
    @overrides
    def get_action(self, observation):
        actions, agent_infos = self.get_actions([observation])
        return actions[0], {k: v[0] for k, v in agent_infos.items()}

    @overrides
    def get_actions(self, observations):
        flat_obs = self.observation_space.flatten_n(observations)
        if self.state_include_action:
            assert self.prev_actions is not None
            all_input = np.concatenate([flat_obs, self.prev_actions], axis=-1)
        else:
            all_input = flat_obs
        means, log_stds, hidden_vec = self.f_step_mean_std(
            all_input, self.prev_hiddens)
        rnd = np.random.normal(size=means.shape)
        actions = rnd * np.exp(log_stds) + means
        prev_actions = self.prev_actions
        self.prev_actions = self.action_space.flatten_n(actions)
        self.prev_hiddens = hidden_vec
        agent_info = dict(mean=means, log_std=log_stds)
        if self.state_include_action:
            agent_info["prev_action"] = np.copy(prev_actions)
        return actions, agent_info

    @property
    @overrides
    def recurrent(self):
        return True

    @property
    def distribution(self):
        return self.dist

    @property
    def state_info_specs(self):
        if self.state_include_action:
            return [
                ("prev_action", (self.action_dim, )),
            ]
        else:
            return []

    def log_diagnostics(self, paths):
        log_stds = np.vstack(
            [path["agent_infos"]["log_std"] for path in paths])
        tabular.record_tabular('AveragePolicyStd', np.mean(np.exp(log_stds)))
