"""Discrete MLP QFunction."""
import tensorflow as tf

from garage.misc.overrides import overrides
from garage.tf.core.mlp import mlp
from garage.tf.q_functions import QFunction


class DiscreteMLPQFunction(QFunction):
    """
    Discrete MLP Function class.

    This class implements a Q-value network. It predicts Q-value based on the
    input state and action. It uses an MLP to fit the function Q(s, a).

    Args:
        env_spec: environment specification
        hidden_sizes: A list of numbers of hidden units
            for all hidden layers.
        hidden_nonlinearity: An activation shared by all fc layers.
        output_nonlinearity: An activation used by the output layer.
        layer_norm: A bool to indicate whether to perform
            layer normalization or not.
    """

    def __init__(self,
                 env_spec,
                 hidden_sizes=(32, 32),
                 hidden_nonlinearity=tf.nn.relu,
                 output_nonlinearity=None,
                 layer_norm=False):
        super(DiscreteMLPQFunction, self).__init__()

        self._action_dim = env_spec.action_space.n
        self._hidden_sizes = hidden_sizes
        self._hidden_nonlinearity = hidden_nonlinearity
        self._output_nonlinearity = output_nonlinearity
        self._layer_norm = layer_norm

    @overrides
    def build_net(self, name, input, dueling=False, layer_norm=False):
        """
        Build the q-network.

        Args:
            name: scope of the network.
            input: input Tensor of the network.
            dueling: use dueling network or not.
            layer_norm: Boolean for layer normalization.

        Return:
            The tf.Tensor of Discrete DiscreteMLPQFunction.
        """
        if isinstance(self._hidden_sizes, int):  # single layer
            return super().q_func(
                input_network=input,
                output_dim=self._action_dim,
                hidden_sizes=[self._hidden_sizes],
                name=name,
                hidden_nonlinearity=self._hidden_nonlinearity,
                output_nonlinearity=self._output_nonlinearity,
                layer_normalization=self._layer_norm,
                dueling=dueling)

        network = mlp(
            input_var=input,
            output_dim=self._hidden_sizes[-2],
            hidden_sizes=self._hidden_sizes[:-2],
            name=name,
            hidden_nonlinearity=self._hidden_nonlinearity,
            output_nonlinearity=self._hidden_nonlinearity,
            layer_normalization=self._layer_norm)

        return super().q_func(
            input_network=network,
            output_dim=self._action_dim,
            hidden_sizes=[self._hidden_sizes[-1]],
            name=name,
            hidden_nonlinearity=self._hidden_nonlinearity,
            output_nonlinearity=self._output_nonlinearity,
            layer_normalization=self._layer_norm,
            dueling=dueling)
