import tensorflow as tf 
from tf_agents.networks import network


class ResNetPreActivationLayer(tf.keras.layers.Layer):
    """ResNet layer, improved 'pre-activation' version."""
    def __init__(self, hidden_sizes, rate, kernel_initializer, bias_initializer, **kwargs):

        super(ResNetPreActivationLayer, self).__init__(**kwargs)

        # ResNet wants layers to be even numbers, but remember there will be an additional
        # layer just to project to the first hidden size.
        assert len(hidden_sizes) % 2 == 0
        self._projection_layer = tf.keras.layers.Dense(hidden_sizes[0], activation=None,
                                                       kernel_initializer=kernel_initializer,
                                                       bias_initializer=bias_initializer)
        self._weight_layers = []
        self._norm_layers = []
        self._activation_layers = []
        self._dropouts = []

        self._weight_layers_2 = []
        self._norm_layers_2 = []
        self._activation_layers_2 = []
        self._dropouts_2 = []

        def create_dense_layer(width):
            return tf.keras.layers.Dense(width, activation=None, kernel_initializer=kernel_initializer,
                                         bias_initializer=bias_initializer)

        # Step every other
        for layer in range(0, len(hidden_sizes), 2):
            self._weight_layers.append(create_dense_layer(hidden_sizes[layer]))
            self._activation_layers.append(tf.keras.layers.ReLU())
            self._dropouts.append(tf.keras.layers.Dropout(rate))

            self._weight_layers_2.append(create_dense_layer(hidden_sizes[layer+1]))
            self._activation_layers_2.append(tf.keras.layers.ReLU())
            self._dropouts_2.append(tf.keras.layers.Dropout(rate))

    def call(self, x, training):
        x = self._projection_layer(x)

        # Do forward pass through resnet layers.
        for layer in range(len(self._weight_layers)):
            x_start_block = tf.identity(x)
            x = self._activation_layers[layer](x, training=training)
            x = self._dropouts[layer](x, training=training)
            x = self._weight_layers[layer](x, training=training)

            x = self._activation_layers_2[layer](x, training=training)
            x = self._dropouts_2[layer](x, training=training)
            x = self._weight_layers_2[layer](x, training=training)
            x = x_start_block + x
        return x


class MLPEBM(network.Network):
    """MLP with ResNetPreActivation layers and Dense layer."""
    def __init__(self, obs_spec, action_spec, depth=2, width=256, name='MLPEBM'):
        super(MLPEBM, self).__init__(input_tensor_spec=obs_spec, state_spec=(), name=name)

        # Define MLP.
        hidden_sizes = [width for _ in range(depth)]
        self._mlp = ResNetPreActivationLayer(hidden_sizes, 0.0, 'normal', 'normal')

        # Define projection to energy.
        self._project_energy = tf.keras.layers.Dense(action_spec.shape[-1], kernel_initializer='normal',
                                                     bias_initializer='normal')

    def call(self, inputs, training, step_type=(), network_state=()):
        # obs: dict of named obs_spec.
        # act:   [B x act_spec]
        obs, act = inputs

        # Combine dict of observations to concatenated tensor. [B x T x obs_spec]
        obs = tf.concat(tf.nest.flatten(obs), axis=-1)

        # Flatten obs across time: [B x T * obs_spec]
        batch_size = tf.shape(obs)[0]
        obs = tf.reshape(obs, [batch_size, -1])

        # Concat [obs, act].
        x = tf.concat([obs, act], -1)

        # Forward mlp.
        x = self._mlp(x, training=training)

        # Project to energy.
        x = self._project_energy(x, training=training)

        # Squeeze extra dim.
        x = tf.squeeze(x, axis=-1)

        return x, network_state


def get_energy_model(obs_tensor_spec, action_tensor_spec, network_width):
    """Create MLP."""
    energy_model = MLPEBM(obs_spec=(obs_tensor_spec, action_tensor_spec), action_spec=tf.TensorSpec([1]),
                          width=network_width)
    return energy_model
