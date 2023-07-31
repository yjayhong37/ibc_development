import tensorflow as tf
import functools
import numpy as np

from tf_agents.agents import tf_agent
from tf_agents.networks import nest_map, network
from tf_agents.policies import greedy_policy, tf_policy
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import policy_step
from tf_agents.utils import common, nest_utils
import tensorflow_probability as tfp

import mcmc


class ImplicitBCAgent(tf_agent.TFAgent):
    """TFAgent, implementing implicit behavioral cloning."""

    def __init__(self, time_step_spec, action_spec, action_sampling_spec,
                 cloning_network, obs_norm_layer=None, act_norm_layer=None, act_denorm_layer=None,
                 num_counter_examples=8, train_step_counter=None, name=None, learning_rate=1e-3):  # 256
        # tf.Module dependency allows us to capture checkpoints and saved models with the agent.
        tf.Module.__init__(self, name=name)

        self._action_sampling_spec = action_sampling_spec
        self._obs_norm_layer = obs_norm_layer
        self._act_norm_layer = act_norm_layer
        self._act_denorm_layer = act_denorm_layer
        self.cloning_network = cloning_network
        self.cloning_network.create_variables(training=False)

        learning_rate_schedule = (tf.keras.optimizers.schedules.ExponentialDecay(learning_rate, decay_steps=100,
                                                                                 decay_rate=0.99))
        self._optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_schedule)

        self._num_counter_examples = num_counter_examples
        self._fraction_langevin_samples = 1.0

        self.ebm_loss_type = 'info_nce'
        self._kl = tf.keras.losses.KLDivergence(reduction=tf.keras.losses.Reduction.NONE)

        self._mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)

        # Collect policy would normally be used for data collection. In a BCAgent
        # we don't expect to use it, unless we want to upgrade this to a DAGGER like
        # setup.
        collect_policy = IbcPolicy(time_step_spec=time_step_spec, action_spec=action_spec,
                                   action_sampling_spec=action_sampling_spec, actor_network=cloning_network,
                                   obs_norm_layer=self._obs_norm_layer, act_denorm_layer=self._act_denorm_layer)

        policy = greedy_policy.GreedyPolicy(collect_policy)

        super(ImplicitBCAgent, self).__init__(time_step_spec, action_spec, policy, collect_policy,
                                              train_sequence_length=None, debug_summaries=False,
                                              summarize_grads_and_vars=False, train_step_counter=train_step_counter)

    def _train(self, experience, weights=None):
        """Trains energy network & returns loss."""
        variables_to_train = self.cloning_network.trainable_weights
        assert list(variables_to_train), "No variables in the agent's network."

        loss_info, tape = self._loss(experience, variables_to_train, weights=weights, training=True)

        tf.debugging.check_numerics(loss_info.loss, "Loss is inf or nan")

        grads = tape.gradient(loss_info.loss, variables_to_train)
        grads_and_vars = list(zip(grads, variables_to_train))

        self._optimizer.apply_gradients(grads_and_vars)
        self.train_step_counter.assign_add(1)

        return loss_info

    def get_eval_loss(self, experience):
        """Compute loss without training the energy network."""
        loss_dict = self._loss(experience, training=False)
        return loss_dict

    def _loss(self, experience, variables_to_train=None, weights=None, training=False):
        """Sample counter examples for given observation/action pairs and compute loss."""
        # ** Note **: Obs spec includes time dim. but highlighted here since we have to deal with it.
        # Observation: [B x T x obs_spec]
        # Action:      [B x act_spec]
        observations, actions = experience

        # Use first observation to figure out batch/time sizes as they should be the same across all observations.
        single_obs = tf.nest.flatten(observations)[0]
        batch_size = tf.shape(single_obs)[0]

        # Now tile and setup observations to be: [B * n+1 x obs_spec]
        maybe_tiled_obs = nest_utils.tile_batch(observations, self._num_counter_examples + 1)

        # Reshape actions to be: output_shape = [B x 1 x act_spec]
        expanded_actions = tf.nest.map_structure(functools.partial(tf.expand_dims, axis=1), actions)

        with tf.GradientTape(watch_accessed_variables=False) as tape:
            if variables_to_train:
                tape.watch(variables_to_train)

            with tf.name_scope('loss'):
                # generate counter examples
                counter_example_actions, combined_true_counter_actions, chain_data = (
                    self._make_counter_example_actions(observations, expanded_actions, batch_size))
                actions_size_n = tf.broadcast_to(
                    expanded_actions, (batch_size, self._num_counter_examples, expanded_actions.shape[-1]))

                # TODO: Do we use this, need this?
                mse_counter_examples = self._mse(counter_example_actions, actions_size_n)
                mse_counter_examples = common.aggregate_losses(per_example_loss=mse_counter_examples).total_loss

                # Combine observation with actions, both demonstrated & counter examples
                network_inputs = (maybe_tiled_obs, tf.stop_gradient(combined_true_counter_actions))

                # Forward pass through the energy network, output_shape = [B * n+1]
                predictions, _ = self.cloning_network(network_inputs, training=training)

                # output_shape = [B, n+1]
                predictions = tf.reshape(predictions, [batch_size, self._num_counter_examples + 1])

                # Compute InfoNCE loss
                per_example_loss, debug_dict = self._compute_ebm_loss(batch_size, predictions)

                # Add gradient loss to stabilize training
                grad_loss = grad_penalty(self.cloning_network, batch_size, chain_data, maybe_tiled_obs,
                                         combined_true_counter_actions, training)
                per_example_loss += grad_loss

                # TODO: Do I need this? -> TPU
                # Aggregate losses uses some TF magic to make sure aggregation across TPU replicas does the right
                # thing. It does mean we have to calculate per_example_losses though.
                agg_loss = common.aggregate_losses(per_example_loss=per_example_loss, sample_weight=weights,
                                                   regularization_loss=self.cloning_network.losses)
                total_loss = agg_loss.total_loss

                losses_dict = {'ebm_total_loss': total_loss}

                losses_dict.update(debug_dict)
                if grad_loss is not None:
                    losses_dict['grad_loss'] = tf.reduce_mean(grad_loss)

                losses_dict['mse_counter_examples'] = tf.reduce_mean(mse_counter_examples)

                opt_dict = dict()
                if chain_data is not None and chain_data.energies is not None:
                    energies = chain_data.energies
                    opt_dict['overall_energies_avg'] = tf.reduce_mean(energies)
                    first_energies = energies[0]
                    opt_dict['first_energies_avg'] = tf.reduce_mean(first_energies)
                    final_energies = energies[-1]
                    opt_dict['final_energies_avg'] = tf.reduce_mean(final_energies)

                if chain_data is not None and chain_data.grad_norms is not None:
                    grad_norms = chain_data.grad_norms
                    opt_dict['overall_grad_norms_avg'] = tf.reduce_mean(grad_norms)
                    first_grad_norms = grad_norms[0]
                    opt_dict['first_grad_norms_avg'] = tf.reduce_mean(first_grad_norms)
                    final_grad_norms = grad_norms[-1]
                    opt_dict['final_grad_norms_avg'] = tf.reduce_mean(final_grad_norms)

                losses_dict.update(opt_dict)

                common.summarize_scalar_dict(losses_dict, step=self.train_step_counter, name_scope='Losses/')

                # This is a bit of a hack, but it makes it so can compute eval loss, including with various metrics.
                if training:
                    return tf_agent.LossInfo(total_loss, ()), tape
                else:
                    return losses_dict

    def _compute_ebm_loss(self, batch_size, predictions):
        """Compute InfoNCE loss."""
        softmaxed_predictions = tf.nn.softmax(predictions / 1.0, axis=-1)

        # [B x n+1] with 1 in column [:, -1]
        indices = tf.ones((batch_size,), dtype=tf.int32) * self._num_counter_examples
        labels = tf.one_hot(indices, depth=self._num_counter_examples + 1)

        per_example_loss = self._kl(labels, softmaxed_predictions)

        return per_example_loss, dict()

    def _make_counter_example_actions(self, observations,  expanded_actions, batch_size):
        """Given observations (B x obs_spec) and true actions (B x 1 x act_spec), create counter example actions
        (uniformly sampled with MCMC optimization)."""
        # Note that T (time dimension) would be included in obs_spec.

        # Sample counter example actions from uniform distribution, output_shape: [B x num_counter_examples x act_spec]
        random_uniform_example_actions = tensor_spec.sample_spec_nest(self._action_sampling_spec,
                                                                      outer_dims=(batch_size,
                                                                                  self._num_counter_examples))

        # Reshape to put B and num counter examples on same tensor dimension [B*num_counter_examples x act_spec]
        random_uniform_example_actions = tf.reshape(random_uniform_example_actions,
                                                    (batch_size * self._num_counter_examples, -1))

        maybe_tiled_obs_n = nest_utils.tile_batch(observations, self._num_counter_examples)

        # Optimize uniform counter example actions with MCMC
        langevin_return = mcmc.langevin_actions_given_obs(self.cloning_network, maybe_tiled_obs_n,
                                                          random_uniform_example_actions, policy_state=(),
                                                          min_actions=self._action_sampling_spec.minimum,
                                                          max_actions=self._action_sampling_spec.maximum,
                                                          training=False, tfa_step_type=(), return_chain=True)

        lang_opt_counter_example_actions, chain_data = langevin_return

        counter_example_actions = tf.concat(lang_opt_counter_example_actions, axis=1)
        counter_example_actions = tf.reshape(counter_example_actions, (batch_size, self._num_counter_examples, -1))

        def concat_and_squash_actions(counter_example, action):
            return tf.reshape(tf.concat([counter_example, action], axis=1), [-1] + self._action_spec.shape.as_list())

        # Batch consists of num_counter_example rows followed by 1 true action.
        # [B * (n + 1) x act_spec]
        combined_true_counter_actions = tf.nest.map_structure(concat_and_squash_actions, counter_example_actions,
                                                              expanded_actions)

        return counter_example_actions, combined_true_counter_actions, chain_data


class IbcPolicy(tf_policy.TFPolicy):
    """Class to build Actor Policies."""

    def __init__(self, time_step_spec, action_spec, action_sampling_spec, actor_network, policy_state_spec=(),
                 num_action_samples=512, obs_norm_layer=None, act_denorm_layer=None):

        if isinstance(actor_network, network.Network):
            # To work around create_variables we force stuff to be build beforehand.
            # TODO(oars): Generalize networks.create_variables
            assert actor_network.built

            if not policy_state_spec:
                policy_state_spec = actor_network.state_spec

        self._action_sampling_spec = action_sampling_spec

        self._action_sampling_minimum = tf.Variable(self._action_sampling_spec.minimum, trainable=False,
                                                    name='sampling/minimum')
        self._action_sampling_maximum = tf.Variable(self._action_sampling_spec.maximum, trainable=False,
                                                    name='sampling/maximum')

        self._num_action_samples = num_action_samples
        self._obs_norm_layer = obs_norm_layer
        self._act_denorm_layer = act_denorm_layer

        self._actor_network = actor_network

        super(IbcPolicy, self).__init__(time_step_spec=time_step_spec, action_spec=action_spec,
                                        policy_state_spec=policy_state_spec, info_spec=(), clip=True, name=None)

    def _variables(self):
        return self._actor_network.variables + [self._action_sampling_minimum, self._action_sampling_maximum]

    def _distribution(self, time_step, policy_state):
        # Use first observation to figure out batch/time sizes as they should be the
        # same across all observations.
        observations = time_step.observation
        if isinstance(observations, dict) and 'rgb' in observations:
            observations['rgb'] = tf.image.convert_image_dtype(observations['rgb'], dtype=tf.float32)

        if self._obs_norm_layer is not None:
            observations = self._obs_norm_layer(observations)
            if isinstance(self._obs_norm_layer, nest_map.NestMap):
                observations, _ = observations

        single_obs = tf.nest.flatten(observations)[0]
        batch_size = tf.shape(single_obs)[0]

        maybe_tiled_obs = nest_utils.tile_batch(observations, self._num_action_samples)

        # Initialize sample actions with uniform distribution
        action_samples = tensor_spec.sample_spec_nest(self._action_sampling_spec,
                                                      outer_dims=(batch_size * self._num_action_samples,))

        # Optimize action samples with MCMC
        action_samples = mcmc.langevin_actions_given_obs(self._actor_network, maybe_tiled_obs, action_samples,
                                                         policy_state=policy_state,
                                                         min_actions=self._action_sampling_spec.minimum,
                                                         max_actions=self._action_sampling_spec.maximum,
                                                         training=False, tfa_step_type=time_step.step_type)

        probs = mcmc.get_probabilities(self._actor_network, batch_size, self._num_action_samples,
                                       maybe_tiled_obs, action_samples, training=False)

        if self._act_denorm_layer is not None:
            action_samples = self._act_denorm_layer(action_samples)
            if isinstance(self._act_denorm_layer, nest_map.NestMap):
                action_samples, _ = action_samples

        # Make a distribution for sampling.
        distribution = MappedCategorical(probs=probs, mapped_values=action_samples)
        return policy_step.PolicyStep(distribution, policy_state)


@tfp.experimental.register_composite
class MappedCategorical(tfp.distributions.Categorical):
    """Categorical distribution that maps classes to specific values."""

    def __init__(self, logits=None, probs=None, mapped_values=None, dtype=tf.int32, validate_args=False,
                 allow_nan_stats=True, name='MappedCategorical'):
        """Initialize Categorical distributions using class log-probabilities.

        Args:
          logits: An N-D `Tensor`, `N >= 1`, representing the log probabilities of a
            set of Categorical distributions. The first `N - 1` dimensions index
            into a batch of independent distributions and the last dimension
            represents a vector of logits for each class. Only one of `logits` or
            `probs` should be passed in.
          probs: An N-D `Tensor`, `N >= 1`, representing the probabilities of a set
            of Categorical distributions. The first `N - 1` dimensions index into a
            batch of independent distributions and the last dimension represents a
            vector of probabilities for each class. Only one of `logits` or `probs`
            should be passed in.
          mapped_values: Values that map to each category.
          dtype: The type of the event samples (default: int32).
          validate_args: Python `bool`, default `False`. When `True` distribution
            parameters are checked for validity despite possibly degrading runtime
            performance. When `False` invalid inputs may silently render incorrect
            outputs.
          allow_nan_stats: Python `bool`, default `True`. When `True`, statistics
            (e.g., mean, mode, variance) use the value "`NaN`" to indicate the
            result is undefined. When `False`, an exception is raised if one or more
            of the statistic's batch members are undefined.
          name: Python `str` name prefixed to Ops created by this class.
        """
        self._mapped_values = mapped_values
        super(MappedCategorical, self).__init__(logits=logits, probs=probs, dtype=dtype, validate_args=validate_args,
                                                allow_nan_stats=allow_nan_stats, name=name)

    def mode(self, name='mode'):
        """Mode of the distribution."""
        mode = super(MappedCategorical, self).mode(name)
        return tf.gather(self._mapped_values, [mode], batch_dims=0)

    def sample(self, sample_shape=(), seed=None, name='sample', **kwargs):
        """Generate samples of the specified shape."""
        # TODO(oars): Fix for complex sample_shapes
        sample = super(MappedCategorical, self).sample(
            sample_shape=sample_shape, seed=seed, name=name, **kwargs)
        return tf.gather(self._mapped_values, [sample], batch_dims=0)

    @classmethod
    def _parameter_properties(cls, dtype, num_classes=None):
        return tfp.distributions.Categorical._parameter_properties(dtype=dtype, num_classes=num_classes)


def grad_penalty(energy_network, batch_size, chain_data, observations, combined_true_counter_actions, training,
                 only_apply_final_grad_penalty=True, grad_margin=1.0, square_grad_penalty=True, grad_loss_weight=1.0):
    """Calculate losses based on some norm of dE/dactions from mcmc samples."""
    # Case 1: only add a gradient penalty on the final step.
    if only_apply_final_grad_penalty:
        de_dact, _ = mcmc.gradient_wrt_act(energy_network, observations,
                                           tf.stop_gradient(combined_true_counter_actions), training,
                                           network_state=(), tfa_step_type=())

        # grad norms should now be shape (b*(n+1))
        grad_norms = tf.linalg.norm(de_dact, axis=1, ord=np.inf)  # mcmc.compute_grad_norm(grad_norm_type, de_dact)
        grad_norms = tf.reshape(grad_norms, (batch_size, -1))

    else:
        # Case 2: the full chain was under the gradient tape, or langevin_step
        # stop_chain_grad was set to True. Either way just go add penalties to all
        # the norms.
        assert chain_data.grad_norms is not None
        # grad_norms starts out as: (num_iterations, B*n)
        grad_norms = chain_data.grad_norms
        # now grad_norms is shape: (B*n, num_iterations)
        grad_norms = tf.transpose(grad_norms, perm=[1, 0])
        # now grad_norms is shape: (B, n*num_iterations)
        grad_norms = tf.reshape(grad_norms, (batch_size, -1))

    if grad_margin is not None:
        grad_norms -= grad_margin
        # assume 1e10 is big enough
        grad_norms = tf.clip_by_value(grad_norms, 0., 1e10)

    if square_grad_penalty:
        grad_norms = grad_norms ** 2

    grad_loss = tf.reduce_mean(grad_norms, axis=1)
    return grad_loss * grad_loss_weight


def generate_registration_functions(policy, policy_network, strategy):
    """Generates a tf.function and a concrete function matching policy calls."""
    batched_network_input_spec = tensor_spec.add_outer_dims_nest(
        (policy.time_step_spec.observation, policy.action_spec), outer_dims=(None,))
    batched_step_type_spec = tensor_spec.add_outer_dims_nest(
        policy.time_step_spec.step_type, outer_dims=(None,))
    batched_policy_state_spec = tensor_spec.add_outer_dims_nest(
        policy.policy_state_spec, outer_dims=(None,))

    @tf.function
    def _create_variables(specs, training, step_type, network_state):
        return strategy.run(policy_network, args=(specs,), kwargs={'step_type': step_type,
                                                                   'network_state': network_state,
                                                                   'training': training})

    # Called for the side effect of tracing the function so that it is captured by
    # the saved model.
    _create_variables.get_concrete_function(batched_network_input_spec, step_type=batched_step_type_spec,
                                            network_state=batched_policy_state_spec,
                                            training=tensor_spec.TensorSpec(shape=(), dtype=tf.bool))

    return _create_variables
