import tensorflow as tf
import numpy as np
import collections


class PolynomialSchedule:
    """Polynomial learning rate schedule for Langevin sampler."""

    def __init__(self, init, final, power, num_steps):
        self._init = init
        self._final = final
        self._power = power
        self._num_steps = num_steps

    def get_rate(self, index):
        """Get learning rate for index."""
        return ((self._init - self._final) *
                ((1 - (float(index) / float(self._num_steps - 1))) ** (self._power))) + self._final


def update_chain_data(num_iterations, step_index, actions, energies, grad_norms, full_chain_actions,
                      full_chain_energies, full_chain_grad_norms):
    """Helper function to keep track of data during the mcmc."""
    # I really wish tensorflow made assignment-by-index easy.
    # Then this function could just be:
    # full_chain_actions[step_index] = actions
    # full_chain_energies[step_index] = energies
    # full_chain_grad_norms[step_index] = grad_norms

    iter_onehot = tf.one_hot(step_index, num_iterations)[Ellipsis, None]
    iter_onehot = tf.broadcast_to(iter_onehot, tf.shape(full_chain_energies))

    new_energies = energies * iter_onehot
    full_chain_energies += new_energies

    new_grad_norms = grad_norms * iter_onehot
    full_chain_grad_norms += new_grad_norms

    iter_onehot = iter_onehot[Ellipsis, None]
    iter_onehot = tf.broadcast_to(iter_onehot, tf.shape(full_chain_actions))
    actions_expanded = actions[None, Ellipsis]
    actions_expanded = tf.broadcast_to(actions_expanded, tf.shape(iter_onehot))
    new_actions_expanded = actions_expanded * iter_onehot
    full_chain_actions += new_actions_expanded
    return full_chain_actions, full_chain_energies, full_chain_grad_norms


@tf.function
def gradient_wrt_act(energy_network, observations, actions, training, network_state, tfa_step_type):
    """Compute dE(obs,act)/dact, also return energy."""

    with tf.GradientTape() as g:
        g.watch(actions)

        energies, _ = energy_network((observations, actions), training=training, network_state=network_state,
                                     step_type=tfa_step_type)

    # My energy sign is flipped relative to Igor's code,
    # so -1.0 here.
    denergies_dactions = g.gradient(energies, actions) * -1.0

    return denergies_dactions, energies


def langevin_step(energy_network, observations, actions, training, policy_state, tfa_step_type, stepsize,
                  min_actions, max_actions):
    """Single step of Langevin update."""
    l_lambda = 1.0
    delta_action_clip = 0.1

    # Langevin dynamics step
    de_dact, energies = gradient_wrt_act(energy_network, observations, actions, training, policy_state, tfa_step_type)

    # This effectively scales the gradient as if the actions were in a min-max range of -1 to 1.
    delta_action_clip = delta_action_clip * 0.5 * (max_actions - min_actions)

    # TODO(peteflorence): can I get rid of this copy, for performance?
    # Times 1.0 since I don't trust tf.identity to make a deep copy.
    unclipped_de_dact = de_dact * 1.0
    grad_norms = tf.linalg.norm(unclipped_de_dact, axis=1, ord=np.inf)

    gradient_scale = 0.5  # this is in the Langevin dynamics equation.
    de_dact = (gradient_scale * l_lambda * de_dact + tf.random.normal(tf.shape(actions)) * l_lambda * 1.0)
    delta_actions = stepsize * de_dact

    # Clip to box.
    delta_actions = tf.clip_by_value(delta_actions, -delta_action_clip, delta_action_clip)

    actions = actions - delta_actions
    actions = tf.clip_by_value(actions, min_actions, max_actions)

    return actions, energies, grad_norms


@tf.function
def langevin_actions_given_obs(energy_network, observations, action_samples, policy_state, min_actions, max_actions,
                               num_iterations=100, training=False, tfa_step_type=(), stop_chain_grad=True,
                               return_chain=False):
    """Given obs and actions, use dE(obs,act)/dact to perform Langevin MCMC."""
    stepsize = 1e-1
    actions = tf.identity(action_samples)

    schedule = PolynomialSchedule(1e-1, 1e-5, 2.0, num_iterations)
    b_times_n = tf.shape(action_samples)[0]
    act_dim = tf.shape(action_samples)[-1]

    # full_chain_actions is actually currently [1, ..., N]
    full_chain_actions = tf.zeros((num_iterations, b_times_n, act_dim))
    # full_chain_energies will also be for [0, ..., N-1]
    full_chain_energies = tf.zeros((num_iterations, b_times_n))
    # full_chain_grad_norms will be for [0, ..., N-1]
    full_chain_grad_norms = tf.zeros((num_iterations, b_times_n))

    for step_index in tf.range(num_iterations):

        actions, energies, grad_norms = langevin_step(energy_network, observations, actions, training, policy_state,
                                                      tfa_step_type, stepsize, min_actions, max_actions)

        actions = tf.stop_gradient(actions)

        if stop_chain_grad:
            actions = tf.stop_gradient(actions)
        stepsize = schedule.get_rate(step_index + 1)  # Get it for the next round.

        if return_chain:
            (full_chain_actions, full_chain_energies,
             full_chain_grad_norms) = update_chain_data(num_iterations, step_index, actions, energies, grad_norms,
                                                        full_chain_actions, full_chain_energies, full_chain_grad_norms)

    if return_chain:
        data_fields = ['actions', 'energies', 'grad_norms']
        ChainData = collections.namedtuple('ChainData', data_fields)
        chain_data = ChainData(full_chain_actions, full_chain_energies, full_chain_grad_norms)
        return actions, chain_data
    else:
        return actions


def get_probabilities(energy_network, batch_size, num_action_samples, observations, actions, training):
    """Get probabilities to post-process Langevin results."""
    net_logits, _ = energy_network((observations, actions), training=training)
    net_logits = tf.reshape(net_logits, (batch_size, num_action_samples))
    probs = tf.nn.softmax(net_logits / 1.0, axis=1)
    probs = tf.reshape(probs, (-1,))
    return probs
