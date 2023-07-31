import collections
import copy

import gym
from gym import spaces
from gym.envs import registration
import matplotlib.pyplot as plt
import numpy as np


from tf_agents.metrics import py_metrics
from tf_agents.typing import types
from tf_agents.utils import nest_utils
from tf_agents.utils import numpy_storage


class Particle1d(gym.Env):

    def get_metrics(self, num_episodes):
        metrics = [AverageGoalDistance(self, buffer_size=num_episodes),
                   AverageFinalGoalDistance(self, buffer_size=num_episodes),
                   AverageSuccessMetric(self, buffer_size=num_episodes)]

        success_metric = metrics[-1]

        return metrics, success_metric

    def __init__(self, n_steps=50, seed=None, dt=0.005, repeat_actions=10, k_p=10., k_v=5.,
                 goal_distance=0.05):

        self.n_steps = n_steps
        self.goal_distance = goal_distance

        self._rng = np.random.RandomState(seed=seed)

        self.action_space = spaces.Box(low=-4., high=4., shape=(1,), dtype=np.float32)
        self.observation_space = self._create_observation_space()

        self.dt = dt
        self.repeat_actions = repeat_actions
        # Make sure is a multiple.
        assert int(1 / self.dt) % self.repeat_actions == 0

        self.k_p = k_p
        self.k_v = k_v

        self.reset()

    def _create_observation_space(self):
        obs_dict = collections.OrderedDict(
            pos_agent=spaces.Box(low=-2., high=2., shape=(1,), dtype=np.float32),
            vel_agent=spaces.Box(low=-1e2, high=1e2, shape=(1,), dtype=np.float32),
            pos_goal=spaces.Box(low=-2., high=2., shape=(1,), dtype=np.float32),
        )

        return spaces.Dict(obs_dict)

    def seed(self, seed=None):
        self._rng = np.random.RandomState(seed=seed)

    def reset(self):
        self.steps = 0

        self.obs_log = []
        self.act_log = []

        obs = dict()
        obs['pos_agent'] = self._rng.uniform(low=-2., high=2., size=1).astype(np.float32)
        obs['vel_agent'] = np.array([0.])
        obs['pos_goal'] = self._rng.uniform(low=-2., high=2., size=1).astype(np.float32)

        self.obs_log.append(obs)

        self.min_dist_to_goal = np.inf

        return self._get_state()

    def _get_state(self):
        return copy.deepcopy(self.obs_log[-1])

    def dist(self, goal):
        current_position = self.obs_log[-1]['pos_agent']
        return np.linalg.norm(current_position - goal)

    def _internal_step(self, action):
        self.act_log.append({'motion vector': action})  # pytype: disable=attribute-error
        obs = self.obs_log[-1]  # pytype: disable=attribute-error
        # u = k_p (x_{desired} - x) + k_v (xdot_{desired} - xdot)
        # xdot_{desired} is always (0, 0) -->
        # u = k_p (x_{desired} - x) - k_v (xdot)

        u_agent = self.k_p * (action) - self.k_v * (obs['vel_agent'])
        new_xy_agent = obs['pos_agent'] + obs['vel_agent'] * self.dt  # pytype: disable=attribute-error
        new_velocity_agent = obs['vel_agent'] + u_agent * self.dt  # pytype: disable=attribute-error

        obs = copy.deepcopy(obs)
        obs['pos_agent'] = new_xy_agent
        obs['vel_agent'] = new_velocity_agent
        self.obs_log.append(obs)  # pytype: disable=attribute-error


        # u_agent = self.k_p * (action - obs['pos_agent']) - self.k_v * (  # pytype: disable=attribute-error
        #     obs['vel_agent'])
        # new_xy_agent = obs['pos_agent'] + obs['vel_agent'] * self.dt  # pytype: disable=attribute-error
        # new_velocity_agent = obs['vel_agent'] + u_agent * self.dt  # pytype: disable=attribute-error
        # obs = copy.deepcopy(obs)
        # obs['pos_agent'] = new_xy_agent
        # obs['vel_agent'] = new_velocity_agent
        # self.obs_log.append(obs)  # pytype: disable=attribute-error

    def step(self, action):
        self.steps += 1

        # self.act_log.append({'motion vector': action})
        # obs = self.obs_log[-1]
        #
        # if np.linalg.norm(action) > self.max_step_size:
        #     action = (action / np.linalg.norm(action)) * self.max_step_size
        #
        # new_pos = obs['pos_agent'] + action
        #
        # new_obs = copy.deepcopy(obs)
        # new_obs['pos_agent'] = new_pos
        #
        # self.obs_log.append(new_obs)

        self._internal_step(action)
        for _ in range(self.repeat_actions - 1):  # pytype: disable=attribute-error
            self._internal_step(action)

        state = self._get_state()
        done = True if self.steps >= self.n_steps else False
        reward = self._get_reward(done)

        return state, reward, done, {}

    def _get_reward(self, done):
        self.min_dist_to_goal = min(self.dist(self.obs_log[0]['pos_goal']), self.min_dist_to_goal)

        reward_goal = True if self.min_dist_to_goal < self.goal_distance else False
        return 1.0 if (reward_goal and done) else 0.0

    @property
    def succeeded(self):
        hit_goal = True if self.min_dist_to_goal < self.goal_distance else False

        current_distance = self.dist(self.obs_log[0]['pos_goal'])
        still_at_goal = True if current_distance < self.goal_distance else False

        return hit_goal and still_at_goal

    def render(self, mode='rgb_array'):
        fig, _ = visualize_1d(self.obs_log, self.act_log)

        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        return data


class AverageGoalDistance(py_metrics.StreamingMetric):
    def __init__(self, env, name='AverageGoalDistance', buffer_size=10, batch_size=None):
        self._env = env
        super(AverageGoalDistance, self).__init__(name, buffer_size=buffer_size, batch_size=batch_size)

    def _reset(self, batch_size):
        """Resets stat gathering variables."""
        pass

    def _batched_call(self, trajectory):
        """Processes the trajectory to update the metric.

        Args:
        trajectory: a tf_agents.trajectory.Trajectory.
        """
        lasts = trajectory.is_last()
        if np.any(lasts):
            is_last = np.where(lasts)
            goal_distance = np.asarray(self._env.min_dist_to_goal, np.float32)

            if goal_distance.shape is ():  # pylint: disable=literal-comparison
                goal_distance = nest_utils.batch_nested_array(goal_distance)

            self.add_to_buffer(goal_distance[is_last])


class AverageFinalGoalDistance(py_metrics.StreamingMetric):
    def __init__(self, env, name='AverageFinalGoalDistance', buffer_size=10, batch_size=None):
        """Creates an AverageReturnMetric."""
        self._env = env
        super(AverageFinalGoalDistance, self).__init__(name, buffer_size=buffer_size, batch_size=batch_size)

    def _reset(self, batch_size):
        """Resets stat gathering variables."""
        pass

    def _batched_call(self, trajectory):
        """Processes the trajectory to update the metric.

        Args:
        trajectory: a tf_agents.trajectory.Trajectory.
        """
        lasts = trajectory.is_last()
        if np.any(lasts):
            is_last = np.where(lasts)
            final_dist = self._env.dist(self._env.obs_log[0]['pos_goal'])
            goal_distance = np.asarray(final_dist, np.float32)

            if goal_distance.shape is ():  # pylint: disable=literal-comparison
                goal_distance = nest_utils.batch_nested_array(goal_distance)

            self.add_to_buffer(goal_distance[is_last])


class AverageSuccessMetric(py_metrics.StreamingMetric):
    """Computes the average success of the environment."""

    def __init__(self, env, name='AverageSuccessMetric', buffer_size=10, batch_size=None):
        """Creates an AverageReturnMetric."""
        self._np_state = numpy_storage.NumpyState()
        self._env = env
        # TODO(ask Oscar to fix this for batched envs.)
        # We had a raise ValueError here but self._batched didn't exist.

        # Set a dummy value on self._np_state so it gets included in
        # the first checkpoint (before metric is first called).
        self._np_state.success = np.float64(0)
        super(AverageSuccessMetric, self).__init__(
            name, buffer_size=buffer_size, batch_size=batch_size)

    def _reset(self, batch_size):
        """Resets stat gathering variables."""
        self._np_state.success = np.zeros(shape=(batch_size,), dtype=np.float64)

    def _batched_call(self, trajectory):
        """Processes the trajectory to update the metric.

        Args:
        trajectory: a tf_agents.trajectory.Trajectory.
        """
        lasts = trajectory.is_last()
        if np.any(lasts):
            is_last = np.where(lasts)
            if self._env.succeeded:
                succeed = 1.0
            else:
                succeed = 0.0
            succeed = np.asarray(succeed, np.float32)
            if succeed.shape is ():  # pylint: disable=literal-comparison
                succeed = nest_utils.batch_nested_array(succeed)

            self.add_to_buffer(succeed[is_last])


def make_vector(step):
    """step is either an obs or an act."""
    return np.hstack(list(step.values()))


def make_vector_traj(log):
    """log is either an obs_log or an act_log."""
    vector_traj = []
    for step in log:
        vector_traj.append(make_vector(step))
    return np.array(vector_traj)


def visualize_1d(obs_log, act_log, ax=None, fig=None, show=False, last_big=False):
    # Assert it's 1D
    assert len(obs_log[0]['pos_agent']) == 1

    if ax is None:
        fig, ax = plt.subplots()
        # ax.set_aspect('equal')
        ax.set_xlim(-2., 2.)
        ax.set_ylim(0., 1.)

    # Since when render is called we don't know what the actions will be,
    # We may need to ignore the last obs.
    if len(obs_log) != len(act_log):
        if len(obs_log) == len(act_log) + 1:
            obs_log_ = obs_log[:-1]
        else:
            raise ValueError('Wrong length logs.')
    else:
        obs_log_ = obs_log

    # Visualize observations.
    pos_goal = obs_log[0]['pos_goal']
    ax.add_patch(plt.Circle((pos_goal[0], 0.5), 0.01, color='g'))

    # Now obs_log_ might be empty, in which case return.
    if not obs_log_:
        return fig, ax

    # Visualize actions.
    act_traj = make_vector_traj(act_log)
    # ax.scatter(act_traj[:, 0], act_traj[:, 1], marker='x', s=100, alpha=0.1, color='red')

    for i in range(len(obs_log_) - 1):
        alpha = float(i) / len(obs_log_)
        pos_agent_k = obs_log_[i]['pos_agent']
        pos_agent_kplus1 = obs_log_[i + 1]['pos_agent']
        pos_agent_2step = np.stack((pos_agent_k, pos_agent_kplus1))
        ax.plot(pos_agent_2step[:, 0], np.repeat(0.5, pos_agent_2step.shape[0]), alpha=alpha, linestyle=':', color='black')
    if last_big:
        ax.scatter(obs_log_[-1]['pos_agent'][0], 0.5, marker='o', s=50, color='black')
        # ax.scatter(act_traj[-1, 0], act_traj[-1, 0.5], marker='x', color='red', s=100)
    if show:
        plt.show()
    return fig, ax


# Register environment, so it can be loaded via name
if 'Particle1D-v1' in registration.registry.env_specs:
    del registration.registry.env_specs['Particle1D-v1']

registration.register(id='Particle1D-v1', entry_point=Particle1d)

