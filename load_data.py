import tensorflow as tf
import numpy as np
import tqdm
import collections

from tf_agents.networks import nest_map
from tf_agents.trajectories.time_step import StepType
from tf_agents.trajectories.trajectory import Trajectory
from tf_agents.utils import example_encoding, example_encoding_dataset
from tf_agents.utils import tensor_normalizer


def filter_episodes(traj):
    step_types = traj.step_type
    seq_len = tf.cast(tf.shape(step_types)[0], tf.int32)

    first_frames = tf.where(step_types == StepType.FIRST)

    if tf.shape(first_frames)[0] == 0:
        # No first frame, return sequence as is.
        inds = tf.range(0, seq_len)
    else:
        ind_start = tf.cast(first_frames[-1, 0], tf.int32)
        if ind_start == 0:
            # Last episode starts on the first frame, return as is.
            inds = tf.range(0, seq_len)
        else:
            # Otherwise, resample so that the last episode's first frame is
            # replicated to the beginning of the sample. In the example above we want:
            # [3, 3, 3, 3, 4, 5].
            inds_start = tf.tile(ind_start[None], ind_start[None])
            inds_end = tf.range(ind_start, seq_len)
            inds = tf.concat([inds_start, inds_end], axis=0)

    def _resample(arr):
        if isinstance(arr, tf.Tensor):
            return tf.gather(arr, inds)
        else:
            return arr  # empty or None

    observation = tf.nest.map_structure(_resample, traj.observation)

    return Trajectory(
        step_type=_resample(traj.step_type),
        action=_resample(traj.action),
        policy_info=_resample(traj.policy_info),
        next_step_type=_resample(traj.next_step_type),
        reward=_resample(traj.reward),
        discount=_resample(traj.discount),
        observation=observation)


def load_tf_record_dataset_sequence(path_to_shards, seq_len):
    """Create tf.data.Dataset from sharded tfrecords."""
    specs = []
    for dataset_file in path_to_shards:
        spec_path = dataset_file + example_encoding_dataset._SPEC_FILE_EXTENSION
        dataset_spec = example_encoding_dataset.parse_encoded_spec_from_file(spec_path)
        specs.append(dataset_spec)

    def interleave_func(shard):
        dataset = tf.data.TFRecordDataset(shard, buffer_size=100).cache().repeat()
        dataset = dataset.window(seq_len, shift=1, stride=1, drop_remainder=True)
        return dataset.flat_map(lambda window: window.batch(seq_len, drop_remainder=True))

    def set_shape_obs(traj):
        def set_elem_shape(obs):
            obs_shape = obs.get_shape()
            return tf.ensure_shape(obs, [seq_len] + obs_shape[1:])

        observation = tf.nest.map_structure(set_elem_shape, traj.observation)
        return traj._replace(observation=observation)

    decoder = example_encoding.get_example_decoder(specs[0], batched=True, compress_image=True)
    dataset = tf.data.Dataset.from_tensor_slices(path_to_shards).repeat()
    num_parallel_calls = len(path_to_shards)
    dataset = dataset.interleave(interleave_func, deterministic=False, cycle_length=len(path_to_shards), block_length=1,
                                 num_parallel_calls=num_parallel_calls)
    dataset = dataset.map(decoder, num_parallel_calls=num_parallel_calls)
    dataset = dataset.map(filter_episodes, num_parallel_calls=num_parallel_calls)
    dataset = dataset.map(set_shape_obs, num_parallel_calls=num_parallel_calls)
    return dataset


def get_shards(dataset_path, separator=','):
    """Globs a dataset or aggregates records from a set of datasets."""
    if separator in dataset_path:
        # Data is a ','-separated list of training paths. Glob them all and then
        # aggregate into one dataset.
        dataset_paths = dataset_path.split(separator)
        shards = []
        for d in dataset_paths:
            shards.extend(tf.io.gfile.glob(d))
    else:
        shards = tf.io.gfile.glob(dataset_path)
    return shards


def create_sequence_datasets(dataset_path, batch_size):
    """Get dataset from a given path of tfrecords shards"""
    path_to_shards = get_shards(dataset_path)

    sequence_dataset = load_tf_record_dataset_sequence(path_to_shards, seq_len=2)
    sequence_dataset = sequence_dataset.repeat().shuffle(10000).batch(batch_size, drop_remainder=True)

    return sequence_dataset


def create_dataset_fn(dataset_path, batch_size, norm_function=None):
    """Returns function for loading dataset from tfrecords."""
    def create_train_fn():
        train_data = create_sequence_datasets(dataset_path, batch_size)

        def flatten_and_cast_action(action):
            flat_actions = tf.nest.flatten(action)
            flat_actions = [tf.cast(a, tf.float32) for a in flat_actions]
            return tf.concat(flat_actions, axis=-1)

        train_data = train_data.map(lambda trajectory:
                                    trajectory._replace(action=flatten_and_cast_action(trajectory.action)))
        train_data = train_data.map(lambda trajectory: ((trajectory.observation,
                                                         trajectory.action[:, -1, Ellipsis]), ()))

        if norm_function:
            train_data = train_data.map(norm_function)

        return train_data

    return create_train_fn


# Normalizer ------------------

class ChanRunningStatistics:
    """Implements Chan's algorithm.

    For more details, see the parallel algorithm of Chan et al. at:
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    """

    def __init__(self, initial_mean):
        self._n = 0
        if len(initial_mean.shape) > 1:
            self._mean = initial_mean[0, 0, :] * 0
        else:
            self._mean = initial_mean * 0
        self._m2 = 0
        self._m2_b_c = 0

    def update_running_statistics(self, sample):
        """Applies Chan's update rule to the running statistics."""
        sample_n = 1

        if len(sample.shape) > 1:
            # Compute statistics for last dim only. Reshape to turn other dims into
            # batch dim.
            sample = np.reshape(sample, [-1, sample.shape[-1]])
            sample_n = sample.shape[0]

            avg_sample, var_sample = tf.nn.moments(tf.convert_to_tensor(sample), axes=[0])
            avg_sample = avg_sample.numpy()
            var_sample = var_sample.numpy()
            m2_sample = var_sample * sample_n

        else:
            avg_sample = sample
            m2_sample = 0.0

        self._n, self._mean, self._m2, self._m2_b_c = (
            tensor_normalizer.parallel_variance_calculation(sample_n, avg_sample, m2_sample, self._n, self._mean,
                                                            self._m2, self._m2_b_c, ))

    @property
    def mean(self):
        return self._mean

    @property
    def std(self):
        return np.sqrt(self.variance)

    @property
    def variance(self):
        return self._m2 / self._n

    @property
    def n(self):
        return self._n


def _action_update(action, min_action, max_action):
    """Updates the action statistics."""
    action = action.numpy()

    # Adding a batch dimension so that numpy can do per-dimension min
    action = action[None, Ellipsis]
    if min_action is None:
        min_action = action.min(axis=0)
        max_action = action.max(axis=0)
    else:
        min_action = np.minimum(min_action, action.min(axis=0))
        max_action = np.maximum(max_action, action.max(axis=0))

    return min_action, max_action


def compute_dataset_statistics(dataset, num_samples):
    obs_statistics = None
    act_statistics = None

    with tqdm.tqdm(desc="Computing Dataset Statistics", total=num_samples) as progress_bar:

        observation = None
        action = None

        for observation, action in dataset.unbatch().take(num_samples):
            flat_obs = tf.nest.flatten(observation)
            flat_actions = tf.nest.flatten(action)

            if obs_statistics is None:
                # Initialize all params
                num_obs = len(flat_obs)
                num_act = len(flat_actions)

                # [0] on the observation to take single value out of time dim.
                obs_statistics = [ChanRunningStatistics(o[0].numpy()) for o in flat_obs]
                act_statistics = [ChanRunningStatistics(a.numpy()) for a in flat_actions]

                min_actions = [None for _ in range(num_act)]
                max_actions = [None for _ in range(num_act)]

            for obs, obs_stat in zip(flat_obs, obs_statistics):
                # Iterate over time dim.
                for o in obs:
                    obs_stat.update_running_statistics(o.numpy())

            for act, act_stat in zip(flat_actions, act_statistics):
                act_stat.update_running_statistics(act.numpy())

            min_actions, max_actions = zip(*tf.nest.map_structure(_action_update, flat_actions, min_actions,
                                                                  max_actions, check_types=False))

            progress_bar.update(1)

    assert obs_statistics[0].n > 2

    obs_norm_layers = []
    act_norm_layers = []
    act_denorm_layers = []
    for obs_stat in obs_statistics:
        obs_norm_layers.append(StdNormalizationLayer(mean=obs_stat.mean, std=obs_stat.std))

    for act_stat in act_statistics:
        act_norm_layers.append(MinMaxNormalizationLayer(vmin=min_actions[0], vmax=max_actions[0]))
        act_denorm_layers.append(MinMaxDenormalizationLayer(vmin=min_actions[0], vmax=max_actions[0]))

    obs_norm_layers = nest_map.NestMap(tf.nest.pack_sequence_as(observation, obs_norm_layers))

    act_norm_layers = act_norm_layers[0]
    act_denorm_layers = act_denorm_layers[0]
    min_actions = min_actions[0]
    max_actions = max_actions[0]

    # Initialize act_denorm_layers:
    act_denorm_layers(min_actions)
    return obs_norm_layers, act_norm_layers, act_denorm_layers, min_actions, max_actions


EPS = np.finfo(np.float32).eps


class IdentityLayer(tf.keras.layers.Layer):
    def __init__(self, cast_dtype):
        super(IdentityLayer, self).__init__(trainable=False)
        self.cast_dtype = cast_dtype

    def __call__(self, x, **kwargs):
        return tf.cast(x, self.cast_dtype)


class StdNormalizationLayer(tf.keras.layers.Layer):
    """Maps an un-normalized vector to zmuv."""

    def __init__(self, mean, std):
        super(StdNormalizationLayer, self).__init__(trainable=False)
        self._mean = mean.astype(np.float32)
        self._std = std.astype(np.float32)

    def __call__(self, vector, **kwargs):
        vector = tf.cast(vector, tf.float32)
        return (vector - self._mean) / tf.maximum(self._std, EPS)


class StdDenormalizationLayer(tf.keras.layers.Layer):
    """Maps a zmuv-normalized vector back to its original mean and std."""

    def __init__(self, mean, std):
        super(StdDenormalizationLayer, self).__init__(trainable=False)
        self._mean = mean.astype(np.float32)
        self._std = std.astype(np.float32)

    def __call__(self, vector, mean_offset=True, **kwargs):
        vector = tf.cast(vector, tf.float32)
        result = (vector * tf.maximum(self._std, EPS))
        if mean_offset:
            result += self._mean
        return result


class MinMaxLayer(tf.keras.layers.Layer):
    def __init__(self, vmin, vmax):
        super(MinMaxLayer, self).__init__(trainable=False)
        self._min = vmin.astype(np.float32)
        self._max = vmax.astype(np.float32)
        self._mean_range = (self._min + self._max) / 2.0
        self._half_range = (0.5 * (self._max - self._min))
        # Half_range shouldn't already be negative.
        self._half_range = tf.maximum(self._half_range, EPS)


class MinMaxNormalizationLayer(MinMaxLayer):
    """Maps an un-normalized vector to -1, 1."""

    def __call__(self, vector, **kwargs):
        vector = tf.cast(vector, tf.float32)
        return (vector - self._mean_range) / self._half_range


class MinMaxDenormalizationLayer(MinMaxLayer):
    """Maps -1, 1 vector back to un-normalized."""

    def __call__(self, vector, **kwargs):
        vector = tf.cast(vector, tf.float32)
        return (vector * self._half_range) + self._mean_range


NormalizationInfo = collections.namedtuple('NormalizationInfo', ['obs_norm_layer', 'act_norm_layer', 'act_denorm_layer',
                                                                 'min_actions', 'max_actions'])


def drop_info_and_float_cast(sample, _):
    obs, action = sample

    for img_key in ['rgb', 'front', 'image']:
        if isinstance(obs, dict) and img_key in obs:
            obs[img_key] = tf.image.convert_image_dtype(obs[img_key], dtype=tf.float32)

    return tf.nest.map_structure(lambda t: tf.cast(t, tf.float32), (obs, action))


def get_normalizers(train_data):
    """Compute dataset statistics and returns data normalization layers."""
    statistics_dataset = train_data
    statistics_dataset = statistics_dataset.map(drop_info_and_float_cast,
                                                num_parallel_calls=tf.data.experimental.AUTOTUNE).prefetch(100)

    num_samples = 5000
    (obs_norm_layer, act_norm_layer, act_denorm_layer,
     min_actions, max_actions) = (compute_dataset_statistics(statistics_dataset, num_samples=num_samples))

    def norm_train_data_fn(obs_and_act, nothing):
        obs = obs_and_act[0]
        for img_key in ['rgb', 'front', 'image']:
            if isinstance(obs, dict) and img_key in obs:
                obs[img_key] = tf.image.convert_image_dtype(obs[img_key], dtype=tf.float32)
        act = obs_and_act[1]
        normalized_obs = obs_norm_layer(obs)
        if isinstance(obs_norm_layer, nest_map.NestMap):
            normalized_obs, _ = normalized_obs
        normalized_act = act_norm_layer(act)
        if isinstance(act_norm_layer, nest_map.NestMap):
            normalized_act, _ = normalized_act
        return ((normalized_obs, normalized_act), nothing)

    norm_info = NormalizationInfo(obs_norm_layer, act_norm_layer, act_denorm_layer, min_actions, max_actions)

    return norm_info, norm_train_data_fn
