import os
import numpy as np
import functools
import tensorflow as tf 
from colorama import Fore, Style



class TFRecordMotionDataset(object):
    """
    Dataset class for AMASS, CMU, and H3.6M datasets stored as TFRecord files.
    """
    def __init__(self, data_path, meta_data_path, batch_size, shuffle, windows_size, window_type, num_parallel_calls, normalize):
        print(f'{Fore.YELLOW}')
        print('Loading motion data from {}'.format(os.path.abspath(data_path)))
        print(f'{Style.RESET_ALL}')
        # Extract a window randomly. If the sequence is shorter, ignore it.
        self.windows_size = windows_size 
        # Whether to extract windows randomly, from the beginning or the middle of the sequence.
        self.window_type = window_type
        self.num_parallel_calls = num_parallel_calls
        self.normalize = normalize
        
        self.tf_data = None
        self.data_path = data_path
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Load statistics and other data summary stored in the meta-data file.
        self.meta_data = self.load_meta_data(meta_data_path)

        self.mean_all = self.meta_data['mean_all']
        self.var_all = self.meta_data['var_all']
        self.mean_channel = self.meta_data['mean_channel']
        self.var_channel = self.meta_data['var_channel']

        self.tf_data_transformations()
        self.tf_data_normalization()
        self.tf_data_to_model()

        self.tf_samples = self.tf_data.as_numpy_iterator()


    def tf_data_transformations(self):
        """
        Loads the raw data and apply preprocessing.
        This method is also used in calculation of the dataset statistics (i.e., meta-data file).
        """
        tf_data_opt = tf.data.Options()

        self.tf_data = tf.data.TFRecordDataset.list_files(self.data_path, seed=1234, shuffle=self.shuffle)
        self.tf_data = self.tf_data.with_options(tf_data_opt)
        self.tf_data = self.tf_data.apply(tf.data.experimental.parallel_interleave(tf.data.TFRecordDataset, cycle_length=self.num_parallel_calls, block_length=1, sloppy=self.shuffle))
        self.tf_data = self.tf_data.map(functools.partial(self.parse_single_tfexample_fn), num_parallel_calls=self.num_parallel_calls)
        self.tf_data = self.tf_data.prefetch(self.batch_size*10)
        if self.shuffle:
            self.tf_data = self.tf_data.shuffle(self.batch_size*10)

        if self.windows_size > 0:
            self.tf_data = self.tf_data.filter(functools.partial(self.pp_filter))
            if self.window_type == 'from_beginning':
                self.tf_data = self.tf_data.map(functools.partial(self.pp_get_windows_beginning), num_parallel_calls=self.num_parallel_calls)
            elif self.window_type == 'from_center':
                self.tf_data = self.tf_data.map(functools.partial(self.pp_get_windows_middle), num_parallel_calls=self.num_parallel_calls)
            elif self.window_type == 'random':
                self.tf_data = self.tf_data.map(functools.partial(self.pp_get_windows_random), num_parallel_calls=self.num_parallel_calls)
            else:
                raise Exception("Unknown window type.")


    def tf_data_normalization(self):
        # Applies normalization.
        if self.normalize:
            self.tf_data = self.tf_data.map(functools.partial(self.normalize_zero_mean_unit_variance_channel, key="poses"), num_parallel_calls=self.num_parallel_calls)
        else:  # Some models require the feature size.
            self.tf_data = self.tf_data.map(functools.partial(self.pp_set_feature_size), num_parallel_calls=self.num_parallel_calls)


    def tf_data_to_model(self):
        # Converts the data into the format that a model expects. Creates input, target, sequence_length, etc.
        self.tf_data = self.tf_data.map(functools.partial(self.to_model_inputs), num_parallel_calls=self.num_parallel_calls)
        self.tf_data = self.tf_data.padded_batch(self.batch_size, padded_shapes=tf.compat.v1.data.get_output_shapes(self.tf_data))
        self.tf_data = self.tf_data.prefetch(2)
        if tf.test.is_gpu_available():
            self.tf_data = self.tf_data.apply(tf.data.experimental.prefetch_to_device('/device:GPU:0'))


    def load_meta_data(self, meta_data_path):
        """
        Loads meta-data file given the path. It is assumed to be in numpy.
        Args:
            meta_data_path:
        Returns:
            Meta-data dictionary or False if it is not found.
        """
        if not meta_data_path or not os.path.exists(meta_data_path):
            print("Meta-data not found.")
            return False
        else:
            return np.load(meta_data_path, allow_pickle=True)['stats'].tolist()


    def pp_set_feature_size(self, sample):
        seq_len = sample["poses"].get_shape().as_list()[0]
        sample["poses"].set_shape([seq_len, self.mean_channel.shape[0]])
        return sample


    def pp_filter(self, sample):
        return tf.shape(sample["poses"])[0] >= self.windows_size


    def pp_get_windows_random(self, sample):
        start = tf.random.uniform((1, 1), minval=0, maxval=tf.shape(sample["poses"])[0]-self.windows_size+1, dtype=tf.int32)[0][0]
        end = tf.minimum(start+self.windows_size, tf.shape(sample["poses"])[0])
        sample["poses"] = sample["poses"][start:end, :]
        sample["shape"] = tf.shape(sample["poses"])
        return sample


    def pp_get_windows_beginning(self, sample):
        # Extract a window from the beginning of the sequence.
        sample["poses"] = sample["poses"][0:self.windows_size, :]
        sample["shape"] = tf.shape(sample["poses"])
        return sample


    def pp_get_windows_middle(self, sample):
        # Window is located at the center of the sequence.
        seq_len = tf.shape(sample["poses"])[0]
        start = tf.maximum((seq_len//2) - (self.windows_size//2), 0)
        end = start + self.windows_size
        sample["poses"] = sample["poses"][start:end, :]
        sample["shape"] = tf.shape(sample["poses"])
        return sample


    def to_model_inputs(self, tf_sample_dict):
        """
        Transforms a TFRecord sample into a more general sample representation where we use global keys to represent
        the required fields by the models.
        Args:
            tf_sample_dict:
        Returns:
        """
        model_sample = dict()
        model_sample['seq_len'] = tf_sample_dict["shape"][0]
        model_sample['inputs'] = tf_sample_dict["poses"]
        model_sample['motion_targets'] = tf_sample_dict["poses"]
        model_sample['id'] = tf_sample_dict["sample_id"]
        return model_sample


    def parse_single_tfexample_fn(self, proto):
        feature_to_type = {
            "file_id": tf.io.FixedLenFeature([], dtype=tf.string),
            "db_name": tf.io.FixedLenFeature([], dtype=tf.string),
            "shape": tf.io.FixedLenFeature([2], dtype=tf.int64),
            "poses": tf.io.VarLenFeature(dtype=tf.float32),
        }

        parsed_features = tf.io.parse_single_example(proto, feature_to_type)
        parsed_features["poses"] = tf.reshape(tf.sparse.to_dense(parsed_features["poses"]), parsed_features["shape"])

        file_id = tf.strings.substr(parsed_features["file_id"], 0, tf.strings.length(parsed_features["file_id"]))
        parsed_features["sample_id"] = tf.strings.join([parsed_features["db_name"], file_id], separator="/")

        return parsed_features


    def normalize_zero_mean_unit_variance_all(self, sample_dict, key):
        sample_dict[key] = (sample_dict[key] - self.mean_all) / self.var_all
        return sample_dict


    def normalize_zero_mean_unit_variance_channel(self, sample_dict, key):
        sample_dict[key] = (sample_dict[key] - self.mean_channel) / self.var_channel
        return sample_dict


    def unnormalize_zero_mean_unit_variance_all(self, sample_dict, key):
        if self.normalize:
            sample_dict[key] = sample_dict[key] * self.var_all + self.mean_all
        return sample_dict


    def unnormalize_zero_mean_unit_variance_channel(self, sample_dict, key):
        if self.normalize:
            sample_dict[key] = sample_dict[key] * self.var_channel + self.mean_channel
        return sample_dict


    def get_tf_samples(self):
        self.tf_samples = self.tf_data.as_numpy_iterator()
        return self.tf_samples


    def __len__(self):
        return sum(1 for _ in self.tf_data)