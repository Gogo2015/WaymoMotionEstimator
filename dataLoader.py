import os
import tensorflow as tf

state_features = {
    'state/future/valid':
        tf.io.FixedLenFeature([128, 80], tf.int64, default_value=None),
    'state/future/x':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/y':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/past/valid':
        tf.io.FixedLenFeature([128, 10], tf.int64, default_value=None),
    'state/past/x':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/y':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/tracks_to_predict':
        tf.io.FixedLenFeature([128], tf.int64, default_value=None),
}

features_description = {}
features_description.update(state_features)


def transform(single_example, features, past_steps, future_steps):
    """
    Input:
    single_example: single example gotten from flat_map function

    Output:
    transformed: data split into (past, future, future_valid)
        - past: [N, past_steps, 2]
        - future: [N, future_steps, 2]

    """

    #Parse example
    parsed = tf.io.parse_single_example(single_example, features)

    #Retrieve features
    past_x, past_y = parsed['state/past/x'], parsed['state/past/y']
    future_x, future_y = parsed['state/future/x'], parsed['state/future/y']
    future_valid = parsed['state/future/valid']
    tracks = parsed['state/tracks_to_predict']

    #Get mask for agents that we want to predict
    mask = tracks > 0

    #Apply mask
    past_x = tf.boolean_mask(past_x, mask)
    past_y = tf.boolean_mask(past_y, mask)
    future_x = tf.boolean_mask(future_x, mask)
    future_y = tf.boolean_mask(future_y, mask)
    future_valid = tf.boolean_mask(future_valid, mask)

    n = tf.shape(past_x)[0]

    #Translate agent's position so last point at origin
    anchor_x = past_x[:, -1] if past_steps > 0 else tf.zeros((n,), tf.float32)
    anchor_y = past_y[:, -1] if past_steps > 0 else tf.zeros((n,), tf.float32)
    past_x -= anchor_x[:, None]
    past_y -= anchor_y[:, None]
    future_x -= anchor_x[:, None]
    future_y -= anchor_y[:, None]

    #Rotate agent so motion is along +x axis
    dx = past_x[:, -1] - past_x[:, -2]
    dy = past_y[:, -1] - past_y[:, -2]
    yaw = tf.atan2(dy, dx)
    c, s = tf.cos(-yaw)[:, None], tf.sin(-yaw)[:, None]

    def rot(x, y):
        return x * c - y * s, x * s + y * c
    
    past_x, past_y = rot(past_x, past_y)
    future_x, future_y = rot(future_x, future_y)

    #Create tensors for returning
    past = tf.stack([past_x, past_y], axis=-1)
    future = tf.stack([future_x, future_y], axis=-1)

    return tf.data.Dataset.from_tensor_slices((past, future, future_valid))

def get_data(data_dir, batch_size, past_steps=10, future_steps=80, train_split=0.9, training=True):
    """
    Load and process trajectory data from TFRecord files.

    Args:
        data_dir: Directory containing TFRecord files
        batch_size: Batch size for training
        past_steps: Number of past timesteps
        future_steps: Number of future timesteps to predict
        train_split: Fraction of data to use for training (vs validation)
        training: If True, return train and val datasets; if False, return test dataset
    """
    #Get dataset of all files in data folder
    files = [
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if ".tfrecord-" in f
            ]
    dataset = tf.data.TFRecordDataset(files)

    dataset = dataset.flat_map(lambda x: transform(x, features_description, past_steps, future_steps))

    dataset = dataset.shuffle(2048)
    dataset_size = sum(1 for _ in dataset)

    if training:
        train_size = int(train_split * dataset_size)

        train_ds = dataset.take(train_size)
        val_ds = dataset.skip(train_size)

        train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        return train_ds, val_ds

    else:
        test_ds = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        return test_ds




