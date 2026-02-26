import tensorflow as tf

def mse(y_true, y_pred, valid_mask):
    valid_mask = tf.cast(valid_mask, tf.float32)[..., None]  # (B, T, 1)
    diff = tf.math.squared_difference(y_true, y_pred) * valid_mask  # (B, T, 2)

    # sum over coords/time for each sample
    per_sample_denom = tf.reduce_sum(valid_mask, axis=(1, 2)) + 1e-6   # (B,)
    per_sample_loss = tf.reduce_sum(diff, axis=(1, 2)) / per_sample_denom  # (B,)
    return tf.reduce_mean(per_sample_loss)

def ade(y_true, y_pred, valid_mask):
    # average displacement error over valid steps
    valid_mask = tf.cast(valid_mask, tf.float32)[..., None]

    dist = tf.norm(y_true - y_pred, axis=-1, keepdims=True)  # (B, 80, 1)
    dist = dist * valid_mask

    per_sample_tot = tf.reduce_sum(dist, axis=(1,2))
    per_sample_count = tf.reduce_sum(valid_mask, axis=(1,2)) + 1e-6

    return tf.reduce_mean(per_sample_tot / per_sample_count)

def fde(y_true, y_pred, valid_mask):
    valid_mask = tf.cast(valid_mask, tf.float32)

    # Check for samples with no valid timesteps
    has_valid = tf.reduce_sum(valid_mask, axis=1) > 0  # (B,)

    time_indices = tf.range(tf.shape(valid_mask)[1])[tf.newaxis, :]
    masked_times = valid_mask * tf.cast(time_indices, tf.float32)
    last_valid_idx = tf.cast(tf.math.reduce_max(masked_times, axis=1), tf.int32)

    batch_idx = tf.range(tf.shape(y_true)[0], dtype=tf.int32)
    idx = tf.stack([batch_idx, last_valid_idx], axis=1)

    last_true = tf.gather_nd(y_true, idx)
    last_pred = tf.gather_nd(y_pred, idx)

    per_sample_fde = tf.norm(last_true - last_pred, axis=-1)

    # Zero out FDE for samples with no valid steps
    per_sample_fde = per_sample_fde * tf.cast(has_valid, tf.float32)
    num_valid_samples = tf.reduce_sum(tf.cast(has_valid, tf.float32)) + 1e-6

    return tf.reduce_sum(per_sample_fde) / num_valid_samples
