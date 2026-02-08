import tensorflow as tf

def multimodal_loss(y_true, pred_trajectories, confidences, valid_mask):
    """
    Winner-takes-all loss for multi-modal prediction
    
    Input:
        y_true: (batch, 80, 2) - ground truth future trajectory
        pred_trajectories: (batch, K, 80, 2) - K predicted trajectories
        confidences: (batch, K) - predicted mode probabilities
        valid_mask: (batch, 80) - which timesteps are valid (1.0) or not (0.0)
    
    Output:
        total_loss: scalar
        reg_loss: scalar (for logging)
        cls_loss: scalar (for logging)
    """
    batch_size = tf.shape(y_true)[0]
    num_modes = tf.shape(pred_trajectories)[1]
    
    # Compute distance from GT to each mode
    # Expand: (batch, 80, 2) to (batch, 1, 80, 2)
    y_true_expanded = tf.expand_dims(y_true, 1)
    
    # Diff
    diff = pred_trajectories - y_true_expanded
    
    #Compute distances(Euclidean)
    distances = tf.sqrt(tf.reduce_sum(tf.square(diff), axis=-1) + 1e-8)
    
    # Apply mask
    valid_mask_expanded = tf.expand_dims(tf.cast(valid_mask, tf.float32), 1)
    distances = distances * valid_mask_expanded
    
    num_valid = tf.reduce_sum(valid_mask_expanded, axis=-1)  
    ade_per_mode = tf.reduce_sum(distances, axis=-1) / (num_valid + 1e-8)
    
    #Find best mode (minADE)
    best_mode_idx = tf.argmin(ade_per_mode, axis=1)
    best_mode_mask = tf.one_hot(best_mode_idx, num_modes)
    
    #control loss only for best mode
    control_loss = tf.reduce_sum(ade_per_mode * best_mode_mask, axis=1)
    control_loss = tf.reduce_mean(control_loss)
    
    #intent loss - predict which mode is best
    intent_loss = tf.keras.losses.categorical_crossentropy(
        best_mode_mask, confidences
    )
    intent_loss = tf.reduce_mean(intent_loss)
    
    # Combined loss
    total_loss = control_loss + intent_loss
    
    return total_loss, control_loss, intent_loss