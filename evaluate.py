import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 0 = all, 1 = info, 2 = warning, 3 = error

import tensorflow as tf
import numpy as np
import glob
from dataLoader import get_data
from models.ConvMLP import ConvMLP
from models.MultiModalConvMLP import MultiModalConvMLP
from visualize import visualize_trajectory, visualize_multimodal_trajectory
from metrics import mse, ade, fde

PAST_STEPS = 10
FUTURE_STEPS = 80
DATA_DIR = "gs://waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example/validation"

MODELS = [
    ConvMLP,
    MultiModalConvMLP
]

def get_latest_checkpoint(model_name):
    prefix_map = {
        'ConvMLP': 'waymo_public_baseline_*',
        'MultiModalConvMLP': 'waymo_public_multimodal_*',
    }
    pattern = f"trained_models/checkpoints/{prefix_map[model_name]}"
    dirs = sorted(glob.glob(pattern))
    if not dirs:
        print(f"No checkpoint dirs found for {model_name}")
        return None
    ckpt_path = tf.train.latest_checkpoint(dirs[-1])
    if ckpt_path is None:
        print(f"No checkpoint files in {dirs[-1]}")
        return None
    print(f"Using checkpoint: {ckpt_path}")
    return ckpt_path


def visualize(model_to_visualize):
    # Load test dataset
    test_ds = get_data(DATA_DIR, batch_size=1, training=False)
    model_name = model_to_visualize.__name__

    # Build model first
    model = model_to_visualize(PAST_STEPS, FUTURE_STEPS)
    model.build(input_shape=(None, PAST_STEPS, 2))

    # Load from checkpoint
    ckpt = tf.train.Checkpoint(model=model)
    ckpt_path = get_latest_checkpoint(model_name)
    if ckpt_path is None:
        return
    ckpt.restore(ckpt_path).expect_partial()
    print(f"Loaded {model_name} from checkpoint")

    os.makedirs(f"./gifs/{model_name}", exist_ok=True)

    # Generate 5 gifs
    for i, (past, future, valid) in enumerate(test_ds.take(5)):
        pred_future = model(past, training=False).numpy()[0]      # (FUTURE_STEPS, 2)
        past_np = past.numpy()[0]
        true_future_np = future.numpy()[0]
        save_path = f"./gifs/{model_name}/agent_traj{i}.gif"
        visualize_trajectory(past_np, true_future_np, pred_future, save_path)


def visualize_multimodal(model_to_visualize):
    """Visualize multi-modal predictions"""
    test_ds = get_data(DATA_DIR, batch_size=1, training=False)
    model_name = model_to_visualize.__name__

    model = model_to_visualize()
    model.build(input_shape=(None, PAST_STEPS, 2))

    ckpt = tf.train.Checkpoint(model=model)
    ckpt_path = get_latest_checkpoint(model_name)
    if ckpt_path is None:
        return
    ckpt.restore(ckpt_path).expect_partial()
    print(f"Loaded {model_name} from checkpoint")

    os.makedirs(f"./gifs/{model_name}", exist_ok=True)
    
    # Generate 5 gifs
    for i, (past, future, valid) in enumerate(test_ds.take(5)):
        pred_trajectories, confidences = model(past, training=False)
        
        # Get first sample from batch
        pred_traj_np = pred_trajectories.numpy()[0]  # (K, 80, 2)
        conf_np = confidences.numpy()[0]  # (K,)
        past_np = past.numpy()[0]
        true_future_np = future.numpy()[0]
        
        save_path = f"./gifs/{model_name}/test_agent_{i}.gif"
        visualize_multimodal_trajectory(past_np, true_future_np, 
                                       pred_traj_np, conf_np, save_path)


def test_model(model_to_test):
    test_ds = get_data(DATA_DIR, batch_size=64, training=False)
    is_multimodal = (model_to_test == MultiModalConvMLP)

    model = model_to_test(PAST_STEPS, FUTURE_STEPS) if not is_multimodal else model_to_test()
    model_name = model_to_test.__name__
    model_path = f"trained_models/{model_name}.keras"

    # Load weights from checkpoint
    ckpt_path = get_latest_checkpoint(model_name)
    if ckpt_path is None:
        return

    ckpt = tf.train.Checkpoint(model=model)
    ckpt.restore(ckpt_path).expect_partial()
    print(f"Loaded {model_name} from checkpoint")

    avg_loss, avg_ade, avg_fde = 0.0, 0.0, 0.0
    num_batches = 0

    for past, future, valid in test_ds:
        if is_multimodal:
            pred_traj, conf = model(past, training=False)  # (B, K, 80, 2), (B, K)

            # minADE: pick best mode per sample
            diff = pred_traj - tf.expand_dims(future, 1)  # (B, K, 80, 2)
            dist = tf.norm(diff, axis=-1)  # (B, K, 80)
            valid_f = tf.cast(valid, tf.float32)
            masked_dist = dist * tf.expand_dims(valid_f, 1)  # (B, K, 80)
            num_valid = tf.reduce_sum(valid_f, axis=1, keepdims=True) + 1e-6  # (B, 1)
            ade_per_mode = tf.reduce_sum(masked_dist, axis=2) / num_valid  # (B, K)
            best_mode = tf.argmin(ade_per_mode, axis=1)  # (B,)

            # Gather best trajectory per sample
            batch_idx = tf.range(tf.shape(past)[0], dtype=tf.int64)
            gather_idx = tf.stack([batch_idx, best_mode], axis=1)
            best_pred = tf.gather_nd(pred_traj, gather_idx)  # (B, 80, 2)

            batch_loss = float(mse(future, best_pred, valid).numpy())
            batch_ade = float(ade(future, best_pred, valid).numpy())
            batch_fde = float(fde(future, best_pred, valid).numpy())
        else:
            pred = model(past, training=False)
            batch_loss = float(mse(future, pred, valid).numpy())
            batch_ade = float(ade(future, pred, valid).numpy())
            batch_fde = float(fde(future, pred, valid).numpy())

        avg_loss += batch_loss
        avg_ade += batch_ade
        avg_fde += batch_fde
        num_batches += 1

        print(f"loss {batch_loss:.4f} | ADE {batch_ade:.4f} | FDE {batch_fde:.4f}")

    avg_loss /= num_batches
    avg_ade /= num_batches
    avg_fde /= num_batches
    print(f"\nAvg loss {avg_loss:.4f} | Avg ADE {avg_ade:.4f} | Avg FDE {avg_fde:.4f}")


if __name__ == "__main__":
    choice = input("test || vis? ")

    if choice.lower() == 'test':
        for model in MODELS:
            test_model(model)
    else:
        for model in MODELS:
            if model == MultiModalConvMLP:
                visualize_multimodal(model)
            else:
                visualize(model)