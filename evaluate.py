import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 0 = all, 1 = info, 2 = warning, 3 = error

import tensorflow as tf
import numpy as np
from dataLoader import get_data
from models.ConvMLP import ConvMLP
from models.MultiModalConvMLP import MultiModalConvMLP
from visualize import visualize_trajectory, visualize_multimodal_trajectory
from metrics import mse, ade, fde

PAST_STEPS = 10
FUTURE_STEPS = 80
DATA_DIR = "./test_data"

MODELS = [
    #ConvMLP,
    MultiModalConvMLP
]


def visualize(model_to_visualize):
    # Load test dataset
    test_ds = get_data(DATA_DIR, BATCH_SIZE=1, training=False)

    model_name = model_to_visualize.__name__  

    # Load trained model
    model_path = f"trained_models/{model_name}.keras"
    model = tf.keras.models.load_model(model_path)

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
    test_ds = get_data(DATA_DIR, BATCH_SIZE=1, training=False)
    
    model_name = model_to_visualize.__name__
    model_path = f"trained_models/{model_name}.keras"
    model = tf.keras.models.load_model(model_path)
    
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
    #get dataset
    test_ds = get_data(DATA_DIR, BATCH_SIZE=64, training=False)

    #build_model
    model = model_to_test(PAST_STEPS, FUTURE_STEPS)
    model_name = model_to_test.__name__
    model_path = f"trained_models/{model_name}.keras"

    try:
        model = tf.keras.models.load_model(model_path)
    except FileNotFoundError:
        print("File not found")
        return 0

    avg_loss, avg_ade, avg_fde = 0.0, 0.0, 0.0
    num_batches = 0

    for past, future, valid in test_ds:
        pred = model(past, training=True)
        loss = mse(future, pred, valid)
        batch_ade = ade(future, pred, valid)
        batch_fde = fde(future, pred, valid)

        # add batch metrics
        avg_loss += float(loss.numpy())
        avg_ade  += float(batch_ade.numpy())
        avg_fde  += float(batch_fde.numpy())
        num_batches += 1

        print(f"loss {loss:.4f} | ADE {batch_ade:.4f} | FDE {batch_fde:.4f}")
    

    # average over the epoch
    avg_loss /= num_batches
    avg_ade  /= num_batches
    avg_fde  /= num_batches
    print("\n\n")
    print(f"Avg loss {avg_loss:.4f} | Avg ADE {avg_ade:.4f} | Avg FDE {avg_fde:.4f}")


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