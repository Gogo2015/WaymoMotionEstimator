import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 0 = all, 1 = info, 2 = warning, 3 = error

import argparse
import datetime
import tensorflow as tf
from dataLoader import get_data
from metrics import mse, ade, fde
from losses import multimodal_loss
from config import load_config, save_config

# import models
from models.ConvMLP import ConvMLP
from models.MultiModalConvMLP import MultiModalConvMLP

# Model registry - add new models here
MODEL_REGISTRY = {
    "ConvMLP": ConvMLP,
    "MultiModalConvMLP": MultiModalConvMLP
}

@tf.function
def train_step(model, optimizer, past, future, valid):
    with tf.GradientTape() as tape:
        pred = model(past, training=True)
        loss = mse(future, pred, valid)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    batch_ade = ade(future, pred, valid)
    batch_fde = fde(future, pred, valid)

    return loss, batch_ade, batch_fde

@tf.function
def train_step_multimodal(model, optimizer, past, future, valid):
    with tf.GradientTape() as tape:
        pred_trajectories, confidences = model(past, training=True)
        loss, control_loss, intent_loss = multimodal_loss(future, pred_trajectories, confidences, valid)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return loss, control_loss, intent_loss

def train_model(config):
    """
    Train a model using the provided configuration.

    Args:
        config: Config object with experiment parameters
    """
    # Get model class from registry
    if config.model.name not in MODEL_REGISTRY:
        raise ValueError(f"Model {config.model.name} not found in MODEL_REGISTRY")
    model_class = MODEL_REGISTRY[config.model.name]

    # Check if multimodal
    is_multimodal = (model_class == MultiModalConvMLP)

    #get dataset
    train_ds, val_ds = get_data(
        data_dir=config.data.train_dir,
        batch_size=config.data.batch_size,
        past_steps=config.data.past_steps,
        future_steps=config.data.future_steps,
        train_split=config.data.train_split,
        training=True
    )

    #build_model
    if is_multimodal:
        model = model_class(
            num_modes=config.model.get('num_modes', 6),
            past_steps=config.data.past_steps,
            future_steps=config.data.future_steps
        )
    else:
        model = model_class(config.data.past_steps, config.data.future_steps)

    model.build(input_shape=(None, config.data.past_steps, 2))

    #Initialize for TensorBoard
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_name = f"{config.experiment.name}_{current_time}"
    train_log_dir = os.path.join(config.logging.log_dir, exp_name, "train")
    val_log_dir = os.path.join(config.logging.log_dir, exp_name, "val")

    train_writer = tf.summary.create_file_writer(train_log_dir)
    val_writer = tf.summary.create_file_writer(val_log_dir)

    # Save config with this experiment
    save_config(config, os.path.join(config.logging.log_dir, exp_name))

    optimizer = tf.keras.optimizers.Adam(config.training.learning_rate)

    print(f"\nStarting experiment: {config.experiment.name}")
    print(f"Description: {config.experiment.description}")
    print(f"Model: {config.model.name}")
    if is_multimodal:
        print(f"Num modes: {config.model.get('num_modes', 6)}")
    print(f"Batch size: {config.data.batch_size}, Epochs: {config.training.epochs}\n")

    for epoch in range(config.training.epochs):
        epoch_loss, epoch_metric1, epoch_metric2 = 0.0, 0.0, 0.0
        num_batches = 0

        for past, future, valid in train_ds:
            if is_multimodal:
                batch_loss, batch_control, batch_intent = train_step_multimodal(model, optimizer, past, future, valid)
            else:
                batch_loss, batch_control, batch_intent = train_step(model, optimizer, past, future, valid)

            # add batch metrics
            epoch_loss += float(batch_loss.numpy())
            epoch_metric1 += float(batch_control.numpy())
            epoch_metric2 += float(batch_intent.numpy())
            num_batches += 1

        # average over the epoch
        epoch_loss /= num_batches
        epoch_metric1 /= num_batches
        epoch_metric2 /= num_batches

        # write train metrics
        with train_writer.as_default():
            tf.summary.scalar("loss", epoch_loss, step=epoch)
            if is_multimodal:
                tf.summary.scalar("control_loss", epoch_metric1, step=epoch)
                tf.summary.scalar("intent_loss", epoch_metric2, step=epoch)
            else:
                tf.summary.scalar("ADE", epoch_metric1, step=epoch)
                tf.summary.scalar("FDE", epoch_metric2, step=epoch)

        if is_multimodal:
            print(f"epoch {epoch+1}: total {epoch_loss:.4f} | control {epoch_metric1:.4f} | intent {epoch_metric2:.4f}")
        else:
            print(f"epoch {epoch+1}: loss {epoch_loss:.4f} | ADE {epoch_metric1:.4f} | FDE {epoch_metric2:.4f}")

        val_loss, val_metric1, val_metric2 = 0.0, 0.0, 0.0
        val_batches = 0

        for past, future, valid in val_ds:
            if is_multimodal:
                pred_traj, conf = model(past, training=False)
                loss, control, intent = multimodal_loss(future, pred_traj, conf, valid)
                val_loss += float(loss.numpy())
                val_metric1 += float(control.numpy())
                val_metric2 += float(intent.numpy())
            else:
                pred = model(past, training=False)
                val_loss += float(mse(future, pred, valid).numpy())
                val_metric1 += float(ade(future, pred, valid).numpy())
                val_metric2 += float(fde(future, pred, valid).numpy())

            val_batches += 1

        val_loss /= val_batches
        val_metric1 /= val_batches
        val_metric2 /= val_batches

        with val_writer.as_default():
            tf.summary.scalar("loss", val_loss, step=epoch)
            if is_multimodal:
                tf.summary.scalar("control_loss", val_metric1, step=epoch)
                tf.summary.scalar("intent_loss", val_metric2, step=epoch)
            else:
                tf.summary.scalar("ADE", val_metric1, step=epoch)
                tf.summary.scalar("FDE", val_metric2, step=epoch)

        if is_multimodal:
            print(f"Val: total {val_loss:.4f} | control {val_metric1:.4f} | intent {val_metric2:.4f}")
        else:
            print(f"Val: loss {val_loss:.4f} | ADE {val_metric1:.4f} | FDE {val_metric2:.4f}")

    # Save final model
    os.makedirs(config.logging.save_dir, exist_ok=True)
    save_path = os.path.join(config.logging.save_dir, f"{exp_name}.keras")
    model.save(save_path)
    print(f"\nModel saved to: {save_path}")
    print(f"Experiment logs: {os.path.join(config.logging.log_dir, exp_name)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train trajectory prediction model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config file (default: configs/default.yaml)"
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Train model
    train_model(config)
