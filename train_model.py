import os
#os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 0 = all, 1 = info, 2 = warning, 3 = error

import argparse
import datetime
import tensorflow as tf
from dataLoader import get_data
from metrics import mse, ade, fde
<<<<<<< HEAD
from config import load_config, save_config
=======
from losses import multimodal_loss
>>>>>>> dc6b94ea0e8349af8a669df5956758b1d8be8fd1

# import models
from models.ConvMLP import ConvMLP
from models.MultiModalConvMLP import MultiModalConvMLP

<<<<<<< HEAD
# Model registry - add new models here
MODEL_REGISTRY = {
    "ConvMLP": ConvMLP
}
=======
LOG_DIR = "logs"

MODELS = [
    #ConvMLP,
    MultiModalConvMLP
]

DATA_DIR = "./training_data"

PAST_STEPS = 10
FUTURE_STEPS = 80
>>>>>>> dc6b94ea0e8349af8a669df5956758b1d8be8fd1

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

<<<<<<< HEAD
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

=======
@tf.function
def train_step_multimodal(model, optimizer, past, future, valid):
    with tf.GradientTape() as tape:
        pred_trajectories, confidences = model(past, training=True)
        loss, control_loss, intent_loss = multimodal_loss(future, pred_trajectories, confidences, valid)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return loss, control_loss, intent_loss

def train_model(model_to_train, BATCH_SIZE=64, epochs=10):
>>>>>>> dc6b94ea0e8349af8a669df5956758b1d8be8fd1
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
<<<<<<< HEAD
    model = model_class(config.data.past_steps, config.data.future_steps)
    model.build(input_shape=(None, config.data.past_steps, 2))
=======
    is_multimodal = (model_to_train == MultiModalConvMLP)
    
    if is_multimodal:
        model = model_to_train(num_modes=6, past_steps=PAST_STEPS, future_steps=FUTURE_STEPS)
    else:
        model = model_to_train(PAST_STEPS, FUTURE_STEPS)
    
    model.build(input_shape=(None, PAST_STEPS, 2))
>>>>>>> dc6b94ea0e8349af8a669df5956758b1d8be8fd1

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
    print(f"Batch size: {config.data.batch_size}, Epochs: {config.training.epochs}\n")

    for epoch in range(config.training.epochs):
        epoch_loss, epoch_ade, epoch_fde = 0.0, 0.0, 0.0
        num_batches = 0

        for past, future, valid in train_ds:
            if is_multimodal:
                batch_loss, batch_ade, batch_fde = train_step_multimodal(model, optimizer, past, future, valid)
            else:
                batch_loss, batch_ade, batch_fde = train_step(model, optimizer, past, future, valid)

            # add batch metrics
            epoch_loss += float(batch_loss.numpy())
            epoch_ade  += float(batch_ade.numpy())
            epoch_fde  += float(batch_fde.numpy())
            num_batches += 1

        # average over the epoch
        epoch_loss /= num_batches
        epoch_ade  /= num_batches
        epoch_fde  /= num_batches

        # write train metrics
        with train_writer.as_default():
            tf.summary.scalar("loss", epoch_loss, step=epoch)
            tf.summary.scalar("ADE", epoch_ade, step=epoch)
            tf.summary.scalar("FDE", epoch_fde, step=epoch)

        if is_multimodal:
            print(f"epoch {epoch+1}: total {epoch_loss:.4f} | control {epoch_ade:.4f} | intent {epoch_fde:.4f}")
        else:
            print(f"epoch {epoch+1}: loss {epoch_loss:.4f} | ADE {epoch_ade:.4f} | FDE {epoch_fde:.4f}")

        val_loss, val_ade, val_fde = 0.0, 0.0, 0.0
        val_batches = 0

        for past, future, valid in val_ds:
            if is_multimodal:
                pred_traj, conf = model(past, training=False)
                loss, control, intent = multimodal_loss(future, pred_traj, conf, valid)
                val_loss += float(loss.numpy())
                val_ade += float(control.numpy())
                val_fde += float(intent.numpy())
            else:
                pred = model(past, training=False)
                val_loss += float(mse(future, pred, valid).numpy())
                val_ade  += float(ade(future, pred, valid).numpy())
                val_fde  += float(fde(future, pred, valid).numpy())
            
            val_batches += 1

        val_loss /= val_batches
        val_ade  /= val_batches
        val_fde  /= val_batches

        with val_writer.as_default():
            tf.summary.scalar("loss", val_loss, step=epoch)
            tf.summary.scalar("ADE", val_ade, step=epoch)
            tf.summary.scalar("FDE", val_fde, step=epoch)

        if is_multimodal:
            print(f"Val: total {val_loss:.4f} | control {val_ade:.4f} | intent {val_fde:.4f}")
        else:
            print(f"Val: loss {val_loss:.4f} | ADE {val_ade:.4f} | FDE {val_fde:.4f}")

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