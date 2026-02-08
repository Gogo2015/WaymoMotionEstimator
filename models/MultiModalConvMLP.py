import tensorflow as tf
from tensorflow.keras import layers, Model

class MultiModalConvMLP(Model):
    def __init__(self, num_modes=6, past_steps=10, future_steps=80):
        super().__init__()
        self.num_modes = num_modes
        self.future_steps = future_steps
        
        # Encoder
        self.conv1 = layers.Conv1D(64, 3, padding='causal', activation='relu')
        self.conv2 = layers.Conv1D(64, 3, padding='causal', activation='relu')
        self.flatten = layers.Flatten()
        self.shared_fc = layers.Dense(128, activation='relu')
        
        # K separate trajectory prediction heads
        self.trajectory_heads = [
            layers.Dense(future_steps * 2, name=f'mode_{i}')
            for i in range(num_modes)
        ]
        
        # Mode confidence predictor
        self.confidence_head = layers.Dense(num_modes, activation='softmax')
    
    def call(self, inputs, training=False):
        # inputs: (batch, 10, 2)
        
        # Encode past trajectory
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.flatten(x)
        features = self.shared_fc(x)  # (batch, 128)
        
        # Predict K trajectories
        trajectories = []
        for head in self.trajectory_heads:
            traj_flat = head(features)  # (batch, 160)
            traj = tf.reshape(traj_flat, (-1, self.future_steps, 2))  # (batch, 80, 2)
            trajectories.append(traj)
        
        trajectories = tf.stack(trajectories, axis=1)  # (batch, K, 80, 2)
        
        # Predict mode confidences
        confidences = self.confidence_head(features)  # (batch, K)
        
        return trajectories, confidences