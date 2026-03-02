import tensorflow as tf

class ConvMLP(tf.keras.Model):
    def __init__(self, PAST_STEPS, FUTURE_STEPS, activation="relu", **kwargs):
        super().__init__(**kwargs)

        #Conv Layers
        self.conv_layer_1 = tf.keras.layers.Conv1D(64, kernel_size=3, padding="causal", activation=activation)
        self.conv_layer_2 = tf.keras.layers.Conv1D(64, kernel_size=3, padding="causal", activation=activation)

        self.flatten = tf.keras.layers.Flatten()

        #Dense Layers
        self.dense_1 = tf.keras.layers.Dense(128, activation=activation)
        self.dense_2 = tf.keras.layers.Dense(FUTURE_STEPS * 2)

        self.reshape = tf.keras.layers.Reshape((FUTURE_STEPS, 2), name="future_xy")

    def get_config(self):
        config = super().get_config()
        config.update({
            "PAST_STEPS": self.PAST_STEPS,
            "FUTURE_STEPS": self.FUTURE_STEPS,
            "activation": self.activation,
        })
        return config


    def call(self, input):
        """
        Input: (B, 10, 2)
        Output: (B, 80, 2)
        """

        #Convolutional Encoder
        x = self.conv_layer_1(input)
        x = self.conv_layer_2(x)

        x = self.flatten(x)

        #MLP/Dense Decoder
        x = self.dense_1(x)
        x = self.dense_2(x)

        outputs = self.reshape(x)

        return outputs