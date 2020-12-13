import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

"""## Custom sampling layer

"""

class Sample(layers.Layer):    
    def call(self, inputs):
      """
      Uses mean and log variance to the digit vector
      """
      z_mu, z_sig = inputs
      batch = tf.shape(z_mu)[0]
      dim = tf.shape(z_mu)[1]
      epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
      return z_mu + tf.exp(0.5 * z_sig) * epsilon

"""## Encoder

"""

def createEncoder(latent_dim=2):
  inputs = keras.Input(shape=(28, 28, 1))
  prev_layer = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(inputs)
  prev_layer = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(prev_layer)
  prev_layer = layers.Flatten()(prev_layer)
  prev_layer = layers.Dense(16, activation="relu")(prev_layer)
  z_mu = layers.Dense(latent_dim, name="mean")(prev_layer)
  z_sig = layers.Dense(latent_dim, name="log_var")(prev_layer)
  z = Sample()([z_mu, z_sig])
  model = keras.Model(inputs, [z_mu, z_sig, z], name="encoder")
  return model

encoder = createEncoder()
encoder.summary()

"""## Decoder

"""

def createDecoder(latent_dim=2):
  inputs = keras.Input(shape=(latent_dim,))
  prev_layer = layers.Dense(7 * 7 * 64, activation="relu")(inputs)
  prev_layer = layers.Reshape((7, 7, 64))(prev_layer)
  prev_layer = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(prev_layer)
  prev_layer = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(prev_layer)
  outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(prev_layer)
  model = keras.Model(inputs, outputs, name="decoder")
  
  return model

decoder = createDecoder()
decoder.summary()

"""## Define and train VAE

"""

class VAE(keras.Model):
    def __init__(self, encoder, decoder):
        self.encoder = encoder
        self.decoder = decoder

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]

        with tf.GradientTape() as tape:
            z_mu, z_sig, z = self.encoder(data)
            recreate = self.decoder(z)
            recreate_loss = tf.reduce_mean(
                keras.losses.binary_crossentropy(data, recreate)
            ) * 28 * 28

            kl_loss = 1 + z_sig - tf.square(z_mu) - tf.exp(z_sig)
            kl_loss = tf.reduce_mean(kl_loss) * -0.5
 
            loss = reconstruction_loss + kl_loss
 
        gradients = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
        
        return {
            "loss": loss,
            "reconstruction_loss": recreate_loss,
            "kl_loss": kl_loss,
        }

"""## Load the data"""

(x_train, y_train), (x_test, _) = keras.datasets.mnist.load_data()
mnist_digits = np.concatenate([x_train, x_test], axis=0)
mnist_digits = np.expand_dims(mnist_digits, -1).astype("float32") / 255

"""## Declare VAE and Train it"""

vae_model = VAE(encoder, decoder)
vae_model.compile(optimizer=keras.optimizers.Adam())
vae_model.fit(mnist_digits, epochs=30, batch_size=128)

"""## Plot the recreated data

"""

def renderImage(encoder, decoder, n=30, dim=28):
    scale = 2.0
    fig = np.zeros((dim * n, dim * n))
    
    x, y = np.linspace(-scale, scale, n), np.linspace(-scale, scale, n)[::-1]

    for i, yi in enumerate(y):
        for j, xi in enumerate(x):
            sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(sample)
            digit = x_decoded[0].reshape(dim, dim)
            fig[i*dim : (i + 1) * dim, j*dim : (j + 1) * dim,] = digit

    plt.imshow(fig, cmap="gray")
    plt.show()


renderImage(encoder, decoder)

"""## Plot Clusters of the digits

"""

def plotClusters(encoder, decoder, data, labels):
    z_mu, _, _ = encoder.predict(data)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mu[:, 0], z_mu[:, 1], c=labels)
    plt.colorbar()
    plt.show()

plotClusters(encoder, decoder, x_train, y_train)