from __future__ import print_function, division

from keras.datasets import cifar10
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D, Embedding
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from os import path, mkdir, getcwd
import tensorflow as tf
import matplotlib.pyplot as plt
import sys
import numpy as np

sys.path.append(path.join(getcwd(), 'utils'))

# Set rng seeds
seed_no = 123
tf.set_random_seed(seed_no)
np.random.seed(seed_no)

# Resolve gpu errors/memory scaling
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

if not path.exists('images'):
    mkdir('images')


class CGAN():

    def __init__(self):
        # Input shape
        self.img_rows = 32
        self.img_cols = 32
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        self.latent_dim = 200
        self.num_classes = 10

        optimizer = Adam(0.0002, 0.5)
        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()

        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()
        # The generator takes noise as input and generates imgs
        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,))
        img = self.generator([noise, label])

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated image as input and determines validity
        # and the label of that image
        valid = self.discriminator([img, label])

        # The combined model  (stacked generator and discriminator)
        # Trains generator to fool discriminator
        self.combined = Model([noise, label], valid)

        self.combined.compile(loss=['binary_crossentropy'], optimizer=optimizer)

    def build_generator(self):
        noise = Input(shape=(self.latent_dim,))
        x1 = Dense(128 * 8 * 8, activation="relu", input_dim=self.latent_dim)(noise)
        x1 = Reshape((8, 8, 128))(x1)

        label = Input(shape=(1,), dtype='int32')
        label_embedding = Flatten()(Embedding(self.num_classes, 8 * 8)(label))
        x2 = Dense(8 * 8, activation="relu")(label_embedding)
        x2 = Reshape((8, 8, 1))(x2)

        x = Concatenate()([x1, x2])
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization(momentum=0.8)(x)

        x = UpSampling2D()(x)  # Conv2DTranspose
        x = Conv2D(128, kernel_size=3, padding="same")(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization(momentum=0.8)(x)

        x = UpSampling2D()(x)
        x = Conv2D(64, kernel_size=3, padding="same")(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization(momentum=0.8)(x)

        x = Conv2D(self.channels, kernel_size=3, padding="same")(x)
        img = Activation("tanh")(x)

        return Model([noise, label], img)

    def build_discriminator(self):

        img = Input(shape=self.img_shape)

        label = Input(shape=(1,), dtype='int32')
        label_embedding = Flatten()(Embedding(self.num_classes, 32 * 32 * 3)(label))
        x2 = Dense(32 * 32 * 3, activation="relu")(label_embedding)
        x2 = Reshape((32, 32, 3))(x2)
        x = Concatenate()([img, x2])

        x = Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same")(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.25)(x)  # 0.5
        x = Conv2D(64, kernel_size=3, strides=2, padding="same")(x)
        x = ZeroPadding2D(padding=((0, 1), (0, 1)))(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.25)(x)
        x = Conv2D(128, kernel_size=3, strides=2, padding="same")(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.25)(x)
        x = Conv2D(256, kernel_size=3, strides=1, padding="same")(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.25)(x)
        x = Flatten()(x)
        x = Dense(1, activation='sigmoid')(x)

        # model.summary()

        return Model([img, label], x)

    def train(self, epochs, batch_size=128, sample_interval=50):

        # Load the dataset

        (x_train, y_train), (_, _) = cifar10.load_data()

        # Configure input
        x_train = (x_train.astype(np.float32) - 127.5) / 127.5
        y_train = y_train.reshape(-1, 1)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        for epoch in range(epochs):
            # ---------------------
            #  Train Discriminator
            # ---------------------
            # Select a random half batch of images
            idx = np.asarray(range(x_train.shape[0]))
            np.random.shuffle(idx)
            for i in range(int(x_train.shape[0] / batch_size)):
                sub_idx = idx[i * batch_size:(i + 1) * batch_size]
                imgs, labels = x_train[sub_idx], y_train[sub_idx]

                # Sample noise as generator input
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

                # Generate a half batch of new images
                gen_imgs = self.generator.predict([noise, labels])

                # Train the discriminator
                d_loss_real = self.discriminator.train_on_batch([imgs, labels], valid)
                d_loss_fake = self.discriminator.train_on_batch([gen_imgs, labels], fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # ---------------------
                #  Train Generator
                # ---------------------

                # Condition on labels
                sampled_labels = np.random.randint(0, 10, batch_size).reshape(-1, 1)

                # Train the generator
                g_loss = self.combined.train_on_batch([noise, sampled_labels], valid)

                # Plot the progress
                print(
                    "%d (%d) [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, i, d_loss[0], 100 * d_loss[1], g_loss))

            # If at save interval => save generated image samples

            if epoch % sample_interval == 0:
                self.sample_images(epoch)

    def sample_images(self, epoch):

        r, c = 2, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        sampled_labels = np.arange(0, 10).reshape(-1, 1)
        gen_imgs = self.generator.predict([noise, sampled_labels])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, :])
                axs[i, j].title.set_text("SomeText")  # "t.categories[cnt])
                axs[i, j].axis('off')
                cnt += 1
        fig.suptitle('epochs = ' + str(epoch))
        fig.savefig(path.join(getcwd(), 'images', "cifar10_%d_%d.png" % (self.latent_dim, epoch)))
        plt.close()


if __name__ == '__main__':
    cgan = CGAN()
    cgan.train(epochs=101, batch_size=128, sample_interval=1)
