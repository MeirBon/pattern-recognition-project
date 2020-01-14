from __future__ import print_function, division

from keras.datasets import cifar10
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D, Embedding
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from os import path, mkdir, getcwd
from skimage.transform import resize
import tensorflow as tf
import matplotlib.pyplot as plt
import sys
import numpy as np
from scipy.linalg import sqrtm

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


# calculate frechet inception distance
def calculate_fid(model, images1, images2):
    # calculate activations
    act1 = model.predict(images1)
    act2 = model.predict(images2)
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


class CGAN():
    LATENT_DIM = 200
    NUM_CLASSES = 10

    def __init__(self, id=""):
        # Input shape
        self.trained_epochs = 0
        self.id = id
        self.img_rows = 32
        self.img_cols = 32
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        optimizer = Adam(0.0002, 0.5)
        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()

        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()
        # The generator takes noise as input and generates imgs
        noise = Input(shape=(CGAN.LATENT_DIM,))
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
        noise = Input(shape=(CGAN.LATENT_DIM,))
        x1 = Dense(128 * 8 * 8, activation="relu", input_dim=CGAN.LATENT_DIM)(noise)
        x1 = Reshape((8, 8, 128))(x1)

        label = Input(shape=(1,), dtype='int32')
        label_embedding = Flatten()(Embedding(CGAN.NUM_CLASSES, 8 * 8)(label))
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
        label_embedding = Flatten()(Embedding(CGAN.NUM_CLASSES, 32 * 32 * 3)(label))
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
                noise = np.random.normal(0, 1, (batch_size, CGAN.LATENT_DIM))

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
                print("%d (%d) [D-loss: %f, acc: %.2f%%] [G-loss: %f]" % (self.trained_epochs, i, d_loss[0], 100 * d_loss[1], g_loss))

            self.trained_epochs += 1

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)

    def sample_images(self, epoch):
        r, c = 2, 5
        noise = np.random.normal(0, 1, (r * c, CGAN.LATENT_DIM))
        sampled_labels = np.arange(0, r * c).reshape(-1, 1)
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
        fig.savefig(path.join(getcwd(), 'images', "cgan[{}]_cifar10_%d_%d.png".format(id) % (CGAN.LATENT_DIM, epoch)))
        plt.close()


if __name__ == '__main__':
    FIDS = [[] for i in range(3)]

    compare_FIDs = True
    numgen = 1000
    check_shape = (299, 299, 3)

    print('load cifar10 images')
    (_, _), (images1, _) = cifar10.load_data()
    print('converting images to float32')
    images1 = images1.astype('float32')

    for id in range(3):
        cgan = CGAN(str(id))

        not_improved_since = 0
        best_FID_10 = 99999

        while cgan.trained_epochs < 200:
            cgan.train(epochs=1, batch_size=128, sample_interval=1)

            print('generating {} images'.format(numgen))
            noise = np.random.normal(loc=0, scale=1, size=(numgen, CGAN.LATENT_DIM))
            sampled_labels = np.arange(0, numgen).reshape(-1, 1)
            images2 = cgan.generator.predict([noise, sampled_labels])

            # prepare the inception v3 model
            print('prepare inception v3 model for Frechet Inception Distance')
            model = InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))

            print('resizing images')
            np.random.shuffle(images1)
            test_images = np.asarray([resize(image, check_shape, 0) for image in images1[:numgen]])
            images2 = np.asarray([resize(image, check_shape, 0) for image in ((0.5 * images2 + 0.5) * 255.0)])

            # pre-process images
            print('pre-processing images')
            test_images = preprocess_input(test_images)
            images2 = preprocess_input(images2)

            # calculate fid
            fid = calculate_fid(model, test_images, images2)
            print('Epochs: %i, FID: %.3f' % (cgan.trained_epochs, fid))
            FIDS[id].append([cgan.trained_epochs, fid])

            del images2, noise

            if best_FID_10 > fid:
                not_improved_since = 0
                best_FID_10 = fid
            else:
                not_improved_since += 1

            if not_improved_since >= 10:
                break

    print("FIDs:")
    for num, fids in enumerate(FIDS):
        print('Network (%i):' % num)
        for epochs, fid in fids:
            print("Epochs: %i, FID: %.3f" % (epochs, fid))
