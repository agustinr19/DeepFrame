import os

from keras.models import Model
from keras.layers import *
from keras import optimizers
from keras import losses

# emulates DenseSLAMNet from section 5.6 in https://arxiv.org/pdf/1805.06558.pdf
class DenseSLAMNet(object):
    def __init__(self, frame_size):

        a = Input(shape=input_size)

        # encoding portion
        b = Conv2D(18, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='same')(a)
        c = MaxPooling2D(pool_size=(2, 2))(b)
        d = Conv2D(36, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='same')(c)
        e = MaxPooling2D(pool_size=(2, 2))(d)
        f = Conv2D(72, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='same')(e)
        g = MaxPooling2D(pool_size=(2, 2))(f)

        # decoding portion
        h = Conv2D(36, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='same')(g)
        i = UpSampling2D(size=(2, 2))(h)
        j = Conv2D(18+num_bones, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='same')(i)
        k = UpSampling2D(size=(2, 2))(j)
        l = Conv2D(num_bones, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='same')(k)
        m = UpSampling2D(size=(2, 2))(l)

        # compiles model & optimizer
        self.model = Model(inputs=a, outputs=m)
        opt = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss=losses.mean_squared_error, optimizer=opt)

        self.print()

    def save(self, weights_path="weights.h5"):
        self.model.save_weights(weights_path)

    def load(self, weights_path="weights.h5"):
        if os.path.exists(weights_path): # loads previous weights if available
            self.model.load_weights(weights_path)

    def print(self):
        print("LAYER SEQUENCE: ")
        for layer in self.model.layers:
            print(layer.output_shape)

    def run(self, x):
        return self.model.predict(x)

    def train(self, x_train, x_test, y_train, y_test, batch_size=4, epochs=100):
        self.model.fit(
            x=x_train,
            y=y_train,
            batch_size=batch_size, 
            epochs=epochs, 
            validation_data=(x_test, y_test),
            shuffle=True)


