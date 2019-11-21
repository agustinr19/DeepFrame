import os

from keras.models import Model
from keras.layers import *
from keras import optimizers
from keras import losses


# emulates DenseSLAMNet from section 5.6 in https://arxiv.org/pdf/1805.06558.pdf
class DenseSLAMNet(object):
    def __init__(self, frame_size):
        a = Input(shape=frame_size)

        # encoding portion
        b = Conv2D(32, kernel_size=(7, 7), strides=(1, 1), activation='relu', padding='same')(a)
        c = Conv2D(32, kernel_size=(7, 7), strides=(2, 2), activation='relu', padding='same')(b)
        d = Conv2D(64, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='same')(c)
        e = Conv2D(64, kernel_size=(5, 5), strides=(2, 2), activation='relu', padding='same')(d)
        f = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(e)
        g = Conv2D(128, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(f)
        h = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(g)
        i = Conv2D(256, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(h)
        j = Conv2D(512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(i)
        k = Conv2D(512, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(j)
        l = Conv2D(512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(k)
        m = Conv2D(512, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(l)
        n = Conv2D(512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(m)
        o = Conv2D(512, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(n)

        # decoding portion
        p = Conv2DTranspose(512, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(o)
        q = Conv2D(512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(p)
        r = Conv2DTranspose(512, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(q)
        s = Conv2D(512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(r)
        t = Conv2DTranspose(256, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(s)
        u = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(t)
        v = Conv2DTranspose(128, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(u)
        w = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(v)
        x = Conv2DTranspose(64, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(w)
        y = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(x)
        z = Conv2DTranspose(32, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(y)
        aa = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(z)
        ab = Conv2DTranspose(16, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(aa)
        ac = Conv2D(16, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(ab)

        # compiles model & optimizer
        self.model = Model(inputs=a, outputs=ac)
        opt = optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
        self.model.compile(loss=losses.mean_squared_error, optimizer=opt)

        self.print()
        self.load()

    def save(self, weights_path="dsn_weights.h5"):
        self.model.save_weights(weights_path)

    def load(self, weights_path="dsn_weights.h5"):
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
        self.save()


# emulates CNN-SINGLE from section 5.6 in https://arxiv.org/pdf/1805.06558.pdf
class CNNSingle(object):
    def __init__(self, frame_size):
        a = Input(shape=frame_size)

        # encoding portion
        b = Conv2D(32, kernel_size=(7, 7), strides=(1, 1), activation='relu', padding='same')(a)
        c = Conv2D(32, kernel_size=(7, 7), strides=(2, 2), activation='relu', padding='same')(b)
        d = Conv2D(64, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='same')(c)
        e = Conv2D(64, kernel_size=(5, 5), strides=(2, 2), activation='relu', padding='same')(d)
        f = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(e)
        g = Conv2D(128, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(f)
        h = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(g)
        i = Conv2D(256, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(h)
        j = Conv2D(512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(i)
        k = Conv2D(512, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(j)
        l = Conv2D(512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(k)
        m = Conv2D(512, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(l)
        n = Conv2D(512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(m)
        o = Conv2D(512, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(n)

        # decoding portion
        p = Conv2DTranspose(512, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(o)
        q = Conv2D(512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(p)
        r = Conv2DTranspose(512, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(q)
        s = Conv2D(512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(r)
        t = Conv2DTranspose(256, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(s)
        u = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(t)
        v = Conv2DTranspose(128, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(u)
        w = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(v)
        x = Conv2DTranspose(64, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(w)
        y = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(x)
        z = Conv2DTranspose(32, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(y)
        aa = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(z)
        ab = Conv2DTranspose(16, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(aa)
        ac = Conv2D(16, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(ab)

        # compiles model & optimizer
        self.model = Model(inputs=a, outputs=ac)
        opt = optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
        self.model.compile(loss=losses.mean_squared_error, optimizer=opt)

        self.print()
        self.load()

    def save(self, weights_path="csingle_weights.h5"):
        self.model.save_weights(weights_path)

    def load(self, weights_path="csingle_weights.h5"):
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
        self.save()


# emulates CNN-SINGLE from section 5.6 in https://arxiv.org/pdf/1805.06558.pdf
class CNNStack(object):
    def __init__(self, frame_size):
        stack_size = (frame_size[0], frame_size[1], frame_size[2]*10)
        a = Input(shape=stack_size)

        # encoding portion
        b = Conv2D(32, kernel_size=(7, 7), strides=(1, 1), activation='relu', padding='same')(a)
        c = Conv2D(32, kernel_size=(7, 7), strides=(2, 2), activation='relu', padding='same')(b)
        d = Conv2D(64, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='same')(c)
        e = Conv2D(64, kernel_size=(5, 5), strides=(2, 2), activation='relu', padding='same')(d)
        f = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(e)
        g = Conv2D(128, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(f)
        h = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(g)
        i = Conv2D(256, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(h)
        j = Conv2D(512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(i)
        k = Conv2D(512, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(j)
        l = Conv2D(512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(k)
        m = Conv2D(512, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(l)
        n = Conv2D(512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(m)
        o = Conv2D(512, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(n)

        # decoding portion
        p = Conv2DTranspose(512, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(o)
        q = Conv2D(512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(p)
        r = Conv2DTranspose(512, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(q)
        s = Conv2D(512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(r)
        t = Conv2DTranspose(256, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(s)
        u = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(t)
        v = Conv2DTranspose(128, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(u)
        w = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(v)
        x = Conv2DTranspose(64, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(w)
        y = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(x)
        z = Conv2DTranspose(32, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(y)
        aa = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(z)
        ab = Conv2DTranspose(16, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(aa)
        ac = Conv2D(16, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(ab)

        # compiles model & optimizer
        self.model = Model(inputs=a, outputs=ac)
        opt = optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
        self.model.compile(loss=losses.mean_squared_error, optimizer=opt)

        self.print()
        self.load()

    def save(self, weights_path="cstack_weights.h5"):
        self.model.save_weights(weights_path)

    def load(self, weights_path="cstack_weights.h5"):
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
        self.save()

