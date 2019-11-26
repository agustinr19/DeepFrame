import os

from keras.models import Model
from keras.layers import *
from keras import optimizers
from keras import losses
import keras.backend as K


# emulates DenseSLAMNet from section 5.6 in https://arxiv.org/pdf/1805.06558.pdf
# must be trained with unshuffled, time-sequenced frames with minibatch size of 1
class DenseSLAMNetSequential(object):
    def __init__(self, frame_size, planned_batch_size=1):
        a = Input(shape=frame_size, batch_shape=(planned_batch_size, frame_size[0], frame_size[1], frame_size[2]))

        # encoding portion
        _, c, _, e, _, g, _, i, _, k, _, m, _, o = encoding_layers(a)

        # decoding portion
        p = Conv2DTranspose(512, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(o)
        cc_a = concatenate([m, p], axis=3)
        q = Conv2D(512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(cc_a)
        r = Conv2DTranspose(512, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(q)
        cc_b = concatenate([k, r], axis=3)
        s = Conv2D(512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(cc_b)
        t = Conv2DTranspose(256, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(s)
        cc_c = concatenate([i, t], axis=3)
        u = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(cc_c)
        v = Conv2DTranspose(128, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(u)
        cc_d = concatenate([g, v], axis=3)
        w = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(cc_d)
        x = Conv2DTranspose(64, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(w)
        cc_e = concatenate([e, x], axis=3)
        ya = Reshape((1, K.int_shape(cc_e)[1], K.int_shape(cc_e)[2], K.int_shape(cc_e)[3]))(cc_e)
        yb = ConvLSTM2D(128, stateful=True, kernel_size=(3, 3), activation='relu', padding='same')(ya)
        yc = Reshape((K.int_shape(cc_e)[1], K.int_shape(cc_e)[2], K.int_shape(cc_e)[3]))(yb)
        z = Conv2DTranspose(32, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(yc)
        cc_f = concatenate([c, z], axis=3)
        aaa = Reshape((1, K.int_shape(cc_f)[1], K.int_shape(cc_f)[2], K.int_shape(cc_f)[3]))(cc_f)
        aab = ConvLSTM2D(64, stateful=True, kernel_size=(3, 3), activation='relu', padding='same')(aaa)
        aac = Reshape((K.int_shape(cc_f)[1], K.int_shape(cc_f)[2], K.int_shape(cc_f)[3]))(aab)
        ab = Conv2DTranspose(16, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(aac)
        aca = Reshape((1, K.int_shape(ab)[1], K.int_shape(ab)[2], K.int_shape(ab)[3]))(ab)
        acb = ConvLSTM2D(16, stateful=True, kernel_size=(3, 3), activation='relu', padding='same')(aca)
        acc = Reshape((K.int_shape(ab)[1], K.int_shape(ab)[2], K.int_shape(ab)[3]))(acb)

        # compiles model & optimizer
        self.model = Model(inputs=a, outputs=acc)
        opt = optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
        self.model.compile(loss=losses.mean_squared_error, optimizer=opt)

        self.print()
        self.load()

    def save(self, weights_path="dsnseq_weights.h5"):
        self.model.save_weights(weights_path)

    def load(self, weights_path="dsnseq_weights.h5"):
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


# emulates DenseSLAMNet from section 5.6 in https://arxiv.org/pdf/1805.06558.pdf
# time-indexed frames are batched together
class DenseSLAMNet(object):
    def __init__(self, frame_size, frame_timespan=10):
        a = Input(shape=(frame_timespan, frame_size[0], frame_size[1], frame_size[2]))

        # encoding portion
        _, c, _, e, _, g, _, i, _, k, _, m, _, o = encoding_layers(a)

        # decoding portion
        p = Conv2DTranspose(512, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(o)
        cc_a = concatenate([m, p], axis=3)
        q = Conv2D(512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(cc_a)
        r = Conv2DTranspose(512, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(q)
        cc_b = concatenate([k, r], axis=3)
        s = Conv2D(512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(cc_b)
        t = Conv2DTranspose(256, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(s)
        cc_c = concatenate([i, t], axis=3)
        u = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(cc_c)
        v = Conv2DTranspose(128, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(u)
        cc_d = concatenate([g, v], axis=3)
        w = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(cc_d)
        x = Conv2DTranspose(64, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(w)
        cc_e = concatenate([e, x], axis=3)
        ya = Reshape((1, K.int_shape(cc_e)[1], K.int_shape(cc_e)[2], K.int_shape(cc_e)[3]))(cc_e)
        yb = ConvLSTM2D(128, stateful=True, kernel_size=(3, 3), activation='relu', padding='same')(ya)
        yc = Reshape((K.int_shape(cc_e)[1], K.int_shape(cc_e)[2], K.int_shape(cc_e)[3]))(yb)
        z = Conv2DTranspose(32, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(yc)
        cc_f = concatenate([c, z], axis=3)
        aaa = Reshape((1, K.int_shape(cc_f)[1], K.int_shape(cc_f)[2], K.int_shape(cc_f)[3]))(cc_f)
        aab = ConvLSTM2D(64, stateful=True, kernel_size=(3, 3), activation='relu', padding='same')(aaa)
        aac = Reshape((K.int_shape(cc_f)[1], K.int_shape(cc_f)[2], K.int_shape(cc_f)[3]))(aab)
        ab = Conv2DTranspose(16, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(aac)
        aca = Reshape((1, K.int_shape(ab)[1], K.int_shape(ab)[2], K.int_shape(ab)[3]))(ab)
        acb = ConvLSTM2D(16, stateful=True, kernel_size=(3, 3), activation='relu', padding='same')(aca)
        acc = Reshape((K.int_shape(ab)[1], K.int_shape(ab)[2], K.int_shape(ab)[3]))(acb)

        # compiles model & optimizer
        self.model = Model(inputs=a, outputs=acc)
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
        _, c, _, e, _, g, _, i, _, k, _, m, _, o = encoding_layers(a)

        # decoding portion
        p = Conv2DTranspose(512, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(o)
        cc_a = concatenate([m, p], axis=3)
        q = Conv2D(512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(cc_a)
        r = Conv2DTranspose(512, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(q)
        cc_b = concatenate([k, r], axis=3)
        s = Conv2D(512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(cc_b)
        t = Conv2DTranspose(256, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(s)
        cc_c = concatenate([i, t], axis=3)
        u = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(cc_c)
        v = Conv2DTranspose(128, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(u)
        cc_d = concatenate([g, v], axis=3)
        w = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(cc_d)
        x = Conv2DTranspose(64, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(w)
        cc_e = concatenate([e, x], axis=3)
        z = Conv2DTranspose(32, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(cc_e)
        cc_f = concatenate([c, z], axis=3)
        ab = Conv2DTranspose(16, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(cc_f)

        # compiles model & optimizer
        self.model = Model(inputs=a, outputs=ab)
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
        _, c, _, e, _, g, _, i, _, k, _, m, _, o = encoding_layers(a)

        # decoding portion
        p = Conv2DTranspose(512, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(o)
        cc_a = concatenate([m, p], axis=3)
        q = Conv2D(512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(cc_a)
        r = Conv2DTranspose(512, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(q)
        cc_b = concatenate([k, r], axis=3)
        s = Conv2D(512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(cc_b)
        t = Conv2DTranspose(256, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(s)
        cc_c = concatenate([i, t], axis=3)
        u = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(cc_c)
        v = Conv2DTranspose(128, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(u)
        cc_d = concatenate([g, v], axis=3)
        w = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(cc_d)
        x = Conv2DTranspose(64, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(w)
        cc_e = concatenate([e, x], axis=3)
        z = Conv2DTranspose(32, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(cc_e)
        cc_f = concatenate([c, z], axis=3)
        ab = Conv2DTranspose(16, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(cc_f)

        # compiles model & optimizer
        self.model = Model(inputs=a, outputs=ab)
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



## helper functions

# builds encoding layers for all network architectures
def encoding_layers(input_layer):
    b = Conv2D(32, kernel_size=(7, 7), strides=(2, 2), activation='relu', padding='same')(input_layer)
    c = Conv2D(32, kernel_size=(7, 7), strides=(1, 1), activation='relu', padding='same')(b)
    d = Conv2D(64, kernel_size=(5, 5), strides=(2, 2), activation='relu', padding='same')(c)
    e = Conv2D(64, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='same')(d)
    f = Conv2D(128, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(e)
    g = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(f)
    h = Conv2D(256, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(g)
    i = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(h)
    j = Conv2D(512, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(i)
    k = Conv2D(512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(j)
    l = Conv2D(512, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(k)
    m = Conv2D(512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(l)
    n = Conv2D(512, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(m)
    o = Conv2D(512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(n)
    return [b, c, d, e, f, g, h, i, j, k, l, m, n, o]


