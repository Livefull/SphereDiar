
# Headers

from keras.models import *
from keras.layers import *
from keras.constraints import *
from keras.engine import InputSpec
import tensorflow as tf
import keras
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras import backend as K
from keras import regularizers


def amsoftmax_loss(y_true, y_pred, scale = 30, margin = 0.35):
    y_pred = y_pred - y_true*margin
    y_pred = y_pred*scale
    return K.categorical_crossentropy(y_true, y_pred, from_logits = True)


def SphereSpeaker(dimensions = [59, 201], num_speak = 2500, emb_dim=512):

    input_feat = Input(shape=(dimensions[1],dimensions[0]))


    # Bidirectional layers
    x_1 = Bidirectional(CuDNNLSTM(250, return_sequences = True))(input_feat)
    x_2 = Bidirectional(CuDNNLSTM(250, return_sequences = True))(x_1)
    x_3 = Bidirectional(CuDNNLSTM(250, return_sequences = True))(x_2)

    x_conc = Concatenate(axis=2)([x_1, x_2, x_3])
    emb = BatchNormalization()(x_conc)
    emb = Dense(emb_dim, activation = "relu")(emb)
    emb = GlobalAveragePooling1D()(emb)
    emb = BatchNormalization()(emb)
    emb = Lambda(lambda  x: K.l2_normalize(x,axis=1))(emb)

    # Softmax layer
    softmax = Dense(num_speak, activation = "softmax")(emb)

    test_model = Model(inputs=input_feat, outputs=softmax)
    return test_model

class VLAD(keras.engine.Layer):

    def __init__(self, k_centers=8, kernel_initializer='glorot_uniform', **kwargs):
        self.k_centers = k_centers
        self.kernel_initializer = initializers.get(kernel_initializer)
        super(VLAD, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.w = self.add_weight(name='kernel',
                                      shape=(input_shape[2], self.k_centers),
                                      initializer=self.kernel_initializer,
                                      trainable=True)
        self.b = self.add_weight(name='kernel',
                                      shape=(self.k_centers, ),
                                      initializer=self.kernel_initializer,
                                      trainable=True)
        self.c = self.add_weight(name='kernel',
                                      shape=(input_shape[2], self.k_centers),
                                      initializer=self.kernel_initializer,
                                      trainable=True)

        super(VLAD, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):

        Wx_b = K.dot(x, self.w)+self.b
        a = tf.nn.softmax(Wx_b)

        rows = []

        for k in range(self.k_centers):
            error = x-self.c[:, k]

            row = K.batch_dot(a[:, :, k],error)
            row = tf.nn.l2_normalize(row,dim=1)
            rows.append(row)

        output = tf.stack(rows)
        output = tf.transpose(output, perm = [1, 0, 2])
        output = tf.reshape(output, [tf.shape(output)[0], tf.shape(output)[1]*tf.shape(output)[2]])

        return output

def SS_VLAD_best(dimensions = [59, 201], num_speak = 2000, emb_dim = 64, clusters = 14):

    input_feat = Input(shape=(dimensions[1],dimensions[0]))


   # Bidirectional layers
    x_1 = Bidirectional(CuDNNLSTM(200, return_sequences = True))(input_feat)
    x_2 = Bidirectional(CuDNNLSTM(200, return_sequences = True))(x_1)
    x_3 = Bidirectional(CuDNNLSTM(200, return_sequences = True))(x_2)

    x_conc = Concatenate(axis=2)([x_1, x_2, x_3])
    emb = TimeDistributed(Dense(256, activation = "relu"))(x_conc)
    emb = BatchNormalization()(emb)

    # Embedding layer
    emb = VLAD(k_centers = clusters)(emb)
    emb = BatchNormalization()(emb)
    emb = Dense(emb_dim, activation = "relu")(emb)
    emb = BatchNormalization()(emb)
    emb = Lambda(lambda  x: K.l2_normalize(x,axis=1))(emb)

    # Softmax layer
    softmax = Dense(num_speak, activation = "softmax")(emb)

    test_model = Model(inputs=input_feat, outputs=softmax)
    return test_model







