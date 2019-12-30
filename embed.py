
# ===============================================
#            Headers
# ===============================================

import argparse
import os
import wavefile
import numpy as np
from keras.models import *
from keras.layers import *
from keras.constraints import *
from keras.engine import InputSpec
import tensorflow as tf
import keras
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras import backend as K
from keras import regularizers
from spherecluster import SphericalKMeans
from scipy.optimize import brentq
from sklearn.metrics import roc_curve
from scipy.interpolate import interp1d
from librosa.util import frame
from librosa.feature import mfcc
from matplotlib import pyplot as plt
from sklearn import preprocessing


# ===============================================
#           Functions
# ===============================================

class VLAD(keras.engine.Layer):

    def __init__(self, k_centers=8, kernel_initializer='glorot_uniform', **kwargs):
        self.k_centers = k_centers
        self.kernel_initializer = initializers.get(kernel_initializer)
        super(VLAD, self).__init__(**kwargs)

    def build(self, input_shape):
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
        super(VLAD, self).build(input_shape)  

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

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.k_centers*input_shape[2])
    
    def get_config(self):
        config = super(VLAD, self).get_config()
        config['k_centers'] = self.k_centers
        config['kernel_initializer'] = initializers.serialize(self.kernel_initializer)
        return config

def amsoftmax_loss(y_true, y_pred, scale = 30, margin = 0.35):
    y_pred = y_pred - y_true*margin
    y_pred = y_pred*scale
    return K.categorical_crossentropy(y_true, y_pred, from_logits = True)

def feature_extractor(s, fs = 16000):
          
   # MFCC
   mfcc_feat = mfcc(s, n_mfcc = 30, sr = fs, n_fft=512, hop_length=160)  
   mfcc_feat = preprocessing.scale(mfcc_feat, axis = 1)                        
   return mfcc_feat


# ===============================================
#           Arguments
# ===============================================

parser = argparse.ArgumentParser()
parser.add_argument("--signal", help="Input signal file", default = "/teamwork/t40511_asr/p/spherediar/data/augmented_wavs/signal.npy")
parser.add_argument("--frame_len", type = int, default = 1, help="Frame length in seconds") # Ensure that the frame length is less or equal to the signal duration
parser.add_argument("--hop_len",  type = int, default = 1, help="Hop length in seconds")
args = parser.parse_args()
sampling_rate = 16000 # For now, only this sampling rate can be used


# ===============================================
#           Feature extraction
# ===============================================

# Get the type of signal (assuming it follows after ".")
file_format = args.signal.split(".")[1]

# Load signal - for now, only works with wav or numpy files
if file_format == "npy":
   signal = np.load(args.signal)
else:
    (rate,sig) = wavefile.load(args.signal)
    signal = sig[0]


# Frame and compute MFCCs
S = np.transpose(frame(signal, int(args.frame_len*sampling_rate), int(args.hop_len*sampling_rate))) 
X = list(map(lambda s: feature_extractor(s, sampling_rate), S)) 
X = np.array(np.swapaxes(X,1, 2))
X = X.astype(np.float16) # Compression to save memory
num_timesteps = X.shape[1]

# ===============================================
#           Embedding extraction
# ===============================================


emb_model = load_model("/teamwork/t40511_asr/p/spherediar/models/current_best.h5", custom_objects={'VLAD': VLAD, 'amsoftmax_loss': amsoftmax_loss})

# Modify input shape if necessary
if num_timesteps != 201:
   emb_model.layers.pop(0)
   new_input = Input(batch_shape=(None,num_timesteps,30)) 
   new_output = emb_model(new_input)
   emb_model = Model(new_input, new_output)

# Create embeddings
emb = emb_model.predict(X)