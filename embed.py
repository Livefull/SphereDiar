

# ===============================================
#            Foreword
# ===============================================

""" This script works as a demonstration how a given audio file can be transformed into speaker embeddings. When working with multiple audio files, 
consider first creating the MFCC files as described in "feature extraction" and then embed the files separately.
"""

# ===============================================
#            Headers
# ===============================================

import argparse
import os
import wavefile
import numpy as np
from keras.models import Model, load_model
from keras.layers import *
from keras.constraints import *
import tensorflow as tf
import keras
from tensorflow.python.keras import backend as K
from librosa.util import frame
from librosa.feature import mfcc
from sklearn import preprocessing
from scipy.optimize import brentq
from sklearn.metrics import roc_curve
from scipy.interpolate import interp1d


# ===============================================
#           Functions
# ===============================================

class VLAD(keras.engine.Layer): 

    """
    NetVLAD implementation by Tuomas Kaseva based on
    the "NetVLAD: CNN architecture for weakly supervised place recognition" paper
    """

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
   mfcc_feat = mfcc(s, n_mfcc = 30, sr = fs, n_fft=512, hop_length=160)  
   mfcc_feat = preprocessing.scale(mfcc_feat, axis = 1)                        
   return mfcc_feat

def EER_calc(cos_dists, labels):
    fpr, tpr, thresholds = roc_curve(labels, cos_dists, pos_label=1)
    EER = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(EER)
    return EER, threshI

# ===============================================
#                   MAIN
# ===============================================

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--signal", help="Input signal file", default = "/teamwork/t40511_asr/p/spherediar/data/augmented_wavs/signal.npy")
    parser.add_argument("--frame_len", type = int, default = 2000, help="Frame length in milliseconds") # Ensure that the frame length is less or equal to the signal duration
    parser.add_argument("--hop_len",  type = int, default = 500, help="Hop length in milliseconds")
    parser.add_argument("--mode",  type = int, default = 1, help="1: Save all embeddings, 0: Save only the average embedding.")
    parser.add_argument("--dest",  default = "/teamwork/t40511_asr/p/spherediar/data/augmented_wavs", help="Directory to save the embeddings")
    parser.add_argument("--model",  default = "/teamwork/t40511_asr/p/spherediar/models/current_best.h5", help="Model file.")
    args = parser.parse_args()


    # ===============================================
    #           Feature extraction
    # ===============================================

    # Get the type of the signal file 
    file_name = args.signal.split("/")[-1]
    file_format = file_name.split(".")[1]

    # Load signal - for now, only works with wav or numpy files
    if file_format == "npy":
        signal = np.load(args.signal)
    else:
        (rate,sig) = wavefile.load(args.signal)
        signal = sig[0]


    # Frame and compute MFCCs
    S = np.transpose(frame(signal, int(args.frame_len*16), int(args.hop_len*16)))  # For now, only 16kHz sampling rate can be used
    X = list(map(lambda s: feature_extractor(s, 16000), S)) 
    X = np.array(np.swapaxes(X,1, 2))
    X = X.astype(np.float16) # Compression to save memory, 16-bit MFCCs have also been used in the training of the current_best.h5
    num_timesteps = X.shape[1]

    # ===============================================
    #           Embedding extraction
    # ===============================================


    emb_model = load_model(args.model, custom_objects={'VLAD': VLAD, 'amsoftmax_loss': amsoftmax_loss})

    # Modify input shape if necessary
    if num_timesteps != 201:
        emb_model.layers.pop(0)
        new_input = Input(batch_shape=(None,num_timesteps,30)) 
        new_output = emb_model(new_input)
        emb_model = Model(new_input, new_output)

    # Create embeddings 
    embs = emb_model.predict(X)

    # ===============================================
    #           Save
    # ===============================================

    # Get the name of the signal file
    sig_name = file_name.split(".")[0]

    if args.mode == 0:
        avg_emb = np.mean(embs, axis = 0)
        avg_emb = avg_emb/np.sqrt(sum(avg_emb**2))
        dest_file = os.path.join(args.dest, "_".join([sig_name, "single_emb", str(args.frame_len), str(args.hop_len)]))
        np.save(dest_file, avg_emb)
    else:
        dest_file = os.path.join(args.dest, "_".join([sig_name, "embs", str(args.frame_len), str(args.hop_len)]))
        np.save(dest_file, embs)




