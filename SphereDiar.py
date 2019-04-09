from sklearn.utils import linear_assignment_
import numpy as np
from spherecluster import SphericalKMeans
from sklearn.metrics import silhouette_score
import os
import glob
from joblib import Parallel, delayed
import multiprocessing
import scipy
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import normalize
from keras.models import *
from keras.layers import *
from librosa.feature import *
from MulticoreTSNE import MulticoreTSNE as TSNE
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
from librosa.util import frame
import warnings

def feature_extractor(s, fs = 16000):
          
   # MFCC
   mfcc_feat = mfcc(s, n_mfcc = 20, sr = fs, n_fft=512, hop_length=160)  
   mfcc_feat = preprocessing.scale(mfcc_feat, axis = 1)

   # Derivatives
   mfcc_d = delta(mfcc_feat, mode = "nearest")
   mfcc_d2 = delta(mfcc_feat, order = 2, mode = "nearest")
   x = np.concatenate([mfcc_feat, mfcc_d, mfcc_d2], axis = 0)

   # Energy removed
   x = np.delete(x, 0, axis = 0)
                             
   return x


def reorganize_lab(emb_labels):
    
    new_labels = np.zeros(len(emb_labels))
    elements, counts = np.unique(emb_labels, return_counts = True)
    indices = np.argsort(-np.abs(counts))
    for i, ind in enumerate(indices):
        emb_ind = np.where(emb_labels == elements[ind])[0]
        new_labels[emb_ind] = np.ones(len(emb_ind))*i
    return new_labels

def silh_score(emb, guess, mode = 0):

        spkmeans = SphericalKMeans(n_clusters=guess, max_iter=300, n_init=1, n_jobs=1).fit(emb)
        emb_labels = spkmeans.labels_
        centers = spkmeans.cluster_centers_
        if mode == 0:
            return silhouette_score(emb, emb_labels, metric = "cosine"), emb_labels, centers
        else:
            return silhouette_score(emb, emb_labels, metric = "cosine")

def DER(ref_labels, labels):

    labels = LabelEncoder().fit_transform(labels)
    ref_labels = LabelEncoder().fit_transform(ref_labels)
    
    # Reorganize
    labels = reorganize_lab(labels)

    # Calculate DER with Hungarian algorithm
    k = np.unique(ref_labels).size
    G = np.zeros((k,k))
    for i in range(k):
        lbl = ref_labels[labels[0:len(ref_labels)] == i]
        elements, counts = np.unique(lbl, return_counts = True)
        for index, element in enumerate(elements):
            G[i, element] = -counts[index]
    A = linear_assignment_.linear_assignment(G)
    acc = 0.0
    for (cluster, best) in A:
        acc -= G[cluster,best]

    return 1-acc / float(len(ref_labels))


def Top2S(embeddings, threshold = 0.10, rounds = 25, 
                         clust_range = [2, 12], num_cores = 1):

    ## STEP 1: Proposal generation
    label_dict = {}
    score_dict = {}
    center_dict = {}

    for i in np.arange(2,clust_range[1]):
        label_dict[i] = 0
        score_dict[i] = 0
        center_dict[i] = 0
        

    for i in np.arange(rounds):
            print("Clustering round: ", i)
            # Creates clustering configurations
            round_configs = Parallel(n_jobs=num_cores)(delayed(silh_score)(embeddings, 
                                                                             K) 
                for K in np.arange(clust_range[0], clust_range[1]))

            # Update cluster centers, silhouette scores and labels
            for i, config in enumerate(round_configs):
                if score_dict[i+clust_range[0]] < config[0]:
                    score_dict[i+clust_range[0]] = config[0]
                    label_dict[i+clust_range[0]] = config[1]
                    center_dict[i+clust_range[0]] = config[2]

    silh_scores = []
    for i in np.arange(2, clust_range[1]):
        silh_scores.append(score_dict[i])

        
    ## STEP 2: Pick best proposal
    silh_ind = np.argsort(-np.array(silh_scores)) 

    K_top_1 = silh_ind[0]+2
    if (silh_ind[1] > silh_ind[0]) and (silh_scores[silh_ind[1]] > threshold):
        labels = label_dict[K_top_1] 
        K_top_2 = silh_ind[1]+2            
    else:
        return K_top_1, center_dict
    
    # Optional inner cluster search
    found_in_clusters = False
    for i in np.arange(rounds):
        if found_in_clusters:
            break
        print("Inner clustering round: ", i)    
 
        for speaker in np.arange(K_top_1):
            speaker_ind = np.where(labels == speaker)[0]
            silh_values = Parallel(n_jobs=num_cores)(delayed(silh_score)(embeddings[speaker_ind], 
                                                                             K, mode = 1) 
                for K in np.arange(clust_range[0], clust_range[1]))

            if (np.argmax(silh_values) <= 2) and (np.max(silh_values) > threshold):
                found_in_clusters = True
                break
        
    if not found_in_clusters:
        K_top_2 = K_top_1

       
    
    return K_top_2, center_dict


# SPHEREDIAR: SPEAKER DIARIZATION SYSTEM

class SphereDiar():
    
    def __init__(self, SS_model):
        self.embeddings_ = []  
        self.speaker_labels_ = []
        self.emb_2d_ = []
        self.X_ = []
        self.centers_ = {}
        self.opt_speaker_num_ = 0
        
        # Exclude softmax layer
        SS = Model(inputs=SS_model.input,
                            outputs=SS_model.layers[-2].output)
        self.SS_ = SS
               
    def extract_features(self, signal, frame_len = 2, hop_len = 0.5, fs = 16000):  
        
        # Frame duration 2s, overlap duration 1.5s, assuming 16 kHz sampling rate
        S = np.transpose(frame(signal, int(frame_len*fs), int(hop_len*fs)))
        
        # 201 sequences of 59 dimensional MFCC based features
        X = list(map(lambda s: feature_extractor(s, fs), S))
        X = np.swapaxes(X, 1, 2)
        self.X_ = X
        return X
        
    def get_embeddings(self, X = []):
               
        if (len(self.X_) == 0) and (len(X) == 0):
            raise RuntimeError("No features available.")

        elif len(X) != 0:
             self.X_ = X
            
        embeddings = self.SS_.predict(self.X_)       
        self.embeddings_ = embeddings
        return embeddings
        
        
    def cluster(self, rounds = 20, clust_range = [2,12], num_cores = 1, threshold = 0.1, embeddings = []):
        
        
        if (len(self.embeddings_) == 0) and (len(embeddings) == 0):
            raise RuntimeError("No speaker embeddings available.")
            
        # If embeddings are not given
        if len(embeddings) == 0:
            embeddings = self.embeddings_
            
        else:
            self.embeddings_ = embeddings
            
                     
        # Top Two Silhouettes
        opt_center_num, center_dict = Top2S(embeddings, clust_range = clust_range, 
                                       rounds = rounds, num_cores = num_cores, threshold = threshold)
        self.centers_ = center_dict
        self.opt_speaker_num_ = opt_center_num   
        
        # Get speaker labels 
        spkmeans = SphericalKMeans(n_clusters=len(center_dict[opt_center_num]), 
                                                   init = center_dict[opt_center_num], 
                                                   max_iter=1, n_init=1, n_jobs=1).fit(embeddings)  
        self.speaker_labels_ = spkmeans.labels_+1 
    
    def visualize(self, indices = [], center_num = 0, 
                  ref_labels = [], use_colors = True):
        
        
        # If indices are not given
        if len(indices) ==0:
            indices = np.arange(len(self.embeddings_))
        
        # If center number is not given
        if center_num == 0:
            center_num = self.opt_speaker_num_
                
        # If reference labels are used
        if len(ref_labels) != 0:
            speaker_labels = ref_labels   
            
        # Allow visualization of different center number configurations
        else:        
            # Get speaker labels 
            spkmeans = SphericalKMeans(n_clusters=len(self.centers_[center_num]), 
                                                       init = self.centers_[center_num], 
                                                       max_iter=1, n_init=1, n_jobs=1).fit(self.embeddings_[indices])  
            speaker_labels = spkmeans.labels_+1 
        
        
        if len(self.speaker_labels_) == 0:
            raise RuntimeError("Clustering not performed.")
                                       
        # Compute TSNE only once
        if len(self.emb_2d_) == 0:
            
            print("Computing TSNE transform...")
            tsne = TSNE(n_jobs=4)
            self.emb_2d_ = tsne.fit_transform(self.embeddings_)
        
        
        # Visualize
        emb_2d = self.emb_2d_[indices]
        speaker_labels = speaker_labels.astype(np.int)
        speakers = np.unique(speaker_labels)
        colors=cm.rainbow(np.linspace(0,1,len(speakers)))
        plt.figure(figsize=(7,7))

        for speaker in speakers:

            speak_ind = np.where(speaker_labels == speaker)[0]
            x, y = np.transpose(emb_2d[speak_ind])
            if use_colors == True:
               plt.scatter(x, y, c="k", edgecolors=colors[speaker-1], s=2,  label=speaker)
            else:
               plt.scatter(x, y, c="k", edgecolors="k", s=2,  label=speaker)

        plt.legend(title = "Speakers", prop={'size': 10})

        if len(ref_labels) == 0:
            plt.title("Predicted speaker clusters")
        else:
            plt.title("Reference speaker clusters")  
        plt.show()

    def calc_DER(self, ref_labels, ref_indices):
        
        labels = self.speaker_labels_[ref_indices]        
        der = DER(ref_labels, labels)
        print("DER (%): ", round(der*100, 3))
        
