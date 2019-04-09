import numpy as np
import os
import xml.etree.ElementTree as ET
import re
import wavefile 
from librosa.util import frame
from sklearn import preprocessing
import glob

# FUNCTIONS


def label_generator(framed_transc, H_perc_limit = 0.65):
    
        labels = np.array(list(map(lambda f_t: frame_label_generator(f_t), 
                                   framed_transc)))       
        return labels


def frame_label_generator(transcript):

    unique_labels = np.unique(transcript, return_counts = True)
    speaker_label = unique_labels[0][np.argmax(unique_labels[1])]
    
        
    # H_% calculation      
    if speaker_label == -1:      
        
        # Simplification
        H_perc = 0
               
    else:
        H_perc = np.max(unique_labels[1])/np.sum(unique_labels[1])
        
    return speaker_label, H_perc

def exclude_word(v, word, bad_words):

         
    # Check if start and end of the word are reasonable
    try:
        start = int(float(v['starttime'])*16000)
        end = int(float(v['endtime'])*16000) 
    except KeyError:
        return True
    except ValueError:
        return True

    if start == end:
        return True
    
    # Include either word type or word itself
    try:
        word_type = v['type']
        if word_type == "laugh":
            word = "type." + word_type
            return word, start, end
        else:
                return True
    except KeyError:
        if word == None:
            return True
        elif word.lower() not in bad_words:
            word = re.sub("[_'!.?]", '', word)
            return word.lower(), start, end
        else:
            return True

def transcript_label_generator(audio_file, paths):
  
    # Audio
    os.chdir(paths[0])
    (rate,sig) = wavefile.load(audio_file)
    
    # Words to be excluded    
    bad_words = [[], ["uh", "huh", "uh-huh", "uh_huh"]]
    bad_commas = [None, ".", ",", "?"]
    
    tc = np.zeros((len(sig[0]), 2))
    

    for j in np.arange(2):

        os.chdir(paths[j+1]) 
        audio_id = audio_file.split(".")[0]
        tc_files = glob.glob(audio_id + "*")
        
            
        for i, file in enumerate(tc_files):
            tree = ET.parse(file)
            root = tree.getroot()

            # Speaker indexing 
            speaker = i+1

            for child in root:
                v = child.attrib
                word = child.text
                                          
                # Determine if word is excluded
                excword = exclude_word(v, word, bad_commas + bad_words[j])
                if excword == True:
                    continue
                else:
                    word = excword[0]
                    start = excword[1]
                    end = excword[2]
                 
                
                # Mark indices with overlap                
                temp_sig = tc[start:end, j]
                ol_indices = np.where(temp_sig != 0)[0]+start
                tc[ol_indices, j] = -1

                # Individual speaker indices
                is_indices = np.where(temp_sig == 0)[0]+start
                tc[is_indices, j] = speaker

    sig = sig[0]
        
    # Initialize final transcriptions
    vad_tc = np.zeros((len(sig)))

    # Intersection of segments with one speaker
    os_indices_words = np.where(tc[:, 0] > 0)[0]
    os_indices_ASR = np.where(tc[:, 1] > 0)[0]
    os_indices = np.intersect1d(os_indices_words, os_indices_ASR)

    # Intersection of segments with multiple speakers
    ms_indices_words = np.where(tc[:, 0] == -1)[0]
    ms_indices_ASR = np.where(tc[:, 1] == -1)[0]
    ms_indices = np.intersect1d(ms_indices_words, ms_indices_ASR)

    # Concatenation + VAD
    vad_tc[os_indices] = tc[os_indices, 0]
    vad_tc[ms_indices] = tc[ms_indices, 0]
    vad_indices = np.where(vad_tc != 0)[0]
    vad_tc = vad_tc[vad_indices]
    sig = sig[vad_indices]

    transcript = vad_tc
          
        
    return sig, transcript


