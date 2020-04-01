# Headers

import numpy as np
import os
import glob
import time
from keras.models import *
from keras.layers import *
from keras.callbacks import *
from AM_emb_models import *
from keras.utils import multi_gpu_model
from sklearn import preprocessing
from loss_criteria import calculate_EER_2sec

'''
# For parallel GPU computing
gpu_ids = input("GPU IDS: ")
os.environ["CUDA_VISIBLE_DEVICES"]=gpu_ids
gpu_ids = gpu_ids.split(",")
gpu_ids = [ int(x) for x in gpu_ids]
'''

gpu_ids = [1]


# ===============================================
#            Configuration
# ===============================================

# Choose model
print("Available models: ")
os.chdir("/teamwork/t40511_asr/p/spherediar/models")
model_files = os.listdir()
model_files = sorted(model_files)
for index, model_file in enumerate(model_files):
    print(model_file + ": ", index)
model_index = int(input("Choose model file (index): "))
chosen_model_file = model_files[model_index]

# Sanity check
print("You chose: ", chosen_model_file)
chosen_model_file = "/teamwork/t40511_asr/p/spherediar/models/" + chosen_model_file

# Parameters
cont = chosen_model_file.split(".")
num_speak = int(cont[2])
batch_size = 512
emb_dim = int(cont[1])
dataset_num = int(cont[4])
max_mask_freqs = int(cont[5])
scale = int(input("Scale for AM-training (default=25): "))
margin = int(input("Margin for AM-training (default=15): "))
margin = margin/100
epochs = 20


# Dataset configurations - Voxceleb2 and CSLU
config_dicts = [np.load("/teamwork/t40511_asr/p/spherediar/scripts/config_VC2_mfcc_mvnorm_cleaned.npy", allow_pickle = True).item(),
    np.load("/teamwork/t40511_asr/p/spherediar/scripts/config_CSLU_kids_mfcc.npy", allow_pickle = True).item()]

# Feature file paths
feat_paths = ["/teamwork/t40511_asr/p/spherediar/data/VC2_mfcc_mvnorm_cleaned/", "/teamwork/t40511_asr/p/spherediar/data/CSLU_kids_mfcc/"]

# Dimensions of the data, 30 MFCCs extracted every 10 ms from a 2 second duration frame - fixed for now
dimensions = [30, 201]


# ==============================================
#            Model name and terminal ID
# ==============================================

model = load_model(chosen_model_file, custom_objects={'VLAD': VLAD, 'amsoftmax_loss': amsoftmax_loss})
model_name =chosen_model_file.split("/")[-1]
name = ".".join([model_name, "AM", str(scale), str(margin), str(max_mask_freqs)])
print('\33]0;{terminal_title}\a'.format(terminal_title=name), end='', flush=True)

# ===============================================
#            Functions
# ===============================================

def amsoftmax_loss(y_true, y_pred, scale = scale, margin = margin):
    y_pred = y_pred - y_true*margin
    y_pred = y_pred*scale
    return K.categorical_crossentropy(y_true, y_pred, from_logits = True)


def feature_modifier(features, normalize = [True, True], deltas = False): # Some old stuff, not really useful
    
    features = features.astype(np.float64)
    features = preprocessing.scale(features, with_mean=normalize[0], with_std=normalize[1], axis = 1)
    if deltas == True:
        d = delta(mfcc_feat, mode = "nearest")
        dd = delta(mfcc_feat, order = 2, mode = "nearest")
        features = np.concatenate([features, d, dd], axis = 0)
    return features.astype(np.float16)
    
def load_data(feat_paths, config_dicts, dimensions, dataset_num, num_speak):

    # Get dataset configurations
    if dataset_num == 2: # Use both Voxceleb2 and CSLU
       config_dict = config_dicts[0].copy() 
       config_dict.update(config_dicts[1])     
    else:
        config_dict = config_dicts[dataset_num]
    speaker_ids = list(config_dict.keys())
    
    # Get size of the training set
    train_shape = 0
    sample_sizes = []
    for speaker_id in speaker_ids[0:num_speak]:
        sample_size = int(config_dict[speaker_id])
        train_shape += sample_size
        sample_sizes.append(sample_size)
    
    # Allocate memory    
    X_train = np.zeros((train_shape, dimensions[0], dimensions[1]), dtype = np.float16)
    y_train = np.zeros((train_shape, num_speak), dtype = np.int16)

    # Get feature files
    feat_files = []
    feat_file_ids = []
    for feat_path in feat_paths:
        os.chdir(feat_path)
        files = glob.glob(feat_path + "*mfcc*")
        for file in files:
            curr_speaker_id = file.split(".")[0].split("/")[-1] 
            if curr_speaker_id in speaker_ids[0:num_speak]:
                feat_files.append(file)
                feat_file_ids.append(curr_speaker_id)
    feat_file_ids = np.array(feat_file_ids)

    # Fill arrays
    slot_inst_tr = 0
    for index, speaker_id in enumerate(speaker_ids[0:num_speak]):

        if index % 500 == 0: # verbose
           print(index)
        speaker_id = speaker_id
                
        # Load corresponding feat file
        ind = np.where(feat_file_ids == speaker_id)[0][0]
        print(ind)
        feats = np.load(feat_files[ind]) 
        temp_tr_ind = np.arange(slot_inst_tr, slot_inst_tr + sample_sizes[index])
        
        # Data
        X_train[temp_tr_ind, :, :] = feats
        
        # One hot encoding
        y_train[temp_tr_ind, index] = 1      
        slot_inst_tr += sample_sizes[index]
        
                    
    return X_train, y_train


def data_generator(X, y, batch_size, max_mask_freqs = max_mask_freqs):
    
    # Initial
    ind = np.arange(len(X), dtype = np.int32)
    number_of_batches = np.ceil(len(X)/batch_size).astype(np.int32)
    freq_range = np.arange(1,max_mask_freqs+1)
    mfcc_range = np.arange(30)
    mask_choices = [0,1]
    

    # Shuffle
    np.random.shuffle(ind)

    # Feed batch
    counter = 0
    while True:
        batch_ind = ind[batch_size*counter:batch_size*(counter+1)]
        X_batch = X[batch_ind]
        
        # If masking
        if len(freq_range) != 0:
            masks = np.random.choice(mask_choices, batch_size, p = [0.1, 0.9])
            num_freqs = np.random.choice(freq_range, batch_size)
            for i, sample in enumerate(X_batch):
                
                # Choose whether to mask or not           
                if masks[i]:
                    lines = tuple(np.random.choice(mfcc_range, num_freqs[i], replace = False))
                    X_batch[i, :, lines] = 0
            
        y_batch = y[batch_ind]
        counter += 1
        yield (X_batch,y_batch)

        # Restart counter to yield data in the next epoch as well
        if counter >= number_of_batches:
            counter = 0
            np.random.shuffle(ind)

# ===============================================
#            Data loading
# ===============================================

# Training data
print("Loading data...")
X_train, y_train = load_data(feat_paths, config_dicts, dimensions, dataset_num, num_speak) 
print(np.shape(X_train))
print("Data loaded!")

# Verification pairs from Voxceleb1 dataset for validation - 2 second duration pairs
print("Preprocessing verification pairs...")  
pair_root = "/teamwork/t40511_asr/p/spherediar/data/vc_test_pairs/"
mfcc_pairs = np.load(pair_root + "mfcc_test_pairs_2sec.npy", allow_pickle = True)
mfcc_pairs = mfcc_pairs.item()
compositions = np.load(pair_root + "2sec_compositions.npy") # Could be done smarter
pairs = np.load(pair_root + "2sec_pairs.npy")
labels = np.load(pair_root + "2sec_labels.npy")

# Concatenation
MFCCs = np.zeros((np.sum(compositions), dimensions[1], dimensions[0]), dtype = np.float16)
current_index = 0
indices = []
for i in np.arange(1211):
    mfccs = mfcc_pairs[i]
    shape = len(mfccs)
    MFCCs[current_index:current_index + shape] = mfccs
    current_index = current_index + shape

# ===============================================
#            Model training
# ===============================================


# Check if model is already in parallel form
if len(model.layers) < 10:
       model = model.layers[-2]
       emb_model = Model(inputs=model.get_input_at(0),outputs=model.layers[-2].output) 
else:
       emb_model = Model(inputs=model.input,
                            outputs=model.layers[-2].output)

# New output layer
output_layer = Sequential()
output_layer.add(Dense(num_speak, input_dim = emb_dim, bias = False, kernel_constraint = UnitNorm(axis = 0)))

# New model
new_model =  Model(inputs=emb_model.input, outputs=output_layer(emb_model.output))
W = model.layers[-1].get_weights()
new_model.layers[-1].set_weights(np.expand_dims(W[0], axis=0))


# Handle parallel model configuration
if len(gpu_ids) > 1:
   parallel_model = multi_gpu_model(new_model, gpus=len(gpu_ids))
   model = parallel_model
else:
   model = new_model

# Compile
model.compile(loss=amsoftmax_loss, optimizer='adam', metrics=['accuracy'])
best_EER = 1
for epoch in np.arange(epochs):

    # Train   
    history = model.fit_generator(data_generator(np.swapaxes(X_train, 1, 2), y_train,
          batch_size=batch_size),
          epochs=1, steps_per_epoch = np.ceil(len(X_train)/batch_size).astype(np.int32))

    # Evaluate
    if len(gpu_ids) > 1:
       single_model = model.layers[-2]
       emb_model = Model(inputs=single_model.get_input_at(0),outputs=single_model.layers[-2].output) 

    else:
       emb_model = Model(inputs=model.input,
                            outputs=model.layers[-2].output)
    EER = calculate_EER_2sec(MFCCs, pairs, labels, emb_model)

    # Save
    if EER < 1: # EER < best_EER
       print("Model saved...")
       model.save("/teamwork/t40511_asr/p/spherediar/models/" + name + "_EPOCH_" + str(epoch))
       best_EER = EER
    
    print("EER in epoch " + str(epoch) + ": ", EER)
