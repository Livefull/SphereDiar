# SphereDiar: an efficient speaker diarization system for meeting data

To use the system, setup an environment with:

```
Keras >= 2.2.4 
Tensorflow-gpu >= 1.10.1
spherecluster, https://github.com/jasonlaska/spherecluster
Multicore-TSNE, https://github.com/DmitryUlyanov/Multicore-TSNE
scikit-learn
librosa
joblib
```


The system is designed to be run with GPU, atleast following CUDA configuration should work (with Ubuntu 16.04):

```
CUDA version: 9.0
CuDNN version: 7.2.1
GPU: Quadro K2200
```

Check demo.ipynb for further usage instructions.

### Speaker verification results with Voxceleb1 test set:


| Model  | EER (%) |
| ------------- | ------|
| SphereSpeaker  | 6.2  |
| SphereSpeaker 200  | 5.2 |

Current best score with this set is 3.2 % https://arxiv.org/pdf/1902.10107.pdf. The results in the table were obtained by creating a speaker embedding for every 2s frame with 1.5s overlap and then choosing a cluster center for the embeddings using spherical K-means. 



