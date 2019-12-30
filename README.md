# SphereDiar: an efficient speaker diarization system for meeting data

To use the system, setup an environment with:

```
Keras >= 2.2.4 
Tensorflow-gpu = 1.10.1
spherecluster, https://github.com/jasonlaska/spherecluster
Multicore-TSNE, https://github.com/DmitryUlyanov/Multicore-TSNE
scikit-learn
librosa
joblib
wavefile
```


The system is designed to be run with GPU, following CUDA configuration should work (with Ubuntu 16.04):

```
CUDA version: 9.0
CuDNN version: 7.2.1
GPU: Quadro K2200
```

Check demo.ipynb for further usage instructions.

### Speaker verification results with Voxceleb1 test set:

| Model  | Frame length (s) | Distance | EER (%) |
| ------------- |-----| ------| ---- |
| SphereSpeaker  |2| Cosine | 6.2  |
| SphereSpeaker 200 |2| Cosine | 5.2 |
| Current best |2| Cosine | 2.2 |

