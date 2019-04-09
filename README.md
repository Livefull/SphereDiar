# SphereDiar: an efficient speaker diarization system for meeting data

To use the system, setup an environment with:

```
Keras >= 2.2.4 
Tensorflow-gpu >= 1.10.1
spherecluster, https://github.com/jasonlaska/spherecluster
Multicore-TSNE, https://github.com/DmitryUlyanov/Multicore-TSNE
skicit-learn
librosa
joblib
```


The system is designed to be run with GPU, atleast following CUDA configuration should work (with Ubuntu 16.04):

```
CUDA version: 9.0
CuDNN version: 7.2.1
GPU: Quadro K2200
```

SphereDiar_demo.ipynb provides system usage instructions.


