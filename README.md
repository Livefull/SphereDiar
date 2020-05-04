### SphereDiar

This repository is based on the following paper:

```
@inproceedings{kaseva2019spherediar,
  title = {SphereDiar - an effective speaker diarization system for meeting data},
  author = {Tuomas Kaseva and Aku Rouhe 
            and Mikko Kurimo},
  booktitle = {2019 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU)},
  year = {2019},
}
```

In addition, the repository also contains an additional speaker embedding model "current_best.h5" which is based on the rejected journal article "combining.pdf". The model is very similar to the SphereSpeaker with the main difference being the use of the NetVLAD layer instead of the average pooling layer. Moreover, the model has been trained with the full Voxceleb2 dataset, with additive margin softmax and using SpecAugment style MFCC augmentation. The results with the model and SphereSpeaker models are presented below:

| Model  | Training set | Test set | Aggregation | Distance metric | EER (%) |
| -------------|------|-------|-----| ------| ---- |
| SphereSpeaker |Voxceleb2 (2000) | Voxceleb1-test | Average| Cosine | 6.2  |
| SphereSpeaker 200 | Voxceleb2 (2000) | Voxceleb1-test | Average| Cosine | 5.2 |
| Current best | Voxceleb2 | Voxceleb1-test | Average| Cosine | 2.2 |

Each of these scores has been calculated the similar way as in the "combining.pdf".


### Getting started
First, setup an environment with:

```
Keras >= 2.2.4 
Tensorflow-gpu >= 1.10.1
spherecluster, https://github.com/jasonlaska/spherecluster
Multicore-TSNE, https://github.com/DmitryUlyanov/Multicore-TSNE
scikit-learn
librosa
joblib
wavefile
```
Then, check **demo.ipynb** to get the basic idea how to use SphereDiar.py for speaker diarization. In order to transform a given audio file to speaker embeddings with "current_best.h5" simply try:

```
python embed.py --signal /path/to/your/wav_file --dest /path/to/your/embedding/directory
```
Notice that the script "embed.py" is here only for demonstration purposes, that is, it can not be used to embed multiple audio files.





