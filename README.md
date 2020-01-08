### SphereDiar

This repository is based on the following paper:

```
@inproceedings{kaseva2019spherediar,
  title = {SphereDiar - an efficient speaker diarization system for meeting data},
  author = {Tuomas Kaseva and Aku Rouhe 
            and Mikko Kurimo},
  booktitle = {2019 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU)},
  year = {2019},
}
```

### Getting started

To use the tools in this repository, setup an environment with:

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
Then, check **demo.ipynb** to get the basic idea how to use SphereDiar.py for speaker diarization. 

'''
python embed.py
'''

### Performance of the embedding models with Voxceleb1 test set

| Model  | Aggregation | Distance metric | EER (%) |
| ------------- |-----| ------| ---- |
| SphereSpeaker  |Average| Cosine | 6.2  |
| SphereSpeaker 200 |Average| Cosine | 5.2 |
| Current best |Average| Cosine | 2.2 |

Each of these scores has been calculated the similar way as discussed in the "combining.pdf".

