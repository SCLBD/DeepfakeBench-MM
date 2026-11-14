# Face-alignment resources

Download these files separately and place them in `preprocess/dlib_tools/` before preprocessing. The resource files are not bundled with the code.

| File | Purpose | Download |
|---|---|---|
| `mmod_human_face_detector.dat` | Dlib CNN face detector, used when the frontal detector finds no face | [Dlib](https://dlib.net/files/mmod_human_face_detector.dat.bz2) |
| `shape_predictor_68_face_landmarks.dat` | Predicts 68 facial landmarks for alignment and cropping | [Dlib](https://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) |
| `20words_mean_face.npy` | Reference landmark coordinates used to align faces | [Lipreading using Temporal Convolutional Networks](https://raw.githubusercontent.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks/master/preprocessing/20words_mean_face.npy) |

For example, from the repository root:

```bash
curl -fL https://dlib.net/files/mmod_human_face_detector.dat.bz2 -o preprocess/dlib_tools/mmod_human_face_detector.dat.bz2
curl -fL https://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 -o preprocess/dlib_tools/shape_predictor_68_face_landmarks.dat.bz2
bzip2 -d preprocess/dlib_tools/mmod_human_face_detector.dat.bz2 preprocess/dlib_tools/shape_predictor_68_face_landmarks.dat.bz2
curl -fL https://raw.githubusercontent.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks/master/preprocessing/20words_mean_face.npy -o preprocess/dlib_tools/20words_mean_face.npy
```

Extract the two `.bz2` archives and retain the filenames shown below:

```text
preprocess/dlib_tools/
  README.md
  mmod_human_face_detector.dat
  shape_predictor_68_face_landmarks.dat
  20words_mean_face.npy
```
