import os
import numpy as np
import librosa

DATASET_PATH = "dataset"
SAMPLE_RATE = 22050
N_MFCC = 40


def extract_features(file_path):
    try:
        audio, sr = librosa.load(
            file_path,
            sr=SAMPLE_RATE,
            mono=True
        )

        audio, _ = librosa.effects.trim(audio)

        if len(audio) == 0:
            return None

        features = []

        # ============================================
        # MFCC - 40 MEAN FEATURES
        # ============================================

        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=N_MFCC
        )

        features.extend(
            np.mean(mfcc, axis=1)
        )

        # ============================================
        # MFCC - 40 STANDARD DEVIATION FEATURES
        # ============================================

        features.extend(
            np.std(mfcc, axis=1)
        )

        # ============================================
        # MFCC DELTA - 40 FEATURES
        # ============================================

        delta = librosa.feature.delta(mfcc)

        features.extend(
            np.mean(delta, axis=1)
        )

        # ============================================
        # SPECTRAL CENTROID - 2 FEATURES
        # ============================================

        spectral_centroid = librosa.feature.spectral_centroid(
            y=audio,
            sr=sr
        )

        features.append(
            np.mean(spectral_centroid)
        )

        features.append(
            np.std(spectral_centroid)
        )

        # ============================================
        # SPECTRAL BANDWIDTH - 2 FEATURES
        # ============================================

        spectral_bandwidth = librosa.feature.spectral_bandwidth(
            y=audio,
            sr=sr
        )

        features.append(
            np.mean(spectral_bandwidth)
        )

        features.append(
            np.std(spectral_bandwidth)
        )

        # ============================================
        # SPECTRAL ROLLOFF - 2 FEATURES
        # ============================================

        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=audio,
            sr=sr
        )

        features.append(
            np.mean(spectral_rolloff)
        )

        features.append(
            np.std(spectral_rolloff)
        )

        # ============================================
        # ZERO CROSSING RATE - 2 FEATURES
        # ============================================

        zero_crossing = librosa.feature.zero_crossing_rate(
            audio
        )

        features.append(
            np.mean(zero_crossing)
        )

        features.append(
            np.std(zero_crossing)
        )

        # ============================================
        # RMS ENERGY - 2 FEATURES
        # ============================================

        rms = librosa.feature.rms(
            y=audio
        )

        features.append(
            np.mean(rms)
        )

        features.append(
            np.std(rms)
        )

        # ============================================
        # CHROMA - 12 FEATURES
        # ============================================

        chroma = librosa.feature.chroma_stft(
            y=audio,
            sr=sr
        )

        features.extend(
            np.mean(chroma, axis=1)
        )

        # ============================================
        # TOTAL = 142 FEATURES
        # ============================================

        return np.array(features)

    except Exception as e:

        print(
            f"Error processing {file_path}: {e}"
        )

        return None


# ============================================
# DATASET
# ============================================

X = []
y = []

classes = [
    "gunshot",
    "drone",
    "normal"
]


# ============================================
# READ AUDIO FILES
# ============================================

for label in classes:

    folder = os.path.join(
        DATASET_PATH,
        label
    )

    if not os.path.exists(folder):

        print(
            f"WARNING: Folder not found: {folder}"
        )

        continue

    for filename in os.listdir(folder):

        if filename.lower().endswith(
            (
                ".wav",
                ".mp3",
                ".flac",
                ".ogg"
            )
        ):

            file_path = os.path.join(
                folder,
                filename
            )

            print(
                f"Processing: {label}/{filename}"
            )

            features = extract_features(
                file_path
            )

            if features is not None:

                X.append(features)
                y.append(label)


# ============================================
# CONVERT TO NUMPY ARRAYS
# ============================================

X = np.array(X)
y = np.array(y)


# ============================================
# RESULTS
# ============================================

print()
print("============================================")
print("DATASET PREPARATION COMPLETE")
print("============================================")

print(
    "Number of audio files:",
    len(X)
)

print(
    "Feature shape:",
    X.shape
)

print(
    "Labels shape:",
    y.shape
)

print(
    "Classes:",
    np.unique(y)
)


# ============================================
# CHECK FEATURE COUNT
# ============================================

if len(X) > 0:

    print(
        "Number of features per audio:",
        X.shape[1]
    )

    if X.shape[1] != 142:

        print()
        print(
            "WARNING: Expected 142 features!"
        )

    else:

        print(
            "Feature count check: OK"
        )


# ============================================
# SAVE DATA
# ============================================

np.save(
    "features.npy",
    X
)

np.save(
    "labels.npy",
    y
)


print()
print("Saved:")
print("features.npy")
print("labels.npy")

print("============================================")