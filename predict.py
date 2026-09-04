import numpy as np
import librosa
import joblib


# ============================================
# SETTINGS
# ============================================

SAMPLE_RATE = 22050
N_MFCC = 40


# ============================================
# LOAD TRAINED MODEL
# ============================================

model = joblib.load(
    "sentra_model.pkl"
)

label_encoder = joblib.load(
    "label_encoder.pkl"
)


# ============================================
# EXTRACT EXACTLY 142 FEATURES
# ============================================

def extract_features(file_path):

    audio, sr = librosa.load(
        file_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    audio, _ = librosa.effects.trim(
        audio
    )

    if len(audio) == 0:

        raise ValueError(
            "Audio file is empty."
        )

    features = []


    # ========================================
    # MFCC - 40
    # ========================================

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=N_MFCC
    )

    features.extend(
        np.mean(mfcc, axis=1)
    )


    # ========================================
    # MFCC STD - 40
    # ========================================

    features.extend(
        np.std(mfcc, axis=1)
    )


    # ========================================
    # MFCC DELTA - 40
    # ========================================

    delta = librosa.feature.delta(
        mfcc
    )

    features.extend(
        np.mean(delta, axis=1)
    )


    # ========================================
    # SPECTRAL CENTROID - 2
    # ========================================

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


    # ========================================
    # SPECTRAL BANDWIDTH - 2
    # ========================================

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


    # ========================================
    # SPECTRAL ROLLOFF - 2
    # ========================================

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


    # ========================================
    # ZERO CROSSING RATE - 2
    # ========================================

    zero_crossing = librosa.feature.zero_crossing_rate(
        audio
    )

    features.append(
        np.mean(zero_crossing)
    )

    features.append(
        np.std(zero_crossing)
    )


    # ========================================
    # RMS ENERGY - 2
    # ========================================

    rms = librosa.feature.rms(
        y=audio
    )

    features.append(
        np.mean(rms)
    )

    features.append(
        np.std(rms)
    )


    # ========================================
    # CHROMA - 12
    # ========================================

    chroma = librosa.feature.chroma_stft(
        y=audio,
        sr=sr
    )

    features.extend(
        np.mean(chroma, axis=1)
    )


    # ========================================
    # CONVERT TO NUMPY
    # ========================================

    features = np.array(
        features
    )


    # ========================================
    # CHECK FEATURE COUNT
    # ========================================

    print(
        "Live audio feature count:",
        len(features)
    )


    if len(features) != 142:

        raise ValueError(
            f"Feature extraction produced "
            f"{len(features)} features instead of 142."
        )


    return features


# ============================================
# PREDICT AUDIO
# ============================================

def predict_audio(file_path):

    features = extract_features(
        file_path
    )


    # Model expects 2D input
    features = features.reshape(
        1,
        -1
    )


    # ========================================
    # PREDICTION
    # ========================================

    prediction = model.predict(
        features
    )


    # Convert number back to class name
    label = label_encoder.inverse_transform(
        prediction
    )[0]


    # ========================================
    # CONFIDENCE
    # ========================================

    probabilities = model.predict_proba(
        features
    )[0]


    confidence = float(
        np.max(probabilities) * 100
    )


    print(
        "Live prediction:",
        label,
        confidence
    )


    return label, confidence


# ============================================
# TEST FROM COMMAND LINE
# ============================================

if __name__ == "__main__":

    import sys


    if len(sys.argv) < 2:

        print(
            "Usage:"
        )

        print(
            "python predict.py audio.wav"
        )

        exit()


    audio_file = sys.argv[1]


    try:

        result, confidence = predict_audio(
            audio_file
        )


        print()
        print(
            "Detected Sound:",
            result.upper()
        )

        print(
            "Confidence:",
            f"{confidence:.2f}%"
        )


    except Exception as e:

        print()
        print(
            "Prediction Error:",
            e
        )