from flask import Flask, render_template, request
import os
import subprocess
from datetime import datetime
from predict import predict_audio


app = Flask(__name__)

UPLOAD_FOLDER = "uploads"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# Store the latest detection
latest_detection = {
    "result": "No detection yet",
    "confidence": 0,
    "status": "WAITING",
    "threat_level": "UNKNOWN"
}

history = []


# --------------------------------------------------
# HOME
# --------------------------------------------------

@app.route("/")
def home():

    return render_template("index.html")


# --------------------------------------------------
# AUDIO UPLOAD
# --------------------------------------------------

@app.route("/upload", methods=["GET", "POST"])
def upload():

    if request.method == "POST":

        audio_file = request.files.get("audio")

        if not audio_file or not audio_file.filename:

            return "No audio file selected."


        file_path = os.path.join(
            app.config["UPLOAD_FOLDER"],
            audio_file.filename
        )

        audio_file.save(file_path)


        try:

            result, confidence = predict_audio(
                file_path
            )


            result = result.lower()


            if result == "gunshot":

                status = "THREAT DETECTED"

                threat_level = "HIGH"


            elif result == "drone":

                status = "THREAT DETECTED"

                threat_level = "MEDIUM"


            else:

                status = "SAFE"

                threat_level = "LOW"


            # Save latest detection

            latest_detection["result"] = result

            latest_detection["confidence"] = round(
                confidence,
                2
            )

            latest_detection["status"] = status

            latest_detection["threat_level"] = threat_level


            return render_template(
                "result.html",
                result=result,
                confidence=round(confidence, 2),
                status=status
            )


        except Exception as e:

            return f"Error analyzing audio: {e}"


    return render_template("upload.html")


# --------------------------------------------------
# LIVE DETECTION PAGE
# --------------------------------------------------

@app.route("/live")
def live_detection():

    return render_template("live.html")


# --------------------------------------------------
# LIVE AUDIO DETECTION
# --------------------------------------------------

@app.route("/live-detect", methods=["POST"])
def live_detect():

    try:

        audio_file = request.files.get("audio")


        if not audio_file:

            return {
                "error": "No audio received"
            }, 400


        webm_file = os.path.join(
            app.config["UPLOAD_FOLDER"],
            "live_audio.webm"
        )


        wav_file = os.path.join(
            app.config["UPLOAD_FOLDER"],
            "live_audio.wav"
        )


        # Save browser recording

        audio_file.save(webm_file)


        # Convert WebM → WAV

        command = [
            "ffmpeg",
            "-y",
            "-i",
            webm_file,
            "-ar",
            "44100",
            "-ac",
            "1",
            wav_file
        ]


        conversion = subprocess.run(
            command,
            capture_output=True,
            text=True
        )


        if conversion.returncode != 0:

            print(conversion.stderr)

            return {
                "error": "FFmpeg conversion failed"
            }, 500


        # AI prediction

        prediction, confidence = predict_audio(
            wav_file
        )


        prediction = prediction.lower()


        print(
            "Live prediction:",
            prediction,
            confidence
        )


        # Determine threat level

        if prediction == "gunshot":

            status = "THREAT DETECTED"

            threat_level = "HIGH"


        elif prediction == "drone":

            status = "THREAT DETECTED"

            threat_level = "MEDIUM"


        else:

            status = "SAFE"

            threat_level = "LOW"


        # Save latest detection

        latest_detection["result"] = prediction

        latest_detection["confidence"] = round(
            confidence,
            2
        )

        latest_detection["status"] = status

        latest_detection["threat_level"] = threat_level


        return {

            "result": prediction,

            "confidence": round(
                confidence,
                2
            ),

            "status": status,

            "threat_level": threat_level

        }


    except Exception as e:

        print(
            "LIVE DETECTION ERROR:",
            e
        )


        return {

            "error": str(e)

        }, 500


# --------------------------------------------------
# THREAT ANALYSIS
# --------------------------------------------------

@app.route("/threat")
def threat_analysis():

    return render_template(
        "threat.html",
        detection=latest_detection
    )


# --------------------------------------------------
# RUN SERVER
# --------------------------------------------------

if __name__ == "__main__":

    app.run(debug=True)