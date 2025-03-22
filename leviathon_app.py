from flask import Flask, render_template, request
import os
from classify import classify_image

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return "No file uploaded"

    file = request.files["file"]
    if file.filename == "":
        return "No file selected"

    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)

    label, confidence = classify_image(file_path)

    return render_template("result.html", image=file.filename, label=label, confidence=confidence)

if __name__ == "__main__":
    app.run(debug=True)
