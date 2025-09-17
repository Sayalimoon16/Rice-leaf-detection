from flask import Flask, request, render_template, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import shutil

app = Flask(__name__)

model = load_model("rice_disease_model.h5")

classes = ['Brown_Spot', 'Leaf_Blast', 'Bacterial_Blight']

UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "static"

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    if request.method == "POST":
        file = request.files['file']
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        # Copy last uploaded image to static folder for preview
        shutil.copy(filepath, os.path.join(STATIC_FOLDER, "last_upload.jpg"))

        # Preprocess
        img = image.load_img(filepath, target_size=(150,150))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        prediction = model.predict(img_array)
        class_index = np.argmax(prediction[0])
        result = classes[class_index]

    return render_template("index.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
