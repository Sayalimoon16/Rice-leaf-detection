```python
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import shutil
import gdown

app = Flask(__name__)

# -----------------------------
# Google Drive Model Download
# -----------------------------

MODEL_PATH = "rice_disease_model.h5"

# Apni Google Drive File ID yaha paste karna
FILE_ID = "1A_sCX5ZWMtME8d2ySQ3dt9f1n_JcEgo4"

URL = f"https://drive.google.com/uc?id={FILE_ID}"
# Agar model local me nahi hai to download karo
if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    gdown.download(URL, MODEL_PATH, quiet=False)

print("Loading model...")
model = load_model(MODEL_PATH)
print("Model loaded successfully!")

# -----------------------------
# Class Labels
# -----------------------------

classes = [
    "Brown_Spot",
    "Leaf_Blast",
    "Bacterial_Blight"
]

UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "static"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)


@app.route("/", methods=["GET", "POST"])
def home():

    result = None

    if request.method == "POST":

        file = request.files["file"]

        filepath = os.path.join(
            UPLOAD_FOLDER,
            file.filename
        )

        file.save(filepath)

        shutil.copy(
            filepath,
            os.path.join(
                STATIC_FOLDER,
                "last_upload.jpg"
            )
        )

        # Image preprocessing

        img = image.load_img(
            filepath,
            target_size=(150,150)
        )

        img_array = image.img_to_array(img)

        img_array = np.expand_dims(
            img_array,
            axis=0
        )

        img_array = img_array / 255.0

        prediction = model.predict(img_array)

        class_index = np.argmax(prediction[0])

        result = classes[class_index]

    return render_template(
        "index.html",
        prediction=result
    )


if __name__ == "__main__":
    app.run(debug=True)
```
