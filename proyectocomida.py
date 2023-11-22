from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from pathlib import Path
import shutil

app = FastAPI()

# Configuración de CORS para permitir solicitudes de cualquier origen
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite todas las origenes
    allow_credentials=True,
    allow_methods=["*"],  # Permite todos los métodos
    allow_headers=["*"],  # Permite todos los headers
)

# Cargar el modelo
loaded_model = tf.keras.models.load_model('C:/xampp/htdocs/dashboard/wempo/wempo/modelo')
class_names = ['Tizon', 'Roya Comun', 'Mancha gris', 'Maiz feliz']

# Directorio de carga de archivos
upload_folder = Path('uploads')
upload_folder.mkdir(exist_ok=True)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Guardar el archivo temporalmente
    file_path = upload_folder / file.filename
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Realizar la predicción
    img = tf.keras.preprocessing.image.load_img(file_path, target_size=(256, 256))
    predicted_class, confidence = predict_image(img)

    # Eliminar el archivo
    file_path.unlink()

    return {"predicted_class": predicted_class, "confidence": confidence}

def predict_image(img):
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Crear un lote
    predictions = loaded_model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * np.max(predictions[0]), 2)

    return predicted_class, confidence

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
