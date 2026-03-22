# 🐾 Clasificación de Imágenes de Animales con CNN

Este proyecto implementa un modelo de **Deep Learning** utilizando redes neuronales convolucionales (CNN) para clasificar diferentes especies de animales.

---

## 🚀 ¿Qué hace este proyecto?

- Clasifica imágenes de animales (perro, gato, caballo, elefante, etc.)
- Utiliza **TensorFlow/Keras**
- Incluye **data augmentation**
- Permite usar **Transfer Learning (MobileNetV2)**

---

## 📁 Estructura del proyecto


modelo_animales/
│
├── crear_dataset.py
├── modelo_animales.py
├── dataset_animales/
│ ├── perro/
│ ├── gato/
│ ├── caballo/
│ └── elefante/


---

## ⚙️ Instalación

Instalar dependencias:

```bash
python -m pip install tensorflow-cpu matplotlib seaborn scikit-learn numpy
🧠 Uso
1️⃣ Crear dataset

Primero ejecuta:

python crear_dataset.py

👉 Esto generará automáticamente un dataset de animales con imágenes descargadas.

2️⃣ Entrenar el modelo

Después ejecuta:

python modelo_animales.py

👉 El modelo:

Entrena la red neuronal
Muestra métricas
Genera gráficas
Guarda el modelo como modelo_animales.h5
📊 Resultados

El modelo puede alcanzar alta precisión dependiendo del tamaño del dataset.
Con datasets pequeños puede presentarse overfitting.

🧪 Tecnologías utilizadas
Python
TensorFlow / Keras
OpenCV
Matplotlib
Seaborn
💡 Notas
Se recomienda usar al menos 100 imágenes por clase
Se puede mejorar el rendimiento usando GPU
Se puede extender a más clases de animales
👩‍💻 Autor

NWRC - Wen Rocha
