import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# 1. GENERADORES DE DATOS
def crear_generadores_datos(ruta_base, objetivo=(224, 224), batch_size=32):

    datagen_entrenamiento = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )

    datagen_validacion = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

    gen_entrenamiento = datagen_entrenamiento.flow_from_directory(
        ruta_base,
        target_size=objetivo,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    gen_validacion = datagen_validacion.flow_from_directory(
        ruta_base,
        target_size=objetivo,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    return gen_entrenamiento, gen_validacion


# 2. MODELO CNN
def crear_modelo(input_shape, num_clases):

    modelo = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        MaxPooling2D(2,2),

        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),

        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D(2,2),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_clases, activation='softmax')
    ])

    modelo.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return modelo


# 3. TRANSFER LEARNING
def modelo_transfer(input_shape, num_clases):

    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )

    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(num_clases, activation='softmax')(x)

    modelo = tf.keras.Model(inputs=base_model.input, outputs=output)

    modelo.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return modelo


# 4. ENTRENAMIENTO
def entrenar_modelo(modelo, gen_entrenamiento, gen_validacion, epochs=10):

    historial = modelo.fit(
        gen_entrenamiento,
        validation_data=gen_validacion,
        epochs=epochs
    )

    return historial


# 5. GRAFICAS
def graficar_historial(historial):

    plt.figure()
    plt.plot(historial.history['accuracy'], label='Entrenamiento')
    plt.plot(historial.history['val_accuracy'], label='Validación')
    plt.title('Precisión')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(historial.history['loss'], label='Entrenamiento')
    plt.plot(historial.history['val_loss'], label='Validación')
    plt.title('Pérdida')
    plt.legend()
    plt.show()


# 6. MATRIZ DE CONFUSION
def evaluar_modelo(modelo, gen_validacion):

    predicciones = modelo.predict(gen_validacion)
    y_pred = np.argmax(predicciones, axis=1)
    y_real = gen_validacion.classes

    matriz = confusion_matrix(y_real, y_pred)

    print("\nMatriz de Confusión:")
    print(matriz)

    print("\nReporte de Clasificación:")
    print(classification_report(y_real, y_pred, target_names=gen_validacion.class_indices.keys()))

    plt.figure()
    sns.heatmap(matriz, annot=True, fmt='d')
    plt.title('Matriz de Confusión')
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.show()


# 7. MAIN
if __name__ == "__main__":

    # 🔥 CAMBIA ESTA RUTA
    ruta_dataset = "dataset_animales"

    gen_entrenamiento, gen_validacion = crear_generadores_datos(ruta_dataset)

    print("\nClases:", gen_entrenamiento.class_indices)

    # Elegir modelo:
    # modelo = crear_modelo((224,224,3), gen_entrenamiento.num_classes)
    modelo = modelo_transfer((224,224,3), gen_entrenamiento.num_classes)

    modelo.summary()

    historial = entrenar_modelo(modelo, gen_entrenamiento, gen_validacion, epochs=10)

    graficar_historial(historial)

    evaluar_modelo(modelo, gen_validacion)

    # Guardar modelo
    modelo.save("modelo_animales.h5")

    print("\nModelo guardado como modelo_animales.h5")