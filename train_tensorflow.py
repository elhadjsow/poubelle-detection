import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# Utiliser des chemins absolus pour garantir le bon fonctionnement
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
train_dir = os.path.join(BASE_DIR, 'train')
val_dir = os.path.join(BASE_DIR, 'validation')

# Répertoires des données
# train_dir = 'train'
# val_dir = 'validation'

# Paramètres
img_size = (224, 224)
batch_size = 16
epochs = 50  # Augmentation du nombre d'époques pour un meilleur apprentissage

# Préparation des données
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, zoom_range=0.2, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary'
)
val_gen = val_datagen.flow_from_directory(
    val_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary'
)

# Modèle MobileNetV2 + custom head
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=img_size + (3,))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(64, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entraînement
model.fit(
    train_gen,
    epochs=epochs,
    validation_data=val_gen
)

# Sauvegarde du modèle
os.makedirs(os.path.join(BASE_DIR, 'model'), exist_ok=True)
model.save(os.path.join(BASE_DIR, 'model', 'poubelle_model.h5'))

print('Modèle entraîné et sauvegardé dans model/poubelle_model.h5')
