from ultralytics import YOLO
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import shutil
import glob

# Charger le modèle pré-entraîné (YOLOv8n, rapide pour test)
# Script d'entraînement YOLOv8 avec Ultralytics
model = YOLO('yolov8n.pt')

# Entraîner le modèle sur votre dataset
results = model.train(
    data='data.yaml',
    epochs=50,
    imgsz=640,
    batch=8,
    project='runs/train',
    name='poubelle_yolov8',
    exist_ok=True
)

# Afficher le contenu du dossier weights pour diagnostic
weights_dir = os.path.join('runs', 'train', 'poubelle_yolov8', 'weights')
print('Contenu du dossier weights :', glob.glob(os.path.join(weights_dir, '*')))

# Chemin du meilleur modèle entraîné
best_model_path = os.path.join(weights_dir, 'best.pt')
output_model_dir = 'model'
output_model_path = os.path.join(output_model_dir, 'poubelle_yolov8.pt')

# Créer le dossier model s'il n'existe pas
os.makedirs(output_model_dir, exist_ok=True)

# Copier le meilleur modèle dans le dossier model
if os.path.exists(best_model_path):
    shutil.copy(best_model_path, output_model_path)
    print(f"Modèle copié dans {output_model_path}")
else:
    print(f"Erreur : Le fichier {best_model_path} n'existe pas. Vérifiez l'entraînement ou les permissions.")

print("Entraînement terminé. Modèle sauvegardé dans model/poubelle_yolov8.pt")
