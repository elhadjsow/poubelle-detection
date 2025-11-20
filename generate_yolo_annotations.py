# Script pour générer les annotations YOLO pour chaque image du dataset
# Chaque image reçoit une annotation couvrant tout le cadre (bbox = toute l'image)
# Classe 0 = pleine, Classe 1 = vide
import os
from glob import glob
from PIL import Image

def create_yolo_annotation(image_path, label, output_dir):
    # Ouvre l'image pour obtenir sa taille
    with Image.open(image_path) as img:
        w, h = img.size
    # Bounding box couvrant toute l'image (x_center, y_center, width, height) normalisés
    x_center, y_center, width, height = 0.5, 0.5, 1.0, 1.0
    annotation = f"{label} {x_center} {y_center} {width} {height}\n"
    # Nom du fichier .txt
    base = os.path.splitext(os.path.basename(image_path))[0]
    txt_path = os.path.join(output_dir, base + ".txt")
    with open(txt_path, "w") as f:
        f.write(annotation)

def process_folder(img_dir, label, label_dir):
    os.makedirs(label_dir, exist_ok=True)
    for img_path in glob(os.path.join(img_dir, "*.jpg")):
        create_yolo_annotation(img_path, label, label_dir)

# Dossiers à traiter
datasets = [
    ("train/pleine", 0, "train/pleine"),
    ("train/vide", 1, "train/vide"),
    ("validation/pleine", 0, "validation/pleine"),
    ("validation/vide", 1, "validation/vide"),
]

for img_dir, label, label_dir in datasets:
    process_folder(img_dir, label, label_dir)

print("Annotations YOLO générées pour toutes les images.")
