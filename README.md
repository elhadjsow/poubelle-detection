# Poubelle Detection WebApp

Détection intelligente de l’état d’une poubelle (pleine ou vide) à partir d’images, avec interface web moderne et modèle YOLOv8.

## Fonctionnalités

- Upload d’images via une interface web élégante
- Prédiction de l’état (pleine/vide) avec affichage de la confiance et de la bounding box
- Entraînement et utilisation d’un modèle YOLOv8 personnalisé
- Visualisation des résultats avec design moderne (glassmorphism, néon, animations)
- Téléchargement du modèle entraîné

## Installation

1. **Cloner le projet**
   ```bash
   git clone <url-du-repo>
   cd dataset_scrape
   ```

2. **Installer les dépendances**
   ```bash
   pip install -r requirements.txt
   ```

3. **Préparer la base de données**
   ```bash
   python manage.py migrate
   ```

4. **Lancer le serveur**
   ```bash
   python manage.py runserver
   ```

## Entraînement du modèle YOLOv8

- Placez vos images et annotations dans les dossiers `train/` et `validation/`
- Configurez `data.yaml` selon votre dataset
- Lancez l’entraînement :
  ```bash
  python train_yolo.py
  ```
- Le modèle sera sauvegardé dans `model/poubelle_yolov8.pt`

## Structure du projet

```
dataset_scrape/
├── detection/           # App Django principale
├── model/               # Modèles entraînés (.pt, .h5)
├── train/               # Images d'entraînement
├── validation/          # Images de validation
├── templates/           # Templates HTML (upload, result)
├── static/              # Fichiers CSS
├── train_yolo.py        # Script d'entraînement YOLOv8
├── data.yaml            # Config YOLOv8
├── manage.py            # Commandes Django
```

## Utilisation

- Accédez à l’URL principale pour uploader une image
- Visualisez le résultat avec la prédiction, la confiance et la bounding box
- Téléchargez le modèle YOLOv8 entraîné

## Technologies

- Python 3.12
- Django 4.2
- Ultralytics YOLOv8
- Bootstrap 5, FontAwesome, CSS moderne

## Auteur

Projet réalisé par elhadj mocktar sow.
