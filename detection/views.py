from django.shortcuts import render, redirect
from .forms import ImageUploadForm
from .models import UploadedImage
from django.conf import settings
import os
from PIL import Image as PilImage
from ultralytics import YOLO
import torch

MODEL_PATH = os.path.join(settings.BASE_DIR, 'model', 'poubelle_model.h5')


# Ajout pour l'inférence YOLOv8
def predict_image_yolo(img_path):
    # Chargement du modèle YOLOv8 (en cache)
    if not hasattr(predict_image_yolo, 'model'):
        model_path = os.path.join(settings.BASE_DIR, 'model', 'poubelle_yolov8.pt')
        predict_image_yolo.model = YOLO(model_path)
    model = predict_image_yolo.model

    # Inference YOLO
    results = model(img_path)
    boxes = results[0].boxes
    if len(boxes) == 0:
        # Aucun objet détecté
        return (0, 0, 0, 0), 'aucune détection', 0.0
    # On prend la première détection (score le plus élevé)
    box = boxes[0]
    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
    label_id = int(box.cls[0].item())
    score = float(box.conf[0].item())
    label = 'pleine' if label_id == 0 else 'vide'
    # Conversion bbox (x, y, w, h)
    w, h = x2 - x1, y2 - y1
    return (x1, y1, w, h), label, score


def upload_image(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            instance = form.save()
            # Utiliser YOLO pour la prédiction
            box, pred, score = predict_image_yolo(instance.image.path)
            instance.prediction = f"{pred} ({score:.2f})"
            instance.box_x, instance.box_y, instance.box_w, instance.box_h = box
            instance.save()
            return redirect('result', pk=instance.pk)
    else:
        form = ImageUploadForm()
    return render(request, 'detection/upload.html', {'form': form})


def result(request, pk):
    img = UploadedImage.objects.get(pk=pk)
    # Séparer la prédiction et le score pour le template
    pred = img.prediction
    if pred and '(' in pred and pred.endswith(')'):
        pred_label = pred.split(' (')[0]
        pred_score = pred.split(' (')[1][:-1]
    else:
        pred_label = pred
        pred_score = ''
    img.pred = pred_label
    img.score = pred_score
    return render(request, 'detection/result.html', {'img': img})


def download_model(request):
    from django.http import FileResponse
    model_file = MODEL_PATH
    return FileResponse(open(model_file, 'rb'), as_attachment=True, filename='poubelle_model.h5')
