from django.db import models

class UploadedImage(models.Model):
    image = models.ImageField(upload_to='uploads/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    prediction = models.CharField(max_length=20, blank=True, null=True)
    box_x = models.IntegerField(blank=True, null=True)
    box_y = models.IntegerField(blank=True, null=True)
    box_w = models.IntegerField(blank=True, null=True)
    box_h = models.IntegerField(blank=True, null=True)

    def __str__(self):
        return f"Image {self.id} - {self.prediction}"
