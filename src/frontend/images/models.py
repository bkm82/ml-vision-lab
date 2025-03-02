from django.contrib.auth.models import User
from django.db import models


def user_directory_path(instance, filename):
    # This will create a path like: MEDIA_ROOT/user_<id>/<filename>
    return f"user_{instance.user.id}/{filename}"


class ImageUpload(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    image = models.ImageField(upload_to=user_directory_path)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Image {self.id} uploaded by {self.user.username} at {self.uploaded_at}"
