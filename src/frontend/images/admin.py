from django.contrib import admin

from .models import ImageUpload


@admin.register(ImageUpload)
class ImageUploadAdmin(admin.ModelAdmin):
    list_display = ("id", "user", "uploaded_at", "image")  # Specify the fields to display in the admin list
    search_fields = ("user__username",)  # Allow searching by username


# Register your models here.
