# Generated by Django 5.1.6 on 2025-03-02 21:19

from django.db import migrations, models

import images.models


class Migration(migrations.Migration):
    dependencies = [
        ("images", "0001_initial"),
    ]

    operations = [
        migrations.CreateModel(
            name="MyModel",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                (
                    "upload",
                    models.ImageField(upload_to=images.models.user_directory_path),
                ),
            ],
        ),
        migrations.DeleteModel(
            name="ImageUpload",
        ),
    ]
