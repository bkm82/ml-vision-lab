from django.contrib.auth.decorators import login_required
from django.shortcuts import render

from .forms import ImageUploadForm


@login_required
def upload_image(request):
    if request.method == "POST":
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image_upload = form.save(commit=False)
            image_upload.user = request.user
            image_upload.save()
            # Add your prediction logic here
    else:
        form = ImageUploadForm()
    return render(request, "images/upload.html", {"form": form})


# Create your views here.
