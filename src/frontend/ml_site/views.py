"""To render html web pages."""

import random

from django.contrib.auth.decorators import login_required
from django.http import HttpResponse
from django.template.loader import render_to_string
from images.models import ImageUpload

from ml_vision_lab.classifier import classify


@login_required
def home_view(request):
    """
    Take in a request
    Return a response
    """
    user_images = ImageUpload.objects.filter(user=request.user)
    random_image = None
    classification = None
    probability = None
    if user_images.exists():
        random_image = random.choice(user_images)
        probability, classification = classify(random_image.image.path)

    context = {
        "random_image": random_image,
        "classification": classification.item(),
        "probability": probability.item() * 100,
        "request": request,
    }
    HTML_STRING = render_to_string("home-view.html", context=context)

    return HttpResponse(HTML_STRING)
