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
        probability_object, classification_object = classify(random_image.image.path)
        probability = probability_object.item() * 100
        classification = classification_object.item()
    context = {
        "random_image": random_image,
        "classification": classification,
        "probability": probability,
        "request": request,
    }
    HTML_STRING = render_to_string("home-view.html", context=context)

    return HttpResponse(HTML_STRING)
