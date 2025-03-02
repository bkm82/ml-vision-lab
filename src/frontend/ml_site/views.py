"""To render html web pages."""

from django.contrib.auth.decorators import login_required
from django.http import HttpResponse
from django.template.loader import render_to_string


@login_required
def home_view(request):
    """
    Take in a request
    Return a response
    """
    context = {
        "content": "here is some base content",
        "request": request,
    }
    HTML_STRING = render_to_string("home-view.html", context=context)

    return HttpResponse(HTML_STRING)
