from django.contrib.auth import authenticate, login, logout
from django.shortcuts import redirect, render


# Create your views here.
def login_view(request):
    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")
        user = authenticate(request, username=username, password=password)
        if user is None:
            context = {
                "error": "Invalid username or password.",
            }
            return render(request, "accounts/login.html", context=context)
        login(request, user)
        return redirect("home")
    return render(request, "accounts/login.html", context={})


def logout_view(request):
    if request.method == "POST":
        logout(request)
        return redirect("login")
    return render(request, "accounts/logout.html", context={})


def register_view(request):
    context = {}
    return render(request, "accounts/logout.html", context=context)
