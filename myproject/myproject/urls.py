from django.urls import path
from django.contrib import admin
from face_emotion_app import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('detect_intent/', views.detect_intent, name='detect_intent'),
]
