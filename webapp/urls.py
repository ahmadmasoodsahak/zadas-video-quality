from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('api/process/', views.process_frame, name='process_frame'),
    path('api/info/', views.info, name='info'),
]
