from django.urls import path

from . import views

urlpatterns = [
    path('', views.main),
    path('ajax_csv/', views.ajax_csv),
    path('download/', views.file_format_download, name='file_format_download')
]