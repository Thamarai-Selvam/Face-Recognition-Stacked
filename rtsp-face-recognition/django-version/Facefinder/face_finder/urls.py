""
from django.contrib import admin
from django.urls import path


import face_finder
# from face_finder import views


urlpatterns = [
    path('admin/', admin.site.urls),
    # path('face_finder/',views.requested_url)
]
