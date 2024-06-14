from django.urls import path,include
from predictions import views
from django.conf import settings    
from django.conf.urls.static import static


urlpatterns = [
    path('',views.predict_view),
    path('predict_view',views.predict_view),
    ]