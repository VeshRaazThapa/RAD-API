from django.urls import path

# from rest_framework import routers

from . import views
from django.contrib import admin
# router = routers.SimpleRouter()
# router.register(r'model-prediction', views.get_model_prediction, basename='report-page')
# router.register(r'report', views.ReportPageViewSet, basename="report")
# router.register(r'public-link', views.PubliclinkViewSet, basename="public-link")
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
path('admin/', admin.site.urls),
path('model-prediction/', views.get_model_prediction),
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)



# urlpatterns += router.urls
