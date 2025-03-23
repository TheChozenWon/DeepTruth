from django.urls import path
# from .views import VerifyClaimAPIView, FalseNewsListView, RetrainModelAPIView
from .views import VerifyClaimAPIView
urlpatterns = [
    path('verify-claim/', VerifyClaimAPIView.as_view(), name='verify-claim'),
    # path('false-news/', FalseNewsListView.as_view(), name='false-news'),
    # path('retrain-model/', RetrainModelAPIView.as_view(), name='retrain-model'),
] 