import os
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.generics import ListAPIView
from django.db.models import Q
from .serializers import ClaimVerificationSerializer, FalseNewsSerializer, NewsArticleSerializer
from .models import FalseNews, TrueNews
from .services import CombinedAnalysisService, BraveNewsService, ModelTrainingService
from datetime import datetime
from django.utils import timezone

class RetrainModelAPIView(APIView):
    def post(self, request):
        """
        Retrain the DistilBERT model using data from MongoDB
        """
        try:
            training_service = ModelTrainingService()
            result = training_service.retrain_model()
            
            if result['success']:
                return Response(result, status=status.HTTP_200_OK)
            else:
                return Response(result, status=status.HTTP_400_BAD_REQUEST)
                
        except Exception as e:
            return Response({
                'success': False,
                'message': f'Error during retraining: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class VerifyClaimAPIView(APIView):
    def post(self, request):
        try:
            article_title = request.data.get('article_title')
            if not article_title:
                return Response(
                    {'error': 'article_title is required'},
                    status=status.HTTP_400_BAD_REQUEST
                )

            # Get news articles from Brave Search
            combined_service = CombinedAnalysisService()
            result = combined_service.analyze_claim(article_title)

            # Store the result in the appropriate collection based on veracity
            if result['veracity']:
                TrueNews.objects.create(
                    article_title=article_title,
                    veracity=result['veracity'],
                    confidence_score=result['confidence_score'],
                    explanation=result['explanation'],
                    category=result['category'],
                    key_findings=result['key_findings'],
                    impact_level=result['impact_level'],
                    sources=result['sources']
                )
            else:
                FalseNews.objects.create(
                    article_title=article_title,
                    veracity=result['veracity'],
                    confidence_score=result['confidence_score'],
                    explanation=result['explanation'],
                    category=result['category'],
                    key_findings=result['key_findings'],
                    impact_level=result['impact_level'],
                    sources=result['sources']
                )

            return Response(result, status=status.HTTP_200_OK)

        except Exception as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class FalseNewsListView(ListAPIView):
    serializer_class = FalseNewsSerializer
    
    def get_queryset(self):
        queryset = FalseNews.objects.all()
        
        # Filter by category
        category = self.request.query_params.get('category', None)
        if category:
            queryset = queryset.filter(category=category)
        
        # Filter by impact level
        impact = self.request.query_params.get('impact', None)
        if impact:
            queryset = queryset.filter(impact_level=impact)
        
        # Filter by fact check status
        status = self.request.query_params.get('status', None)
        if status:
            queryset = queryset.filter(fact_check_status=status)
        
        # Search by title or explanation
        search = self.request.query_params.get('search', None)
        if search:
            queryset = queryset.filter(
                Q(article_title__icontains=search) |
                Q(explanation__icontains=search) |
                Q(tags__contains=[search.lower()])
            )
        
        return queryset.order_by('-verification_date') 