import os
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.generics import ListAPIView
from django.db.models import Q
from .serializers import ClaimVerificationSerializer, FalseNewsSerializer, NewsArticleSerializer
# from .models import FalseNews, TrueNews
# from .services import CombinedAnalysisService, BraveNewsService, ModelTrainingService
from .models import FalseNews
from .services import BraveNewsService, GeminiService
from datetime import datetime
from django.utils import timezone

# class RetrainModelAPIView(APIView):
#     def post(self, request):
#         """
#         Retrain the DistilBERT model using data from MongoDB
#         """
#         try:
#             training_service = ModelTrainingService()
#             result = training_service.retrain_model()
            
#             if result['success']:
#                 return Response(result, status=status.HTTP_200_OK)
#             else:
#                 return Response(result, status=status.HTTP_400_BAD_REQUEST)
                
#         except Exception as e:
#             return Response({
#                 'success': False,
#                 'message': f'Error during retraining: {str(e)}'
#             }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class VerifyClaimAPIView(APIView):
    def post(self, request):
        article_title = request.data.get('article_title')
        if not article_title:
            return Response(
                {'error': 'Article title is required'},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            # Initialize services
            brave_service = BraveNewsService()
            gemini_service = GeminiService()

            # Get news articles from Brave
            news_articles = brave_service.get_news_articles(article_title)

            # Analyze claim using Gemini
            analysis_result = gemini_service.analyze_claim(article_title, news_articles)

            # Create FalseNews object with only the fields that exist in the model
            false_news = FalseNews.objects.create(
                article_title=article_title,
                veracity=analysis_result.get('veracity', False),
                confidence_score=analysis_result.get('confidence_score', 0.0),
                explanation=analysis_result.get('explanation', ''),
                category=analysis_result.get('category', 'Unknown'),
                key_findings=analysis_result.get('key_findings', []),
                impact_level=analysis_result.get('impact_level', 'PARTIAL'),
                sources=analysis_result.get('sources', [])
            )

            # Serialize and return the response
            serializer = FalseNewsSerializer(false_news)
            return Response(serializer.data, status=status.HTTP_201_CREATED)

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