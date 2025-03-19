import os
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.generics import ListAPIView
from django.db.models import Q
from .serializers import ClaimVerificationSerializer, FalseNewsSerializer, NewsArticleSerializer
from .models import FalseNews
from .services import CombinedAnalysisService, BraveNewsService
from datetime import datetime

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
            analysis_service = CombinedAnalysisService()

            # Get news articles from Brave
            news_articles = brave_service.get_news_articles(article_title)

            # Analyze claim using combined service
            analysis_result = analysis_service.analyze_claim(article_title, news_articles)

            # Create FalseNews object
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

            # Add model scores to response
            response_data = FalseNewsSerializer(false_news).data
            response_data['model_scores'] = analysis_result.get('model_scores', {})

            return Response(response_data, status=status.HTTP_201_CREATED)

        except ValueError as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_400_BAD_REQUEST
            )
        except Exception as e:
            print(f"Error in VerifyClaimAPIView: {str(e)}")
            return Response(
                {'error': 'An unexpected error occurred while processing your request'},
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