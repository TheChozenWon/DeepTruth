from rest_framework import serializers
from .models import FalseNews

class FalseNewsSerializer(serializers.ModelSerializer):
    class Meta:
        model = FalseNews
        fields = [
            '_id',
            'article_title',
            'veracity',
            'confidence_score',
            'explanation',
            'category',
            'key_findings',
            'impact_level',
            'sources',
            'created_at',
            'updated_at'
        ]
        read_only_fields = ['_id', 'created_at', 'updated_at']

class NewsArticleSerializer(serializers.Serializer):
    title = serializers.CharField()
    link = serializers.URLField()
    snippet = serializers.CharField(required=False)
    source = serializers.CharField()
    time_published = serializers.CharField(required=False)

class ClaimVerificationSerializer(serializers.Serializer):
    article_title = serializers.CharField() 