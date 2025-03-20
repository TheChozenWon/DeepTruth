from django.db import models
from djongo import models as djongo_models
from django.utils import timezone

class FalseNews(djongo_models.Model):
    _id = djongo_models.ObjectIdField()
    article_title = djongo_models.CharField(max_length=500)
    veracity = djongo_models.BooleanField(default=False)
    confidence_score = djongo_models.FloatField()
    explanation = djongo_models.TextField()
    category = djongo_models.CharField(max_length=100)
    key_findings = djongo_models.JSONField()
    impact_level = djongo_models.CharField(
        max_length=20,
        choices=[
            ('VERIFIED', 'Verified'),
            ('MISLEADING', 'Misleading'),
            ('PARTIAL', 'Partial')
        ]
    )
    sources = djongo_models.JSONField()
    created_at = djongo_models.DateTimeField(auto_now_add=True)
    updated_at = djongo_models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'false_news'

class TrueNews(djongo_models.Model):
    _id = djongo_models.ObjectIdField()
    article_title = djongo_models.CharField(max_length=500)
    veracity = djongo_models.BooleanField(default=True)
    confidence_score = djongo_models.FloatField()
    explanation = djongo_models.TextField()
    category = djongo_models.CharField(max_length=100)
    key_findings = djongo_models.JSONField()
    impact_level = djongo_models.CharField(
        max_length=20,
        choices=[
            ('VERIFIED', 'Verified'),
            ('MISLEADING', 'Misleading'),
            ('PARTIAL', 'Partial')
        ]
    )
    sources = djongo_models.JSONField()
    created_at = djongo_models.DateTimeField(auto_now_add=True)
    updated_at = djongo_models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'true_news'

    def __str__(self):
        return f"{self.article_title} - {self.veracity}" 