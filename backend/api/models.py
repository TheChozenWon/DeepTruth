from djongo import models
from datetime import datetime

class FalseNews(models.Model):
    _id = models.ObjectIdField()
    article_title = models.CharField(max_length=500)
    veracity = models.BooleanField(default=False)
    confidence_score = models.FloatField()
    explanation = models.TextField()
    category = models.CharField(max_length=100)
    key_findings = models.JSONField(default=list)
    impact_level = models.CharField(
        max_length=20,
        choices=[
            ('VERIFIED', 'Verified'),
            ('MISLEADING', 'Misleading'),
            ('PARTIAL', 'Partial')
        ],
        default='PARTIAL'
    )
    sources = models.JSONField(default=list)  # Store the list of source URLs
    created_at = models.DateTimeField(default=datetime.now)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        db_table = 'false_news'
        
    def __str__(self):
        return f"{self.article_title} - {self.veracity}" 