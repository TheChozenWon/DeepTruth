import requests
from bs4 import BeautifulSoup
import json
from datetime import datetime
import os
from django.conf import settings
from typing import List, Dict, Any
import time
import google.generativeai as genai
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import numpy as np

class BraveNewsService:
    def __init__(self):
        self.api_key = os.getenv('BRAVE_API_KEY')
        if not self.api_key:
            raise ValueError("BRAVE_API_KEY is not set in environment variables")

    def get_news_articles(self, query: str, limit: int = 5) -> List[Dict[str, str]]:
        headers = {
            'Accept': 'application/json',
            'Accept-Encoding': 'gzip',
            'X-Subscription-Token': self.api_key
        }
        
        params = {
            'q': query
        }
        
        try:
            # Add delay to prevent rate limiting
            time.sleep(1)
            
            response = requests.get(
                'https://api.search.brave.com/res/v1/web/search',
                headers=headers,
                params=params,
                timeout=10
            )
            response.raise_for_status()
            
            response_data = response.json()
            
            brave_news = []
            if 'web' in response_data and 'results' in response_data['web']:
                results = response_data['web']['results']
                
                for result in results[:limit]:
                    try:
                        article = {
                            'title': result.get('title', ''),
                            'link': result.get('url', ''),
                            'snippet': result.get('description', ''),
                            'source': result.get('site_name', 'Unknown Source'),
                            'time_published': result.get('published_date', 'Unknown Date'),
                            'retrieved_at': datetime.now().isoformat()
                        }
                        brave_news.append(article)
                    except Exception as e:
                        print(f"Error parsing individual result: {e}")
                        continue
            
            # If we couldn't find any valid results, return a default response
            if not brave_news:
                return [{
                    'title': 'No results found',
                    'link': '',
                    'snippet': 'Could not find relevant news articles',
                    'source': 'System',
                    'time_published': datetime.now().isoformat(),
                    'retrieved_at': datetime.now().isoformat()
                }]
            
            return brave_news
            
        except Exception as e:
            print(f"Error fetching news from Brave API: {e}")
            # Return a default response in case of error
            return [{
                'title': 'Error fetching news',
                'link': '',
                'snippet': f'An error occurred while fetching news: {str(e)}',
                'source': 'System',
                'time_published': datetime.now().isoformat(),
                'retrieved_at': datetime.now().isoformat()
            }]

class DistilBERTService:
    def __init__(self):
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
        self.model.eval()  # Set to evaluation mode

    def analyze_claim(self, title: str) -> float:
        """
        Analyze the claim using DistilBERT model
        Returns confidence score between 0 and 1
        """
        try:
            # Tokenize the input
            inputs = self.tokenizer(title, 
                                  return_tensors="pt", 
                                  truncation=True, 
                                  max_length=512,
                                  padding=True)

            # Get prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=1)
                confidence_score = probabilities[0][1].item()  # Assuming 1 is true class

            return confidence_score
        except Exception as e:
            print(f"DistilBERT Error: {str(e)}")
            return 0.5  # Return neutral score on error

class GeminiService:
    def __init__(self):
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        genai.configure(api_key=api_key)
        self.model = genai.get_model('gemini-1.0-pro')

    def analyze_claim(self, title: str, news_articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        try:
            # Format the context from news articles
            context = "\n\n".join([
                f"Source {i+1}:\nTitle: {article.get('title', '')}\nURL: {article.get('link', '')}\n"
                for i, article in enumerate(news_articles)
            ])

            # Create the prompt
            prompt = f"""Analyze this claim: "{title}"

Context from news sources:
{context}

Provide a detailed fact-check analysis in JSON format with the following fields:
- veracity (boolean): true if claim is verified, false if misleading
- confidence_score (float between 0-1): how confident the analysis is
- explanation (string, min 250 words): detailed analysis referencing sources
- category (string): type of claim
- key_findings (list): main points from analysis
- impact_level (string): one of ["VERIFIED", "MISLEADING", "PARTIAL"]
- sources (list): URLs of relevant sources

Requirements:
1. Reference each source by name/number in explanation
2. Connect evidence across sources
3. Include specific quotes/data points
4. Explain reasoning for confidence score
5. Minimum 250-word explanation
6. Clear true/false determination
7. List key findings with citations"""

            # Get response from Gemini
            response = self.model.generate_content(prompt)
            
            if not response or not response.text:
                raise ValueError("Empty response from Gemini API")

            # Parse the response
            try:
                result = json.loads(response.text)
            except json.JSONDecodeError:
                # If JSON parsing fails, try to extract JSON from the text
                text = response.text
                start = text.find('{')
                end = text.rfind('}') + 1
                if start != -1 and end != 0:
                    result = json.loads(text[start:end])
                else:
                    raise ValueError("Could not parse JSON from response")

            return result

        except Exception as e:
            print(f"Gemini Error: {str(e)}")
            return {
                "veracity": False,
                "confidence_score": 0.0,
                "explanation": f"Error processing request: {str(e)}",
                "category": "Error",
                "key_findings": [],
                "impact_level": "MISLEADING",
                "sources": []
            }

class CombinedAnalysisService:
    def __init__(self):
        self.distilbert_service = DistilBERTService()
        self.gemini_service = GeminiService()
        self.gemini_weight = 0.7  # Giving more weight to Gemini
        self.distilbert_weight = 0.3

    def analyze_claim(self, title: str, news_articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Get DistilBERT analysis
        distilbert_score = self.distilbert_service.analyze_claim(title)
        
        # Get Gemini analysis
        gemini_result = self.gemini_service.analyze_claim(title, news_articles)
        gemini_score = gemini_result["confidence_score"]

        # Calculate weighted average confidence score
        combined_score = (gemini_score * self.gemini_weight + 
                        distilbert_score * self.distilbert_weight)

        # Update the Gemini result with combined score
        gemini_result["confidence_score"] = combined_score
        gemini_result["model_scores"] = {
            "distilbert_score": distilbert_score,
            "gemini_score": gemini_score,
            "combined_score": combined_score
        }

        return gemini_result 