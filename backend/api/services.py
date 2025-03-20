import requests
from bs4 import BeautifulSoup
import json
from datetime import datetime
import os
from django.conf import settings
from typing import List, Dict, Any, Tuple
import time
import google.generativeai as genai
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW
import torch
import numpy as np
from django.utils import timezone
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import logging


class BraveNewsService:

    def __init__(self):
        self.api_key = os.getenv('BRAVE_API_KEY')
        if not self.api_key:
            raise ValueError(
                "BRAVE_API_KEY is not set in environment variables")

    def get_news_articles(self,
                          query: str,
                          limit: int = 5) -> List[Dict[str, str]]:
        headers = {
            'Accept': 'application/json',
            'Accept-Encoding': 'gzip',
            'X-Subscription-Token': self.api_key
        }

        params = {'q': query}

        try:
            # Add delay to prevent rate limiting
            time.sleep(1)

            response = requests.get(
                'https://api.search.brave.com/res/v1/web/search',
                headers=headers,
                params=params,
                timeout=10)
            response.raise_for_status()

            response_data = response.json()

            brave_news = []
            if 'web' in response_data and 'results' in response_data['web']:
                results = response_data['web']['results']

                for result in results[:limit]:
                    try:
                        article = {
                            'title':
                            result.get('title', ''),
                            'link':
                            result.get('url', ''),
                            'snippet':
                            result.get('description', ''),
                            'source':
                            result.get('site_name', 'Unknown Source'),
                            'time_published':
                            result.get('published_date', 'Unknown Date'),
                            'retrieved_at':
                            datetime.now().isoformat()
                        }
                        brave_news.append(article)
                    except Exception as e:
                        print(f"Error parsing individual result: {e}")
                        continue
            # print("BRAVE NEWS: ",brave_news)
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
        try:
            self.tokenizer = DistilBertTokenizer.from_pretrained(
                'distilbert-base-uncased'
            )
            self.model = DistilBertForSequenceClassification.from_pretrained(
                'distilbert-base-uncased',
                num_labels=2  # Binary classification (true/false)
            )
            self.model.eval()  # Set to evaluation mode
        except Exception as e:
            print(f"Error initializing DistilBERT model: {str(e)}")
            raise ValueError(f"Failed to initialize DistilBERT model: {str(e)}")

    def analyze_claim(self, title: str) -> float:
        """
        Analyze the claim using DistilBERT model
        Returns confidence score between 0 and 1
        """
        try:
            # Tokenize the input
            inputs = self.tokenizer(
                title,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )

            # Get prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=1)
                confidence_score = probabilities[0][1].item()  # Assuming 1 is true class
            # print(confidence_score)
            return confidence_score
        except Exception as e:
            print(f"DistilBERT Error: {str(e)}")
            return 0.5  # Return neutral score on error


class GeminiService:

    def __init__(self):
        self.api_key = os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        self.api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent?key={self.api_key}"

    def clean_json_text(self, text: str) -> str:
        """Clean the text to make it JSON-compatible"""
        # Remove backticks and other special characters
        text = text.replace('`', '')
        # Remove any control characters
        text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\r\t')
        # Ensure proper JSON formatting
        text = text.replace('\n', ' ').replace('\r', '')
        # Remove any extra whitespace
        text = ' '.join(text.split())
        return text

    def extract_json_from_text(self, text: str) -> str:
        """Extract JSON object from text"""
        # Find the first '{' and last '}'
        start = text.find('{')
        end = text.rfind('}') + 1
        if start == -1 or end == 0:
            raise ValueError("No JSON object found in text")
        return text[start:end]

    def analyze_claim(self, title: str, news_articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        try:
            # Format the context from news articles
            context = "\n\n".join([
                f"Source {i+1}:\nTitle: {article.get('title', '')}\nDescription: {article.get('snippet', '')}\nURL: {article.get('link', '')}\n"
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

            # Prepare the request payload
            payload = {
                "contents": [{
                    "parts": [{"text": prompt}]
                }],
                "generationConfig": {
                    "temperature": 0.7,
                    "topK": 40,
                    "topP": 0.95,
                    "maxOutputTokens": 2048,
                }
            }

            # Make the API request
            headers = {
                'Content-Type': 'application/json'
            }

            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            response_data = response.json()

            # Extract the generated text from the response
            if 'candidates' in response_data and response_data['candidates']:
                generated_text = response_data['candidates'][0]['content']['parts'][0]['text']
            else:
                raise ValueError("No response content found in Gemini API response")

            # Clean and parse the response
            try:
                # First attempt: try to parse the cleaned text directly
                cleaned_text = self.clean_json_text(generated_text)
                result = json.loads(cleaned_text)
            except json.JSONDecodeError:
                try:
                    # Second attempt: try to extract and parse JSON object
                    json_text = self.extract_json_from_text(cleaned_text)
                    result = json.loads(json_text)
                except json.JSONDecodeError as e:
                    print(f"JSON Parsing Error: {str(e)}")
                    print(f"Cleaned Text: {cleaned_text}")
                    raise ValueError("Could not parse JSON from response")

            # Validate required fields
            required_fields = ['veracity', 'confidence_score', 'explanation', 'category', 
                             'key_findings', 'impact_level', 'sources']
            missing_fields = [field for field in required_fields if field not in result]
            if missing_fields:
                raise ValueError(f"Missing required fields in response: {missing_fields}")

            # Ensure confidence score is between 0 and 1
            result['confidence_score'] = max(0.0, min(1.0, float(result['confidence_score'])))

            return result

        except requests.exceptions.RequestException as e:
            print(f"API Request Error: {str(e)}")
            return {
                "veracity": False,
                "confidence_score": 0.0,
                "explanation": f"API request failed: {str(e)}",
                "category": "Error",
                "key_findings": [],
                "impact_level": "MISLEADING",
                "sources": []
            }
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
        self.gemini_service = GeminiService()
        self.distilbert_service = DistilBERTService()
        self.brave_service = BraveNewsService()

    def analyze_claim(self, article_title: str, news_articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze a claim using multiple services and combine the results
        """
        try:
            # Get news articles from Brave Search
            brave_news = self.brave_service.get_news_articles(article_title)
            
            # Get Gemini analysis
            gemini_result = self.gemini_service.analyze_claim(article_title, brave_news)
            
            # Get DistilBERT analysis
            distilbert_score = self.distilbert_service.analyze_claim(article_title)
            
            # Combine results with weighted scores
            combined_score = ((gemini_result['confidence_score'] * 0.7) + (distilbert_score * 0.3))
            
            # Process sources from both Gemini and Brave
            gemini_sources = gemini_result.get('sources', [])
            brave_sources = [article.get('link', '') for article in brave_news if article.get('link')]
            
            # Combine and deduplicate sources
            all_sources = list(set(gemini_sources + brave_sources))
            
            # Create result dictionary
            result = {
                'veracity': gemini_result['veracity'],
                'confidence_score': combined_score,
                'explanation': gemini_result['explanation'],
                'category': gemini_result['category'],
                'key_findings': gemini_result['key_findings'],
                'impact_level': gemini_result['impact_level'],
                'sources': all_sources,  # Use combined and deduplicated sources
                'created_at': timezone.now(),
                'updated_at': timezone.now()
            }
            
            return result

        except Exception as e:
            print(f"Combined Analysis Error: {str(e)}")
            return {
                'veracity': 'UNKNOWN',
                'confidence_score': 0.5,
                'explanation': f"Error during analysis: {str(e)}",
                'category': 'ERROR',
                'key_findings': [],
                'impact_level': 'PARTIAL',
                'sources': [],
                'created_at': timezone.now(),
                'updated_at': timezone.now()
            }


class NewsDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer: DistilBertTokenizer, max_length: int = 512):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length, return_tensors='pt')
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class ModelTrainingService:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
        self.model.to(self.device)
        self.logger = logging.getLogger(__name__)

    def prepare_training_data(self, true_news: List[Dict[str, Any]], false_news: List[Dict[str, Any]]) -> Tuple[List[str], List[int]]:
        """Prepare training data from MongoDB collections"""
        texts = []
        labels = []
        
        # Add true news
        for news in true_news:
            texts.append(news['article_title'])
            labels.append(1)  # 1 for true
            
        # Add false news
        for news in false_news:
            texts.append(news['article_title'])
            labels.append(0)  # 0 for false
            
        return texts, labels

    def train_model(self, texts: List[str], labels: List[int], epochs: int = 3, batch_size: int = 8) -> bool:
        """Train the model on the provided data"""
        try:
            # Split data into train and validation sets
            train_texts, val_texts, train_labels, val_labels = train_test_split(
                texts, labels, test_size=0.2, random_state=42
            )

            # Create datasets
            train_dataset = NewsDataset(train_texts, train_labels, self.tokenizer)
            val_dataset = NewsDataset(val_texts, val_labels, self.tokenizer)

            # Create dataloaders
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)

            # Setup optimizer
            optimizer = AdamW(self.model.parameters(), lr=2e-5)

            # Training loop
            self.model.train()
            for epoch in range(epochs):
                total_loss = 0
                for batch in train_loader:
                    optimizer.zero_grad()
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)

                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )

                    loss = outputs.loss
                    total_loss += loss.item()

                    loss.backward()
                    optimizer.step()

                avg_loss = total_loss / len(train_loader)
                self.logger.info(f"Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}")

            # Save the model
            model_path = os.path.join(settings.BASE_DIR, 'models', 'distilbert_finetuned')
            os.makedirs(model_path, exist_ok=True)
            self.model.save_pretrained(model_path)
            self.tokenizer.save_pretrained(model_path)
            
            self.logger.info(f"Model saved successfully to {model_path}")
            return True

        except Exception as e:
            self.logger.error(f"Error during model training: {str(e)}")
            return False

    def retrain_model(self) -> Dict[str, Any]:
        """Retrain the model using data from MongoDB"""
        try:
            from .models import TrueNews, FalseNews
            
            # Get data from MongoDB
            true_news = list(TrueNews.objects.values('article_title'))
            false_news = list(FalseNews.objects.values('article_title'))
            
            if not true_news or not false_news:
                return {
                    'success': False,
                    'message': 'Insufficient data for training',
                    'true_news_count': len(true_news),
                    'false_news_count': len(false_news)
                }

            # Prepare training data
            texts, labels = self.prepare_training_data(true_news, false_news)
            
            # Train the model
            success = self.train_model(texts, labels)
            
            if success:
                # Clear the training data from MongoDB
                TrueNews.objects.all().delete()
                FalseNews.objects.all().delete()
                
                return {
                    'success': True,
                    'message': 'Model retrained successfully and training data cleared',
                    'true_news_count': len(true_news),
                    'false_news_count': len(false_news)
                }
            else:
                return {
                    'success': False,
                    'message': 'Model training failed',
                    'true_news_count': len(true_news),
                    'false_news_count': len(false_news)
                }

        except Exception as e:
            self.logger.error(f"Error in retrain_model: {str(e)}")
            return {
                'success': False,
                'message': f'Error during retraining: {str(e)}',
                'true_news_count': 0,
                'false_news_count': 0
            }
