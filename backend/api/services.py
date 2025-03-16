import requests
from bs4 import BeautifulSoup
import json
from datetime import datetime
import os
from django.conf import settings
from typing import List, Dict, Any
import time

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

class GeminiService:
    def __init__(self):
        self.api_key = settings.GEMINI_API_KEY
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is not set")
        
        # Update to use gemini-2.0-flash model
        self.api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

    def analyze_claim(self, article_title: str, news_articles: List[Dict[str, str]]) -> Dict[str, Any]:
        # Prepare the context from news articles with numbered sources
        context = "\n\n".join([
            f"Source {i+1}:\nTitle: {article['title']}\nSource: {article['source']}\nSummary: {article['snippet']}\nURL: {article['link']}\n"
            for i, article in enumerate(news_articles) if article['title'] != 'Error fetching news'
        ])

        # Updated prompt with more emphasis on detailed explanation
        prompt = f"""
        Analyze this claim: "{article_title}"
        
        Based on these news sources:
        {context}
        
        Provide a comprehensive fact-check analysis in the following JSON format:
        {{
            "veracity": false,  # boolean: true if claim is true, false if misleading/false
            "confidence_score": 0.85,  # float between 0 and 1
            "explanation": "Provide an extremely detailed explanation that thoroughly analyzes the claim using information from ALL five sources. For each source, explain how it supports or refutes the claim. Connect the evidence from different sources to form a comprehensive analysis. Include specific details mentioned in the sources and explain their relevance to the claim's veracity. The explanation should be at least 250 words long and reference each source by name",
            "category": "Politics/Technology/Health/etc",
            "key_findings": [
                "Detailed key point 1 with source reference",
                "Detailed key point 2 with source reference",
                "Detailed key point 3 with source reference",
                "Detailed key point 4 with source reference",
                "Detailed key point 5 with source reference"
            ],
            "impact_level": "VERIFIED/MISLEADING/PARTIAL",
            "sources": ["source1_url", "source2_url", "source3_url", "source4_url", "source5_url"]
        }}
        
        Requirements for the analysis:
        1. The explanation must be at least 250 words long
        2. Each source must be referenced by name in the explanation
        3. Key findings must cite specific sources
        4. Explain how each source supports or contradicts the claim
        5. Connect evidence across sources to form a comprehensive analysis
        6. Include specific quotes or data points from the sources
        7. Explain the reasoning behind the confidence score
        
        Only respond with the JSON object, no other text.
        """

        try:
            # Prepare the request payload according to Gemini API specs
            payload = {
                "contents": [{
                    "parts": [{"text": prompt}]
                }]
            }

            # Make the API request with the key as a query parameter
            url = f"{self.api_url}?key={self.api_key}"
            headers = {
                "Content-Type": "application/json"
            }

            # Print request details for debugging
            print(f"Making request to Gemini API: {url}")
            print(f"Request payload: {json.dumps(payload, indent=2)}")

            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=30
            )
            
            # Print response for debugging
            print(f"Gemini API Response Status: {response.status_code}")
            print(f"Gemini API Response: {response.text}")
            
            response.raise_for_status()
            
            # Parse the response according to Gemini API format
            response_data = response.json()
            
            # Handle the response format for gemini-2.0-flash
            if 'candidates' in response_data and len(response_data['candidates']) > 0:
                response_text = response_data['candidates'][0]['content']['parts'][0]['text']
            else:
                print(f"Unexpected response format: {response_data}")
                raise ValueError("No valid response from Gemini API")
            
            # Clean up the response text to ensure valid JSON
            response_text = response_text.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            
            # Parse the response and ensure it's valid JSON
            result = json.loads(response_text)
            
            # Ensure all required fields are present
            required_fields = ['veracity', 'confidence_score', 'explanation', 
                             'category', 'key_findings', 'impact_level', 'sources']
            
            for field in required_fields:
                if field not in result:
                    raise KeyError(f"Missing required field: {field}")
            
            # Ensure confidence_score is between 0 and 1
            result['confidence_score'] = max(0.0, min(1.0, float(result['confidence_score'])))
            
            # Ensure we have the source URLs from the news articles
            if 'sources' not in result or not result['sources']:
                result['sources'] = [article['link'] for article in news_articles if article['link']][:5]
            
            return result
            
        except requests.exceptions.RequestException as e:
            print(f"Request error in Gemini analysis: {e}")
            return {
                'veracity': False,
                'confidence_score': 0.5,
                'explanation': f"Error making request to Gemini API: {str(e)}",
                'category': 'Error',
                'key_findings': ['API request failed'],
                'impact_level': 'PARTIAL',
                'sources': [article['link'] for article in news_articles if article['link']][:5]
            }
        except json.JSONDecodeError as e:
            print(f"JSON parsing error in Gemini analysis: {e}")
            return {
                'veracity': False,
                'confidence_score': 0.5,
                'explanation': f"Error parsing Gemini API response: {str(e)}",
                'category': 'Error',
                'key_findings': ['Invalid response format'],
                'impact_level': 'PARTIAL',
                'sources': [article['link'] for article in news_articles if article['link']][:5]
            }
        except Exception as e:
            print(f"Error in Gemini analysis: {e}")
            return {
                'veracity': False,
                'confidence_score': 0.5,
                'explanation': f"Error analyzing claim: {str(e)}",
                'category': 'Unknown',
                'key_findings': ['Error in analysis'],
                'impact_level': 'PARTIAL',
                'sources': [article['link'] for article in news_articles if article['link']][:5]
            }

    def _determine_impact_level(self, content: str) -> str:
        """Helper method to determine impact level based on content analysis"""
        # Add your impact level determination logic here
        return 'MEDIUM'  # Default return 