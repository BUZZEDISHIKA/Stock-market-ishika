import requests
from textblob import TextBlob
import re
import os
import logging

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    def __init__(self):
        # Remove dotenv loading - it's already loaded in app.py
        self.newsapi_key = os.getenv("NEWS_API_KEY")
        
        if not self.newsapi_key:
            logger.warning("NEWS_API_KEY not found in environment variables")
        else:
            logger.info(f"NewsAPI key loaded successfully (length: {len(self.newsapi_key)})")
    
    # ... rest of your code remains the same ...
    def get_news_sentiment(self, symbol):
        """Get real news sentiment using only NewsAPI"""
        logger.info(f"Fetching news for {symbol} using NewsAPI")
        
        headlines = []
        sentiment_scores = []
        
        # Use only NewsAPI
        try:
            newsapi_articles = self._fetch_newsapi(symbol)
            if newsapi_articles:
                headlines.extend([article['title'] for article in newsapi_articles])
                sentiment_scores.extend([self._analyze_text_sentiment(article['title']) 
                                       for article in newsapi_articles])
                logger.info(f"Successfully fetched {len(newsapi_articles)} articles from NewsAPI")
                
        except Exception as e:
            logger.error(f"NewsAPI error: {e}")
        
        # If no real data, use mock data as fallback
        if not headlines:
            logger.info("Using mock data as fallback")
            return self._get_mock_sentiment(symbol)
        
        # Calculate overall sentiment
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
        
        if avg_sentiment > 0.1:
            overall_sentiment = 'positive'
        elif avg_sentiment < -0.1:
            overall_sentiment = 'negative'
        else:
            overall_sentiment = 'neutral'
        
        confidence = min(0.9, max(0.6, len(headlines) * 0.15))
        
        logger.info(f"Sentiment analysis complete: {overall_sentiment} (score: {avg_sentiment:.2f})")
        
        return {
            'overall_sentiment': overall_sentiment,
            'sentiment_score': round(avg_sentiment, 2),
            'sample_headlines': headlines[:3],  # Return top 3 headlines
            'confidence': round(confidence, 2),
            'source': 'real_news',
            'articles_count': len(headlines)
        }
    
    def _fetch_newsapi(self, symbol):
        """Fetch news from NewsAPI only"""
        if not self.newsapi_key:
            raise Exception("NewsAPI key not configured")
        
        # Map common symbols to company names for better news results
        symbol_to_query = {
            'AAPL': 'Apple OR AAPL',
            'GOOGL': 'Google OR Alphabet OR GOOGL',
            'MSFT': 'Microsoft OR MSFT',
            'AMZN': 'Amazon OR AMZN',
            'TSLA': 'Tesla OR TSLA',
            'META': 'Meta OR Facebook OR META',
            'NFLX': 'Netflix OR NFLX',
            'NVDA': 'NVIDIA OR NVDA',
            'JPM': 'JPMorgan OR JPM',
            'JNJ': 'Johnson & Johnson OR JNJ'
        }
        
        query = symbol_to_query.get(symbol, symbol)
        
        url = "https://newsapi.org/v2/everything"
        params = {
            'q': query,
            'language': 'en',
            'sortBy': 'publishedAt',
            'pageSize': 10,
            'apiKey': self.newsapi_key
        }
        
        logger.info(f"Making NewsAPI request for: {query}")
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code != 200:
            error_msg = response.json().get('message', 'Unknown error')
            logger.error(f"NewsAPI returned error: {error_msg}")
            raise Exception(f"NewsAPI error: {error_msg}")
        
        data = response.json()
        articles = data.get('articles', [])
        
        # Filter out articles with [Removed] title or no content
        filtered_articles = [
            article for article in articles 
            if article.get('title') not in ['[Removed]', None] 
            and article.get('description') not in ['[Removed]', None]
            and len(article.get('title', '')) > 10  # Minimum title length
        ]
        
        return filtered_articles
    
    def _analyze_text_sentiment(self, text):
        """Analyze sentiment of text using TextBlob"""
        try:
            # Clean the text
            clean_text = re.sub(r'http\S+', '', text)
            clean_text = re.sub(r'[^a-zA-Z\s]', '', clean_text)
            clean_text = clean_text.strip()
            
            if len(clean_text) < 10:  # Too short for meaningful analysis
                return 0
                
            analysis = TextBlob(clean_text)
            return analysis.sentiment.polarity
        except Exception as e:
            logger.error(f"Error analyzing text sentiment: {e}")
            return 0
    
    def _get_mock_sentiment(self, symbol):
        """Fallback to mock sentiment data when API fails"""
        import random
        
        # More realistic mock data based on symbol
        mock_data = {
            'AAPL': {
                'positive': [
                    "Apple reports record iPhone sales",
                    "New Apple product launch exceeds expectations",
                    "Analysts raise Apple price target"
                ],
                'negative': [
                    "Apple faces regulatory challenges in Europe",
                    "iPhone demand concerns weigh on Apple stock",
                    "Supply chain issues affect Apple production"
                ]
            },
            'TSLA': {
                'positive': [
                    "Tesla deliveries beat estimates",
                    "New Tesla model receives rave reviews", 
                    "Tesla expands supercharger network globally"
                ],
                'negative': [
                    "Tesla recalls vehicles over safety concerns",
                    "Competition heats up in electric vehicle market",
                    "Tesla faces production delays"
                ]
            },
            'GOOGL': {
                'positive': [
                    "Google AI advancements impress investors",
                    "Strong cloud growth boosts Alphabet earnings",
                    "New Google products gain market traction"
                ],
                'negative': [
                    "Google faces antitrust scrutiny",
                    "Digital ad market slowdown concerns",
                    "Regulatory challenges for Google services"
                ]
            }
        }
        
        # Get symbol-specific data or use general data
        symbol_data = mock_data.get(symbol, {
            'positive': [f"{symbol} shows strong quarterly results", f"Analysts upgrade {symbol} rating"],
            'negative': [f"{symbol} faces market headwinds", f"Competition pressures {symbol} margins"]
        })
        
        # Weight slightly towards positive for demo
        sentiments = ['positive', 'neutral', 'negative']
        weights = [0.5, 0.3, 0.2]
        sentiment = random.choices(sentiments, weights=weights)[0]
        
        if sentiment == 'positive':
            headlines = random.sample(symbol_data['positive'], 2) + [f"Market optimistic about {symbol} future"]
        elif sentiment == 'negative':
            headlines = random.sample(symbol_data['negative'], 2) + [f"Investors cautious on {symbol} outlook"]
        else:
            headlines = [f"{symbol} trades in narrow range", f"Mixed signals for {symbol} stock", f"Analysts divided on {symbol} prospects"]
        
        sentiment_scores = [self._analyze_text_sentiment(headline) for headline in headlines]
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
        
        return {
            'overall_sentiment': sentiment,
            'sentiment_score': round(avg_sentiment, 2),
            'sample_headlines': headlines,
            'confidence': round(random.uniform(0.6, 0.8), 2),
            'source': 'mock_data',
            'articles_count': 0
        }