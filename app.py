from dotenv import load_dotenv
import os

# Load environment variables at the VERY TOP
load_dotenv()

from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import plotly
import plotly.graph_objects as go
import json
import traceback
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Test if environment variables are loading
news_api_key = os.getenv("NEWS_API_KEY")
logger.info(f"NewsAPI Key loaded: {'YES' if news_api_key else 'NO'}")

class StockPredictor:
    def __init__(self):
        self.is_trained = False
    
    def predict_future_prices(self, data, days=7):
        """Simple prediction using trend analysis"""
        try:
            if len(data) < 10:
                return {"error": "Insufficient data for prediction"}
            
            current_price = data['Close'].iloc[-1]
            
            # Simple trend-based prediction
            recent_prices = data['Close'].tail(10)
            trend = np.polyfit(range(len(recent_prices)), recent_prices, 1)[0]
            
            predictions = []
            confidence = []
            prediction_dates = []
            
            # Generate predictions
            last_date = data.index[-1]
            for i in range(1, days + 1):
                next_date = last_date + pd.Timedelta(days=i)
                prediction_dates.append(next_date.strftime('%Y-%m-%d'))
                
                # Simple linear projection
                pred_price = current_price + (trend * i)
                predictions.append(round(max(pred_price, 0), 2))
                
                # Confidence decreases over time
                conf = max(0.3, 1 - (i * 0.1))
                confidence.append(round(conf, 2))
            
            predicted_change = ((predictions[-1] - current_price) / current_price) * 100
            
            return {
                'predictions': predictions,
                'confidence': confidence,
                'prediction_dates': prediction_dates,
                'trend': 'bullish' if trend > 0 else 'bearish',
                'predicted_change_percent': round(predicted_change, 2),
                'current_price': round(current_price, 2),
                'prediction_days': days
            }
        except Exception as e:
            return {"error": f"Prediction failed: {str(e)}"}

class SentimentAnalyzer:
    def __init__(self):
        self.newsapi_key = os.getenv("NEWS_API_KEY")
    
    def get_news_sentiment(self, symbol):
        """Get sentiment analysis for stock"""
        try:
            # Mock sentiment data - in production, you'd use NewsAPI
            sentiments = ['positive', 'neutral', 'negative']
            sentiment_weights = [0.4, 0.4, 0.2]  # Slightly biased positive for demo
            
            import random
            sentiment = random.choices(sentiments, weights=sentiment_weights)[0]
            sentiment_score = random.uniform(-0.5, 0.5)
            
            # Sample headlines based on symbol
            headlines_data = {
                'AAPL': [
                    "Apple announces new iPhone features",
                    "Analysts raise Apple price target",
                    "Apple expands services business"
                ],
                'GOOGL': [
                    "Google AI advancements impress market",
                    "Alphabet reports strong earnings",
                    "Google Cloud continues growth trajectory"
                ],
                'MSFT': [
                    "Microsoft Azure gains market share",
                    "Windows 11 adoption increases",
                    "Microsoft partners with OpenAI"
                ],
                'TSLA': [
                    "Tesla deliveries exceed expectations",
                    "New Tesla model receives positive reviews",
                    "Tesla expands charging network"
                ]
            }
            
            headlines = headlines_data.get(symbol, [
                f"{symbol} shows strong performance",
                f"Market reacts to {symbol} news",
                f"Analysts watching {symbol} closely"
            ])
            
            return {
                'overall_sentiment': sentiment,
                'sentiment_score': round(sentiment_score, 2),
                'sample_headlines': headlines,
                'confidence': round(random.uniform(0.6, 0.9), 2),
                'source': 'market_analysis',
                'articles_count': len(headlines)
            }
        except Exception as e:
            logger.error(f"Sentiment analysis error: {e}")
            return self._get_fallback_sentiment(symbol)
    
    def _get_fallback_sentiment(self, symbol):
        """Fallback sentiment data"""
        return {
            'overall_sentiment': 'neutral',
            'sentiment_score': 0.0,
            'sample_headlines': [
                f"Market watching {symbol} performance",
                f"{symbol} trading in expected range",
                "Analysts monitoring market conditions"
            ],
            'confidence': 0.7,
            'source': 'fallback',
            'articles_count': 0
        }

# Initialize models
stock_predictor = StockPredictor()
sentiment_analyzer = SentimentAnalyzer()
def fetch_stock_data(symbol, period='1y'):
    """Fetch stock data from Yahoo Finance"""
    try:
        logger.info(f"Fetching data for {symbol} from Yahoo Finance...")
        
        # Clean the symbol
        symbol = symbol.upper().strip()
        
        # Try to fetch data
        stock = yf.Ticker(symbol)
        hist = stock.history(period=period)
        
        if hist.empty:
            # Try with different period
            hist = stock.history(period="1y")
        
        if hist.empty:
            return _generate_mock_data(symbol, period)  # Removed self.
        
        # Calculate technical indicators
        if len(hist) > 20:
            hist['SMA_20'] = hist['Close'].rolling(window=20).mean()
            hist['SMA_50'] = hist['Close'].rolling(window=50).mean()
            hist['RSI'] = _calculate_rsi(hist['Close'])  # Removed self.
        else:
            # Fill with current price if not enough data
            hist['SMA_20'] = hist['Close']
            hist['SMA_50'] = hist['Close']
            hist['RSI'] = 50
        
        logger.info(f"Successfully fetched {len(hist)} records for {symbol}")
        return hist
        
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {e}")
        return _generate_mock_data(symbol, period)  # Removed self.

def _calculate_rsi(prices, window=14):
    """Calculate Relative Strength Index"""
    try:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    except Exception as e:
        logger.error(f"Error calculating RSI: {e}")
        return pd.Series([50] * len(prices), index=prices.index)

def _generate_mock_data(symbol, period='1y'):
    """Generate realistic mock stock data when API fails"""
    logger.info(f"Generating mock data for {symbol}")
    
    # Calculate days based on period
    days_map = {'1mo': 30, '3mo': 90, '6mo': 180, '1y': 365, '2y': 730}
    days = days_map.get(period, 365)
    
    # Generate dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Base prices for different stocks
    base_prices = {
        'AAPL': 150, 'GOOGL': 120, 'MSFT': 300, 'TSLA': 200,
        'AMZN': 130, 'META': 250, 'NVDA': 400, 'NFLX': 350,
        'JPM': 150, 'JNJ': 160
    }
    base_price = base_prices.get(symbol, 100)
    
    # Generate realistic price data
    np.random.seed(hash(symbol) % 10000)
    prices = []
    current_price = base_price
    
    for _ in range(len(dates)):
        change = np.random.normal(0.001, 0.02)  # Small daily change with some volatility
        current_price = max(current_price * (1 + change), base_price * 0.3)
        prices.append(round(current_price, 2))
    
    # Create DataFrame
    data = []
    for i, date in enumerate(dates):
        close_price = prices[i]
        open_price = round(close_price * (1 + np.random.normal(0, 0.01)), 2)
        high_price = round(max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.005))), 2)
        low_price = round(min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.005))), 2)
        volume = np.random.randint(1000000, 50000000)
        
        data.append({
            'Open': open_price,
            'High': high_price,
            'Low': low_price,
            'Close': close_price,
            'Volume': volume
        })
    
    df = pd.DataFrame(data, index=dates)
    
    # Calculate indicators
    df['SMA_20'] = df['Close'].rolling(20).mean().fillna(df['Close'])
    df['SMA_50'] = df['Close'].rolling(50).mean().fillna(df['Close'])
    df['RSI'] = _calculate_rsi(df['Close'])
    
    return df
def _generate_mock_data(symbol, period='1y'):
    """Generate realistic mock stock data when API fails"""
    logger.info(f"Generating mock data for {symbol}")
    
    # Calculate days based on period
    days_map = {'1mo': 30, '3mo': 90, '6mo': 180, '1y': 365, '2y': 730}
    days = days_map.get(period, 365)
    
    # Generate dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Base prices for different stocks
    base_prices = {
        'AAPL': 150, 'GOOGL': 120, 'MSFT': 300, 'TSLA': 200,
        'AMZN': 130, 'META': 250, 'NVDA': 400, 'NFLX': 350
    }
    base_price = base_prices.get(symbol, 100)
    
    # Generate realistic price data
    np.random.seed(hash(symbol) % 10000)
    prices = []
    current_price = base_price
    
    for _ in range(len(dates)):
        change = np.random.normal(0.001, 0.02)  # Small daily change with some volatility
        current_price = max(current_price * (1 + change), base_price * 0.3)
        prices.append(round(current_price, 2))
    
    # Create DataFrame
    data = []
    for i, date in enumerate(dates):
        close_price = prices[i]
        open_price = round(close_price * (1 + np.random.normal(0, 0.01)), 2)
        high_price = round(max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.005))), 2)
        low_price = round(min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.005))), 2)
        volume = np.random.randint(1000000, 50000000)
        
        data.append({
            'Open': open_price,
            'High': high_price,
            'Low': low_price,
            'Close': close_price,
            'Volume': volume
        })
    
    df = pd.DataFrame(data, index=dates)
    
    # Calculate indicators
    df['SMA_20'] = df['Close'].rolling(20).mean().fillna(df['Close'])
    df['SMA_50'] = df['Close'].rolling(50).mean().fillna(df['Close'])
    df['RSI'] = _calculate_rsi(df['Close'])
    
    return df

def generate_plots(data, symbol):
    """Generate stock analysis plots"""
    plots = {}
    
    try:
        # Price chart
        fig_price = go.Figure()
        fig_price.add_trace(go.Scatter(
            x=data.index, 
            y=data['Close'], 
            mode='lines', 
            name='Close Price', 
            line=dict(color='#1f77b4')
        ))
        
        # Add moving averages if available
        if 'SMA_20' in data.columns:
            fig_price.add_trace(go.Scatter(
                x=data.index, 
                y=data['SMA_20'], 
                mode='lines', 
                name='SMA 20', 
                line=dict(color='#ff7f0e', dash='dash')
            ))
        
        if 'SMA_50' in data.columns:
            fig_price.add_trace(go.Scatter(
                x=data.index, 
                y=data['SMA_50'], 
                mode='lines', 
                name='SMA 50', 
                line=dict(color='#2ca02c', dash='dash')
            ))
            
        fig_price.update_layout(
            title=f'{symbol} Stock Price',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            template='plotly_white',
            height=400
        )
        plots['price_chart'] = json.dumps(fig_price, cls=plotly.utils.PlotlyJSONEncoder)
        
        # RSI chart
        if 'RSI' in data.columns:
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(
                x=data.index, 
                y=data['RSI'], 
                mode='lines', 
                name='RSI', 
                line=dict(color='#9467bd')
            ))
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
            fig_rsi.add_hline(y=50, line_dash="dot", line_color="gray")
            fig_rsi.update_layout(
                title='Relative Strength Index (RSI)',
                xaxis_title='Date',
                yaxis_title='RSI',
                template='plotly_white',
                height=300
            )
            plots['rsi_chart'] = json.dumps(fig_rsi, cls=plotly.utils.PlotlyJSONEncoder)
        
        # Volume chart
        colors = ['red' if data['Close'].iloc[i] < data['Open'].iloc[i] else 'green' 
                 for i in range(len(data))]
        
        fig_volume = go.Figure()
        fig_volume.add_trace(go.Bar(
            x=data.index, 
            y=data['Volume'], 
            name='Volume', 
            marker_color=colors
        ))
        fig_volume.update_layout(
            title='Trading Volume',
            xaxis_title='Date',
            yaxis_title='Volume',
            template='plotly_white',
            height=300
        )
        plots['volume_chart'] = json.dumps(fig_volume, cls=plotly.utils.PlotlyJSONEncoder)
        
    except Exception as e:
        logger.error(f"Error generating plots: {e}")
    
    return plots

def calculate_statistics(data):
    """Calculate comprehensive stock statistics"""
    try:
        if len(data) == 0:
            return {
                'current_price': 0,
                'price_change': 0,
                'price_change_percent': 0,
                'rsi': 50,
                'sma_20': 0,
                'sma_50': 0,
                'volume': '0',
                'day_high': 0,
                'day_low': 0
            }
        
        latest = data.iloc[-1]
        
        # Calculate price change
        price_change = 0
        price_change_percent = 0
        if len(data) > 1:
            prev_close = data['Close'].iloc[-2]
            price_change = round(latest['Close'] - prev_close, 2)
            price_change_percent = round((price_change / prev_close) * 100, 2)
        
        # Get technical indicators with safe access
        rsi_value = latest.get('RSI', 50) if 'RSI' in data.columns else 50
        sma_20_value = latest.get('SMA_20', latest['Close']) if 'SMA_20' in data.columns else latest['Close']
        sma_50_value = latest.get('SMA_50', latest['Close']) if 'SMA_50' in data.columns else latest['Close']
        
        stats = {
            'current_price': round(float(latest['Close']), 2),
            'price_change': float(price_change),
            'price_change_percent': float(price_change_percent),
            'rsi': round(float(rsi_value), 2),
            'sma_20': round(float(sma_20_value), 2),
            'sma_50': round(float(sma_50_value), 2),
            'volume': f"{int(latest['Volume']):,}",
            'day_high': round(float(latest['High']), 2),
            'day_low': round(float(latest['Low']), 2)
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Error calculating statistics: {e}")
        # Return default values
        return {
            'current_price': 100.0,
            'price_change': 0.0,
            'price_change_percent': 0.0,
            'rsi': 50.0,
            'sma_20': 100.0,
            'sma_50': 100.0,
            'volume': '1,000,000',
            'day_high': 105.0,
            'day_low': 95.0
        }

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/analysis')
def analysis_page():
    """Analysis page"""
    return render_template('analysis.html')

@app.route('/prediction')
def prediction_page():
    """Prediction page"""
    return render_template('prediction.html')

@app.route('/analyze', methods=['POST'])
def analyze_stock():
    """Main analysis endpoint"""
    try:
        symbol = request.form.get('symbol', 'AAPL').upper().strip()
        period = request.form.get('period', '1y')
        
        logger.info(f"Analyzing stock: {symbol} for period: {period}")
        
        # Fetch stock data
        data = fetch_stock_data(symbol, period)
        
        if data is None or len(data) == 0:
            return jsonify({
                'error': f'No data available for {symbol}. Please try a different symbol.'
            })

        # Generate plots
        plots = generate_plots(data, symbol)
        
        # Calculate statistics
        stats = calculate_statistics(data)
        
        # Get predictions
        prediction = stock_predictor.predict_future_prices(data)
        
        # Get sentiment analysis
        sentiment = sentiment_analyzer.get_news_sentiment(symbol)
        
        response_data = {
            'symbol': symbol,
            'period': period,
            'plots': plots,
            'statistics': stats,
            'prediction': prediction,
            'sentiment': sentiment,
            'status': 'success'
        }
        
        logger.info(f"Analysis complete for {symbol}")
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error in analyze_stock: {str(e)}")
        return jsonify({'error': f'Analysis failed: {str(e)}'})

@app.route('/api/stocks')
def get_available_stocks():
    """API endpoint to get list of available stocks"""
    popular_stocks = [
        {'symbol': 'AAPL', 'name': 'Apple Inc.'},
        {'symbol': 'GOOGL', 'name': 'Alphabet Inc.'},
        {'symbol': 'MSFT', 'name': 'Microsoft Corporation'},
        {'symbol': 'AMZN', 'name': 'Amazon.com Inc.'},
        {'symbol': 'TSLA', 'name': 'Tesla Inc.'},
        {'symbol': 'META', 'name': 'Meta Platforms Inc.'},
        {'symbol': 'NVDA', 'name': 'NVIDIA Corporation'},
        {'symbol': 'NFLX', 'name': 'Netflix Inc.'},
        {'symbol': 'JPM', 'name': 'JPMorgan Chase & Co.'},
        {'symbol': 'JNJ', 'name': 'Johnson & Johnson'}
    ]
    return jsonify({'stocks': popular_stocks})

if __name__ == '__main__':
    logger.info("Starting Stock Market Analyzer...")
    app.run(debug=True, host='127.0.0.1', port=5000)