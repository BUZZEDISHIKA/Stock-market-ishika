import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class StockPredictor:
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=100, 
            random_state=42,
            max_depth=10,
            min_samples_split=5
        )
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def create_features(self, data):
        """Create comprehensive technical features"""
        df = data.copy()
        
        # Price-based features
        for lag in [1, 2, 3, 5, 10]:
            df[f'Price_Lag_{lag}'] = df['Close'].shift(lag)
        
        # Moving averages
        for window in [5, 10, 20, 50]:
            df[f'SMA_{window}'] = df['Close'].rolling(window=window).mean()
            df[f'EMA_{window}'] = df['Close'].ewm(span=window).mean()
        
        # Bollinger Bands - FIXED: Create separate columns
        bb_middle = df['Close'].rolling(20).mean()
        bb_std = df['Close'].rolling(20).std()
        df['BB_Upper'] = bb_middle + (bb_std * 2)  # Separate column
        df['BB_Lower'] = bb_middle - (bb_std * 2)  # Separate column  
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / bb_middle
        
        # Volatility features
        df['Volatility_5'] = df['Close'].rolling(5).std()
        df['Volatility_20'] = df['Close'].rolling(20).std()
        
        # Volume features
        df['Volume_SMA'] = df['Volume'].rolling(20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        # Price range features
        df['High_Low_Ratio'] = df['High'] / df['Low']
        df['Close_Open_Ratio'] = df['Close'] / df['Open']
        
        # RSI
        df['RSI'] = self.calculate_rsi(df['Close'])
        
        # MACD
        df['MACD'] = self.calculate_macd(df['Close'])
        
        # Price momentum
        df['Momentum_5'] = df['Close'] / df['Close'].shift(5) - 1
        df['Momentum_10'] = df['Close'] / df['Close'].shift(10) - 1
        
        # Target variable (next day's closing price)
        df['Target'] = df['Close'].shift(-1)
        
        return df.dropna()
    
    def calculate_rsi(self, prices, window=14):
        """Calculate RSI with error handling"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))
        except:
            return pd.Series([50] * len(prices), index=prices.index)
    
    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD"""
        try:
            exp1 = prices.ewm(span=fast, adjust=False).mean()
            exp2 = prices.ewm(span=slow, adjust=False).mean()
            macd = exp1 - exp2
            return macd
        except:
            return pd.Series([0] * len(prices), index=prices.index)
    
    def train_model(self, data):
        """Train the prediction model with feature selection"""
        df_with_features = self.create_features(data)
        
        if len(df_with_features) < 30:
            return False
        
        # Select important features
        feature_columns = [
            'Price_Lag_1', 'Price_Lag_2', 'Price_Lag_3',
            'SMA_5', 'SMA_20', 'EMA_5', 'EMA_20',
            'RSI', 'MACD', 'Volatility_20',
            'Volume_Ratio', 'Momentum_5', 'BB_Width'
        ]
        
        # Only use features that exist in the dataframe
        available_features = [f for f in feature_columns if f in df_with_features.columns]
        
        if len(available_features) < 5:
            return False
        
        X = df_with_features[available_features]
        y = df_with_features['Target']
        
        # Remove any infinite values
        X = X.replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(method='bfill')
        y = y.replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(method='bfill')
        
        if X.isnull().any().any() or y.isnull().any():
            return False
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        self.is_trained = True
        self.feature_names = available_features
        
        return True
    
    def predict_future_prices(self, data, days=7):
        """Predict future stock prices with confidence intervals"""
        try:
            if not self.train_model(data):
                return {"error": "Insufficient data for prediction"}
            
            df_with_features = self.create_features(data)
            if len(df_with_features) < 1:
                return {"error": "No features available for prediction"}
            
            current_data = df_with_features[self.feature_names].iloc[-1:].copy()
            
            predictions = []
            confidence = []
            prediction_dates = []
            
            # Generate dates for predictions
            last_date = data.index[-1]
            for i in range(1, days + 1):
                next_date = last_date + pd.Timedelta(days=i)
                prediction_dates.append(next_date.strftime('%Y-%m-%d'))
            
            current_prediction_data = current_data.copy()
            
            for i in range(days):
                # Scale current data
                current_scaled = self.scaler.transform(current_prediction_data)
                
                # Make prediction
                pred = self.model.predict(current_scaled)[0]
                
                # Calculate confidence based on recent volatility and prediction stability
                recent_volatility = data['Close'].tail(10).std() / data['Close'].tail(10).mean()
                base_confidence = max(0.3, 1 - recent_volatility * 10)
                
                # Adjust confidence based on prediction day (less confident for further days)
                day_adjustment = 1 - (i * 0.1)
                conf = max(0.2, base_confidence * day_adjustment)
                
                predictions.append(round(pred, 2))
                confidence.append(round(conf, 2))
                
                # Update features for next prediction (simplified)
                if i < days - 1:
                    new_row = current_prediction_data.iloc[-1].copy()
                    
                    # Update price lags
                    for lag in range(3, 1, -1):
                        new_row[f'Price_Lag_{lag}'] = new_row[f'Price_Lag_{lag-1}']
                    new_row['Price_Lag_1'] = pred
                    
                    current_prediction_data = pd.DataFrame([new_row])
            
            current_price = data['Close'].iloc[-1]
            predicted_change = ((predictions[-1] - current_price) / current_price) * 100
            
            return {
                'predictions': predictions,
                'confidence': confidence,
                'prediction_dates': prediction_dates,
                'trend': 'bullish' if predicted_change > 0 else 'bearish',
                'predicted_change_percent': round(predicted_change, 2),
                'current_price': round(current_price, 2),
                'prediction_days': days
            }
        except Exception as e:
            return {"error": f"Prediction failed: {str(e)}"}