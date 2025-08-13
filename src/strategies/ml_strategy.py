"""
Machine Learning-Based Trading Strategy

This module implements advanced machine learning strategies for HFT trading,
including signal generation, feature engineering, and model-based decision making.

Educational Notes:
- ML in HFT focuses on pattern recognition in market microstructure
- Features include order book imbalance, price momentum, volatility measures
- Models must be fast enough for real-time trading (microsecond latency)
- Overfitting is a major concern with financial time series data
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score
import joblib
from datetime import datetime, timedelta
import warnings

from src.strategies.base_strategy import BaseStrategy
from src.engine.order_types import Order, MarketDataPoint
from src.utils.constants import OrderSide, OrderType
from src.utils.logger import get_logger, log_performance
from src.performance.metrics import calculate_sharpe_ratio, calculate_max_drawdown


class MLFeatureEngineer:
    """
    Feature engineering for machine learning trading strategies
    
    This class creates features from market microstructure data that are
    predictive of short-term price movements.
    """
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.feature_names = []
        self.scalers = {}
        
    def create_features(self, market_data: pd.DataFrame, 
                       lookback_periods: List[int] = [5, 10, 20, 50]) -> pd.DataFrame:
        """
        Create comprehensive feature set from market data
        
        Args:
            market_data: DataFrame with OHLCV and order book data
            lookback_periods: Periods for rolling calculations
            
        Returns:
            DataFrame with engineered features
        """
        features = market_data.copy()
        
        # Price-based features
        features = self._add_price_features(features, lookback_periods)
        
        # Volume-based features
        features = self._add_volume_features(features, lookback_periods)
        
        # Order book features
        features = self._add_orderbook_features(features, lookback_periods)
        
        # Technical indicators
        features = self._add_technical_features(features, lookback_periods)
        
        # Microstructure features
        features = self._add_microstructure_features(features, lookback_periods)
        
        # Time-based features
        features = self._add_time_features(features)
        
        # Store feature names
        self.feature_names = [col for col in features.columns 
                             if col not in ['timestamp', 'price', 'volume']]
        
        return features
    
    def _add_price_features(self, df: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
        """Add price-based features"""
        for period in periods:
            # Returns
            df[f'return_{period}'] = df['price'].pct_change(period)
            
            # Log returns
            df[f'log_return_{period}'] = np.log(df['price'] / df['price'].shift(period))
            
            # Price momentum
            df[f'momentum_{period}'] = df['price'] / df['price'].rolling(period).mean() - 1
            
            # Price volatility
            df[f'volatility_{period}'] = df['price'].rolling(period).std()
            
            # Price acceleration
            df[f'acceleration_{period}'] = df[f'return_{period}'] - df[f'return_{period}'].shift(1)
        
        return df
    
    def _add_volume_features(self, df: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
        """Add volume-based features"""
        for period in periods:
            # Volume moving average
            df[f'volume_ma_{period}'] = df['volume'].rolling(period).mean()
            
            # Volume ratio
            df[f'volume_ratio_{period}'] = df['volume'] / df[f'volume_ma_{period}']
            
            # Volume-weighted average price
            df[f'vwap_{period}'] = (df['price'] * df['volume']).rolling(period).sum() / \
                                  df['volume'].rolling(period).sum()
            
            # Price vs VWAP
            df[f'price_vwap_ratio_{period}'] = df['price'] / df[f'vwap_{period}'] - 1
        
        return df
    
    def _add_orderbook_features(self, df: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
        """Add order book-based features"""
        # Assume we have bid/ask data
        if 'best_bid' in df.columns and 'best_ask' in df.columns:
            # Spread
            df['spread'] = df['best_ask'] - df['best_bid']
            df['spread_pct'] = df['spread'] / df['price']
            
            # Mid price
            df['mid_price'] = (df['best_bid'] + df['best_ask']) / 2
            df['price_vs_mid'] = df['price'] / df['mid_price'] - 1
            
            for period in periods:
                # Spread statistics
                df[f'spread_ma_{period}'] = df['spread'].rolling(period).mean()
                df[f'spread_std_{period}'] = df['spread'].rolling(period).std()
                
                # Order book imbalance (if we have volume data)
                if 'bid_volume' in df.columns and 'ask_volume' in df.columns:
                    df[f'imbalance_{period}'] = (
                        (df['bid_volume'] - df['ask_volume']) / 
                        (df['bid_volume'] + df['ask_volume'])
                    ).rolling(period).mean()
        
        return df
    
    def _add_technical_features(self, df: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
        """Add technical analysis features"""
        for period in periods:
            # Moving averages
            df[f'sma_{period}'] = df['price'].rolling(period).mean()
            df[f'ema_{period}'] = df['price'].ewm(span=period).mean()
            
            # Price vs moving averages
            df[f'price_sma_ratio_{period}'] = df['price'] / df[f'sma_{period}'] - 1
            df[f'price_ema_ratio_{period}'] = df['price'] / df[f'ema_{period}'] - 1
            
            # Bollinger Bands
            rolling_std = df['price'].rolling(period).std()
            df[f'bb_upper_{period}'] = df[f'sma_{period}'] + 2 * rolling_std
            df[f'bb_lower_{period}'] = df[f'sma_{period}'] - 2 * rolling_std
            df[f'bb_position_{period}'] = (df['price'] - df[f'bb_lower_{period}']) / \
                                         (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}'])
            
            # RSI
            delta = df['price'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rs = gain / loss
            df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        return df
    
    def _add_microstructure_features(self, df: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
        """Add market microstructure features"""
        # Trade direction (if available)
        if 'trade_direction' in df.columns:
            for period in periods:
                # Order flow imbalance
                df[f'order_flow_{period}'] = df['trade_direction'].rolling(period).sum()
                
                # Trade intensity
                df[f'trade_intensity_{period}'] = df['volume'].rolling(period).sum()
        
        # Price impact measures
        for period in [1, 2, 3, 5]:
            df[f'price_impact_{period}'] = df['price'].shift(-period) / df['price'] - 1
        
        return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        if 'timestamp' in df.columns:
            df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
            df['minute'] = pd.to_datetime(df['timestamp']).dt.minute
            df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
            
            # Market session indicators
            df['is_market_open'] = ((df['hour'] >= 9) & (df['hour'] < 16)).astype(int)
            df['is_lunch_time'] = ((df['hour'] >= 12) & (df['hour'] < 13)).astype(int)
        
        return df
    
    def prepare_features(self, df: pd.DataFrame, fit_scalers: bool = True) -> np.ndarray:
        """
        Prepare features for ML model training/prediction
        
        Args:
            df: DataFrame with features
            fit_scalers: Whether to fit scalers (True for training, False for prediction)
            
        Returns:
            Scaled feature array
        """
        feature_df = df[self.feature_names].copy()
        
        # Handle missing values
        feature_df = feature_df.fillna(method='ffill').fillna(0)
        
        # Scale features
        if fit_scalers:
            self.scalers['robust'] = RobustScaler()
            scaled_features = self.scalers['robust'].fit_transform(feature_df)
        else:
            if 'robust' in self.scalers:
                scaled_features = self.scalers['robust'].transform(feature_df)
            else:
                scaled_features = feature_df.values
        
        return scaled_features


class MLTradingStrategy(BaseStrategy):
    """
    Machine Learning-based trading strategy
    
    This strategy uses ML models to predict short-term price movements
    and generate trading signals based on market microstructure features.
    """
    
    def __init__(self, symbol: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize ML trading strategy
        
        Args:
            symbol: Trading symbol
            config: Strategy configuration
        """
        super().__init__(symbol, config)
        
        # ML components
        self.feature_engineer = MLFeatureEngineer()
        self.direction_model = None  # Predict price direction
        self.magnitude_model = None  # Predict price change magnitude
        self.models_trained = False
        
        # Strategy parameters
        self.lookback_periods = config.get('lookback_periods', [5, 10, 20, 50])
        self.prediction_horizon = config.get('prediction_horizon', 5)  # ticks ahead
        self.confidence_threshold = config.get('confidence_threshold', 0.6)
        self.position_size = config.get('position_size', 1000)
        self.max_position = config.get('max_position', 5000)
        
        # Training parameters
        self.retrain_frequency = config.get('retrain_frequency', 1000)  # trades
        self.min_training_samples = config.get('min_training_samples', 500)
        self.validation_split = config.get('validation_split', 0.2)
        
        # Data storage
        self.market_data_buffer = []
        self.max_buffer_size = 10000
        self.trades_since_retrain = 0
        
        # Performance tracking
        self.model_performance = {
            'direction_accuracy': 0.0,
            'magnitude_mse': 0.0,
            'last_retrain': None,
            'predictions_made': 0
        }
        
        self.logger.info(f"MLTradingStrategy initialized for {symbol}")
    
    @log_performance
    def generate_signals(self, market_data: MarketDataPoint) -> List[Order]:
        """
        Generate trading signals using ML predictions
        
        Args:
            market_data: Current market data point
            
        Returns:
            List of orders to execute
        """
        # Add to buffer
        self._update_data_buffer(market_data)
        
        # Check if we need to retrain
        if self._should_retrain():
            self._retrain_models()
        
        # Generate prediction if models are trained
        if not self.models_trained or len(self.market_data_buffer) < max(self.lookback_periods):
            return []
        
        # Create features
        df = self._buffer_to_dataframe()
        features_df = self.feature_engineer.create_features(df, self.lookback_periods)
        
        # Get latest features
        latest_features = self.feature_engineer.prepare_features(
            features_df.tail(1), fit_scalers=False
        )
        
        if latest_features.shape[0] == 0:
            return []
        
        # Make predictions
        direction_prob = self.direction_model.predict_proba(latest_features)[0]
        magnitude_pred = self.magnitude_model.predict(latest_features)[0]
        
        # Generate signals based on predictions
        orders = self._generate_orders_from_predictions(
            direction_prob, magnitude_pred, market_data
        )
        
        self.model_performance['predictions_made'] += 1
        
        return orders
    
    def _update_data_buffer(self, market_data: MarketDataPoint):
        """Update the market data buffer"""
        data_point = {
            'timestamp': market_data.timestamp,
            'price': market_data.price,
            'volume': market_data.volume,
            'best_bid': getattr(market_data, 'best_bid', None),
            'best_ask': getattr(market_data, 'best_ask', None),
            'bid_volume': getattr(market_data, 'bid_volume', None),
            'ask_volume': getattr(market_data, 'ask_volume', None),
        }
        
        self.market_data_buffer.append(data_point)
        
        # Maintain buffer size
        if len(self.market_data_buffer) > self.max_buffer_size:
            self.market_data_buffer = self.market_data_buffer[-self.max_buffer_size:]
    
    def _buffer_to_dataframe(self) -> pd.DataFrame:
        """Convert buffer to DataFrame"""
        return pd.DataFrame(self.market_data_buffer)
    
    def _should_retrain(self) -> bool:
        """Check if models should be retrained"""
        return (
            not self.models_trained or
            self.trades_since_retrain >= self.retrain_frequency or
            len(self.market_data_buffer) >= self.max_buffer_size
        )
    
    @log_performance
    def _retrain_models(self):
        """Retrain ML models with latest data"""
        if len(self.market_data_buffer) < self.min_training_samples:
            return
        
        self.logger.info("Retraining ML models...")
        
        try:
            # Prepare training data
            df = self._buffer_to_dataframe()
            features_df = self.feature_engineer.create_features(df, self.lookback_periods)
            
            # Create labels
            labels_df = self._create_labels(features_df)
            
            # Prepare features and labels
            X = self.feature_engineer.prepare_features(features_df, fit_scalers=True)
            y_direction = labels_df['direction'].values
            y_magnitude = labels_df['magnitude'].values
            
            # Remove NaN values
            valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y_direction) | np.isnan(y_magnitude))
            X = X[valid_mask]
            y_direction = y_direction[valid_mask]
            y_magnitude = y_magnitude[valid_mask]
            
            if len(X) < self.min_training_samples:
                return
            
            # Time series split for validation
            tscv = TimeSeriesSplit(n_splits=3)
            
            # Train direction model (classification)
            self.direction_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=20,
                random_state=42,
                n_jobs=-1
            )
            
            # Train magnitude model (regression)
            self.magnitude_model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
            
            # Cross-validation
            direction_scores = []
            magnitude_scores = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_dir_train, y_dir_val = y_direction[train_idx], y_direction[val_idx]
                y_mag_train, y_mag_val = y_magnitude[train_idx], y_magnitude[val_idx]
                
                # Train models
                self.direction_model.fit(X_train, y_dir_train)
                self.magnitude_model.fit(X_train, y_mag_train)
                
                # Validate
                dir_pred = self.direction_model.predict(X_val)
                mag_pred = self.magnitude_model.predict(X_val)
                
                direction_scores.append(accuracy_score(y_dir_val, dir_pred))
                magnitude_scores.append(np.mean((y_mag_val - mag_pred) ** 2))
            
            # Final training on all data
            self.direction_model.fit(X, y_direction)
            self.magnitude_model.fit(X, y_magnitude)
            
            # Update performance metrics
            self.model_performance.update({
                'direction_accuracy': np.mean(direction_scores),
                'magnitude_mse': np.mean(magnitude_scores),
                'last_retrain': datetime.now(),
            })
            
            self.models_trained = True
            self.trades_since_retrain = 0
            
            self.logger.info(
                f"Models retrained - Direction accuracy: {np.mean(direction_scores):.3f}, "
                f"Magnitude MSE: {np.mean(magnitude_scores):.6f}"
            )
            
        except Exception as e:
            self.logger.error(f"Model retraining failed: {e}")
    
    def _create_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create training labels from price data"""
        labels = pd.DataFrame(index=df.index)
        
        # Future price change
        future_price = df['price'].shift(-self.prediction_horizon)
        current_price = df['price']
        
        # Price change magnitude (log return)
        labels['magnitude'] = np.log(future_price / current_price)
        
        # Price direction (1 for up, 0 for down)
        labels['direction'] = (labels['magnitude'] > 0).astype(int)
        
        return labels
    
    def _generate_orders_from_predictions(self, direction_prob: np.ndarray, 
                                        magnitude_pred: float,
                                        market_data: MarketDataPoint) -> List[Order]:
        """Generate orders based on ML predictions"""
        orders = []
        
        # Get prediction confidence
        max_prob = np.max(direction_prob)
        predicted_direction = np.argmax(direction_prob)
        
        # Only trade if confidence is high enough
        if max_prob < self.confidence_threshold:
            return orders
        
        # Calculate position size based on confidence and magnitude
        confidence_factor = (max_prob - 0.5) * 2  # Scale to 0-1
        magnitude_factor = min(abs(magnitude_pred) * 100, 1.0)  # Cap at 1.0
        
        position_size = int(self.position_size * confidence_factor * magnitude_factor)
        position_size = min(position_size, self.max_position - abs(self.current_position))
        
        if position_size < 100:  # Minimum position size
            return orders
        
        # Determine order side
        if predicted_direction == 1:  # Predicted price increase
            side = OrderSide.BID
            # Use aggressive pricing for high confidence
            price = market_data.best_ask if max_prob > 0.8 else market_data.best_bid
        else:  # Predicted price decrease
            side = OrderSide.ASK
            price = market_data.best_bid if max_prob > 0.8 else market_data.best_ask
        
        # Create order
        order = Order.create_limit_order(
            symbol=self.symbol,
            side=side,
            volume=position_size,
            price=price,
            metadata={
                'strategy': 'ml_strategy',
                'confidence': max_prob,
                'predicted_magnitude': magnitude_pred,
                'model_version': self.model_performance['last_retrain']
            }
        )
        
        orders.append(order)
        self.trades_since_retrain += 1
        
        return orders
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get comprehensive strategy information"""
        base_info = super().get_strategy_info()
        
        ml_info = {
            'models_trained': self.models_trained,
            'model_performance': self.model_performance.copy(),
            'feature_count': len(self.feature_engineer.feature_names),
            'buffer_size': len(self.market_data_buffer),
            'trades_since_retrain': self.trades_since_retrain,
            'parameters': {
                'lookback_periods': self.lookback_periods,
                'prediction_horizon': self.prediction_horizon,
                'confidence_threshold': self.confidence_threshold,
                'retrain_frequency': self.retrain_frequency,
            }
        }
        
        base_info.update(ml_info)
        return base_info
    
    def save_models(self, filepath: str):
        """Save trained models to disk"""
        if self.models_trained:
            model_data = {
                'direction_model': self.direction_model,
                'magnitude_model': self.magnitude_model,
                'feature_engineer': self.feature_engineer,
                'model_performance': self.model_performance,
                'parameters': {
                    'lookback_periods': self.lookback_periods,
                    'prediction_horizon': self.prediction_horizon,
                    'confidence_threshold': self.confidence_threshold,
                }
            }
            joblib.dump(model_data, filepath)
            self.logger.info(f"Models saved to {filepath}")
    
    def load_models(self, filepath: str):
        """Load trained models from disk"""
        try:
            model_data = joblib.load(filepath)
            self.direction_model = model_data['direction_model']
            self.magnitude_model = model_data['magnitude_model']
            self.feature_engineer = model_data['feature_engineer']
            self.model_performance = model_data['model_performance']
            
            # Update parameters if available
            if 'parameters' in model_data:
                params = model_data['parameters']
                self.lookback_periods = params.get('lookback_periods', self.lookback_periods)
                self.prediction_horizon = params.get('prediction_horizon', self.prediction_horizon)
                self.confidence_threshold = params.get('confidence_threshold', self.confidence_threshold)
            
            self.models_trained = True
            self.logger.info(f"Models loaded from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to load models: {e}")
            self.models_trained = False


# Utility functions for ML strategy development
def backtest_ml_strategy(strategy: MLTradingStrategy, 
                        market_data: pd.DataFrame,
                        initial_capital: float = 100000) -> Dict[str, Any]:
    """
    Backtest ML strategy with walk-forward analysis
    
    Args:
        strategy: ML trading strategy instance
        market_data: Historical market data
        initial_capital: Starting capital
        
    Returns:
        Backtest results
    """
    results = {
        'trades': [],
        'pnl_series': [],
        'positions': [],
        'model_retrains': [],
        'performance_metrics': {}
    }
    
    capital = initial_capital
    position = 0
    
    for i, (_, row) in enumerate(market_data.iterrows()):
        # Create market data point
        market_point = MarketDataPoint(
            timestamp=row['timestamp'],
            price=row['price'],
            volume=row['volume'],
            best_bid=row.get('best_bid', row['price'] - 0.01),
            best_ask=row.get('best_ask', row['price'] + 0.01)
        )
        
        # Generate signals
        orders = strategy.generate_signals(market_point)
        
        # Execute orders (simplified)
        for order in orders:
            if order.side == OrderSide.BID and position < strategy.max_position:
                position += order.volume
                capital -= order.volume * order.price
                results['trades'].append({
                    'timestamp': row['timestamp'],
                    'side': 'buy',
                    'volume': order.volume,
                    'price': order.price,
                    'position': position
                })
            elif order.side == OrderSide.ASK and position > -strategy.max_position:
                position -= order.volume
                capital += order.volume * order.price
                results['trades'].append({
                    'timestamp': row['timestamp'],
                    'side': 'sell',
                    'volume': order.volume,
                    'price': order.price,
                    'position': position
                })
        
        # Track PnL
        current_value = capital + position * row['price']
        results['pnl_series'].append({
            'timestamp': row['timestamp'],
            'pnl': current_value - initial_capital,
            'position': position,
            'capital': capital
        })
        
        # Track model retrains
        if strategy.model_performance['last_retrain'] and len(results['model_retrains']) == 0:
            results['model_retrains'].append({
                'timestamp': row['timestamp'],
                'accuracy': strategy.model_performance['direction_accuracy'],
                'mse': strategy.model_performance['magnitude_mse']
            })
    
    # Calculate performance metrics
    pnl_df = pd.DataFrame(results['pnl_series'])
    if len(pnl_df) > 0:
        returns = pnl_df['pnl'].pct_change().dropna()
        results['performance_metrics'] = {
            'total_return': (pnl_df['pnl'].iloc[-1] / initial_capital) * 100,
            'sharpe_ratio': calculate_sharpe_ratio(returns),
            'max_drawdown': calculate_max_drawdown(pnl_df['pnl'].values),
            'num_trades': len(results['trades']),
            'win_rate': len([t for t in results['trades'] if t.get('pnl', 0) > 0]) / max(len(results['trades']), 1),
            'model_retrains': len(results['model_retrains'])
        }
    
    return results