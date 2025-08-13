"""
Chart Generation and Plotting Utilities for HFT Simulator

This module provides comprehensive charting capabilities for visualizing
HFT data, performance metrics, order book dynamics, and trading analytics.

Educational Notes:
- Visualization is crucial for understanding market microstructure
- Real-time charts help identify patterns and anomalies
- Different chart types serve different analytical purposes
- Interactive charts enable deeper exploration of data
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
from datetime import datetime, timedelta
import warnings

from src.utils.logger import get_logger
from src.engine.order_book import OrderBook
from src.engine.market_data import BookSnapshot
from src.performance.metrics import PerformanceMetrics
from src.performance.portfolio import Portfolio


class ChartType(Enum):
    """Types of charts available"""
    LINE = "line"
    CANDLESTICK = "candlestick"
    BAR = "bar"
    SCATTER = "scatter"
    HEATMAP = "heatmap"
    HISTOGRAM = "histogram"
    BOX = "box"
    VIOLIN = "violin"
    SURFACE = "surface"
    WATERFALL = "waterfall"


class ChartTheme(Enum):
    """Chart themes"""
    LIGHT = "plotly_white"
    DARK = "plotly_dark"
    PROFESSIONAL = "simple_white"
    COLORFUL = "plotly"


class ChartGenerator:
    """
    Comprehensive chart generation system for HFT analysis
    
    This class provides methods to create various types of charts
    for analyzing HFT data, performance metrics, and market dynamics.
    
    Key Features:
    - Real-time order book visualization
    - Performance and P&L charts
    - Risk analysis charts
    - Market microstructure analysis
    - Interactive dashboards
    - Export capabilities
    
    Educational Notes:
    - Different chart types reveal different aspects of market behavior
    - Time series charts show trends and patterns over time
    - Distribution charts reveal statistical properties
    - Correlation charts show relationships between variables
    - Real-time charts enable live monitoring and decision making
    """
    
    def __init__(self, theme: ChartTheme = ChartTheme.PROFESSIONAL):
        """
        Initialize chart generator
        
        Args:
            theme: Chart theme to use
        """
        self.theme = theme
        self.logger = get_logger(__name__)
        
        # Chart configuration
        self.default_width = 1200
        self.default_height = 600
        self.color_palette = px.colors.qualitative.Set1
        
        # Set default styling
        self._setup_styling()
        
        self.logger.info(f"Chart generator initialized with {theme.value} theme")
    
    def _setup_styling(self) -> None:
        """Setup default styling for charts"""
        # Matplotlib styling
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Default figure size
        plt.rcParams['figure.figsize'] = (12, 6)
        plt.rcParams['font.size'] = 10
    
    def create_price_chart(self, 
                          data: pd.DataFrame,
                          chart_type: ChartType = ChartType.CANDLESTICK,
                          title: str = "Price Chart",
                          show_volume: bool = True) -> go.Figure:
        """
        Create price chart (candlestick or line)
        
        Args:
            data: DataFrame with OHLCV data
            chart_type: Type of chart to create
            title: Chart title
            show_volume: Whether to show volume subplot
            
        Returns:
            Plotly figure
        """
        
        # Create subplots
        if show_volume:
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                subplot_titles=[title, 'Volume'],
                row_heights=[0.7, 0.3]
            )
        else:
            fig = go.Figure()
        
        # Add price data
        if chart_type == ChartType.CANDLESTICK:
            candlestick = go.Candlestick(
                x=data.index,
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'],
                name='Price'
            )
            
            if show_volume:
                fig.add_trace(candlestick, row=1, col=1)
            else:
                fig.add_trace(candlestick)
        
        elif chart_type == ChartType.LINE:
            line = go.Scatter(
                x=data.index,
                y=data['close'],
                mode='lines',
                name='Price',
                line=dict(color='blue', width=2)
            )
            
            if show_volume:
                fig.add_trace(line, row=1, col=1)
            else:
                fig.add_trace(line)
        
        # Add volume if requested
        if show_volume and 'volume' in data.columns:
            volume_colors = ['red' if close < open else 'green' 
                           for close, open in zip(data['close'], data['open'])]
            
            volume_bar = go.Bar(
                x=data.index,
                y=data['volume'],
                name='Volume',
                marker_color=volume_colors,
                opacity=0.7
            )
            
            fig.add_trace(volume_bar, row=2, col=1)
        
        # Update layout
        fig.update_layout(
            title=title,
            template=self.theme.value,
            width=self.default_width,
            height=self.default_height,
            xaxis_rangeslider_visible=False
        )
        
        return fig
    
    def create_order_book_chart(self, 
                               order_book: OrderBook,
                               depth: int = 10) -> go.Figure:
        """
        Create order book depth chart
        
        Args:
            order_book: OrderBook instance
            depth: Number of levels to show
            
        Returns:
            Plotly figure
        """
        
        # Get order book data
        bids = order_book.get_bids(depth)
        asks = order_book.get_asks(depth)
        
        if not bids or not asks:
            # Create empty chart
            fig = go.Figure()
            fig.add_annotation(
                text="No order book data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        # Prepare data
        bid_prices = [price for price, _ in bids]
        bid_volumes = [volume for _, volume in bids]
        bid_cumulative = np.cumsum(bid_volumes)
        
        ask_prices = [price for price, _ in asks]
        ask_volumes = [volume for _, volume in asks]
        ask_cumulative = np.cumsum(ask_volumes)
        
        # Create figure
        fig = go.Figure()
        
        # Add bid side
        fig.add_trace(go.Scatter(
            x=bid_prices,
            y=bid_cumulative,
            mode='lines+markers',
            name='Bids',
            line=dict(color='green', width=3),
            fill='tonexty',
            fillcolor='rgba(0,255,0,0.2)'
        ))
        
        # Add ask side
        fig.add_trace(go.Scatter(
            x=ask_prices,
            y=ask_cumulative,
            mode='lines+markers',
            name='Asks',
            line=dict(color='red', width=3),
            fill='tonexty',
            fillcolor='rgba(255,0,0,0.2)'
        ))
        
        # Update layout
        fig.update_layout(
            title='Order Book Depth',
            xaxis_title='Price',
            yaxis_title='Cumulative Volume',
            template=self.theme.value,
            width=self.default_width,
            height=self.default_height,
            hovermode='x unified'
        )
        
        return fig
    
    def create_performance_chart(self, 
                                metrics: PerformanceMetrics,
                                portfolio_values: List[Tuple[pd.Timestamp, float]]) -> go.Figure:
        """
        Create comprehensive performance chart
        
        Args:
            metrics: Performance metrics
            portfolio_values: List of (timestamp, value) tuples
            
        Returns:
            Plotly figure
        """
        
        if not portfolio_values:
            fig = go.Figure()
            fig.add_annotation(
                text="No performance data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        # Convert to DataFrame
        df = pd.DataFrame(portfolio_values, columns=['timestamp', 'value'])
        df['returns'] = df['value'].pct_change()
        df['cumulative_returns'] = (df['value'] / df['value'].iloc[0] - 1) * 100
        
        # Calculate rolling metrics
        df['rolling_sharpe'] = df['returns'].rolling(252).apply(
            lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0
        )
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Portfolio Value', 'Cumulative Returns %',
                'Daily Returns Distribution', 'Rolling Sharpe Ratio',
                'Drawdown', 'Return vs Risk'
            ],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Portfolio value
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['value'], 
                      name='Portfolio Value', line=dict(color='blue')),
            row=1, col=1
        )
        
        # Cumulative returns
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['cumulative_returns'], 
                      name='Cumulative Returns', line=dict(color='green')),
            row=1, col=2
        )
        
        # Returns distribution
        fig.add_trace(
            go.Histogram(x=df['returns'].dropna(), name='Daily Returns',
                        nbinsx=50, opacity=0.7),
            row=2, col=1
        )
        
        # Rolling Sharpe
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['rolling_sharpe'], 
                      name='Rolling Sharpe', line=dict(color='orange')),
            row=2, col=2
        )
        
        # Drawdown calculation
        running_max = df['value'].expanding().max()
        drawdown = (df['value'] - running_max) / running_max * 100
        
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=drawdown, 
                      name='Drawdown', line=dict(color='red'),
                      fill='tonexty', fillcolor='rgba(255,0,0,0.2)'),
            row=3, col=1
        )
        
        # Return vs Risk scatter
        if len(df) > 252:  # Need enough data for meaningful calculation
            monthly_returns = df['returns'].rolling(21).sum()  # ~monthly
            monthly_vol = df['returns'].rolling(21).std() * np.sqrt(21)
            
            fig.add_trace(
                go.Scatter(x=monthly_vol, y=monthly_returns, 
                          mode='markers', name='Return vs Risk',
                          marker=dict(color='purple', size=5)),
                row=3, col=2
            )
        
        # Update layout
        fig.update_layout(
            title='Performance Dashboard',
            template=self.theme.value,
            width=self.default_width * 1.5,
            height=self.default_height * 1.5,
            showlegend=False
        )
        
        return fig
    
    def create_risk_chart(self, 
                         risk_metrics: Dict[str, Any],
                         risk_events: List[Dict[str, Any]]) -> go.Figure:
        """
        Create risk analysis chart
        
        Args:
            risk_metrics: Dictionary of risk metrics
            risk_events: List of risk events
            
        Returns:
            Plotly figure
        """
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Risk Metrics Overview', 'Risk Events Timeline',
                'Position Risk Breakdown', 'VaR Analysis'
            ]
        )
        
        # Risk metrics overview (gauge charts)
        metrics_to_show = ['current_drawdown', 'portfolio_volatility', 'concentration_risk']
        colors = ['red', 'orange', 'yellow']
        
        for i, (metric, color) in enumerate(zip(metrics_to_show, colors)):
            if metric in risk_metrics:
                value = risk_metrics[metric] * 100  # Convert to percentage
                
                gauge = go.Indicator(
                    mode="gauge+number",
                    value=value,
                    title={'text': metric.replace('_', ' ').title()},
                    gauge={'axis': {'range': [None, 50]},
                           'bar': {'color': color},
                           'steps': [{'range': [0, 10], 'color': "lightgray"},
                                   {'range': [10, 25], 'color': "gray"}],
                           'threshold': {'line': {'color': "red", 'width': 4},
                                       'thickness': 0.75, 'value': 20}}
                )
                
                # Add to appropriate subplot position
                if i == 0:
                    fig.add_trace(gauge, row=1, col=1)
        
        # Risk events timeline
        if risk_events:
            event_df = pd.DataFrame(risk_events)
            event_df['timestamp'] = pd.to_datetime(event_df['timestamp'])
            
            # Count events by hour
            event_counts = event_df.groupby(
                event_df['timestamp'].dt.floor('H')
            ).size().reset_index()
            event_counts.columns = ['hour', 'count']
            
            fig.add_trace(
                go.Bar(x=event_counts['hour'], y=event_counts['count'],
                      name='Risk Events', marker_color='red'),
                row=1, col=2
            )
        
        # Position risk breakdown (if available)
        if 'position_size_risk' in risk_metrics:
            position_risks = risk_metrics['position_size_risk']
            if position_risks:
                symbols = list(position_risks.keys())
                risks = [position_risks[symbol] * 100 for symbol in symbols]
                
                fig.add_trace(
                    go.Bar(x=symbols, y=risks, name='Position Risk %',
                          marker_color='orange'),
                    row=2, col=1
                )
        
        # VaR analysis (simplified)
        if 'portfolio_var' in risk_metrics:
            var_value = risk_metrics['portfolio_var'] * 100
            
            # Create a simple VaR visualization
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=var_value,
                    title={'text': 'Portfolio VaR (%)'},
                    gauge={'axis': {'range': [None, 10]},
                           'bar': {'color': 'darkred'},
                           'steps': [{'range': [0, 2], 'color': "lightgray"},
                                   {'range': [2, 5], 'color': "gray"}],
                           'threshold': {'line': {'color': "red", 'width': 4},
                                       'thickness': 0.75, 'value': 5}}
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title='Risk Analysis Dashboard',
            template=self.theme.value,
            width=self.default_width * 1.5,
            height=self.default_height * 1.2,
            showlegend=False
        )
        
        return fig
    
    def create_trade_analysis_chart(self, 
                                   trades: List[Dict[str, Any]]) -> go.Figure:
        """
        Create trade analysis chart
        
        Args:
            trades: List of trade dictionaries
            
        Returns:
            Plotly figure
        """
        
        if not trades:
            fig = go.Figure()
            fig.add_annotation(
                text="No trade data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        # Convert to DataFrame
        df = pd.DataFrame(trades)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['pnl'] = df.get('pnl', 0)  # Assume PnL is available
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Trades Over Time', 'Trade Size Distribution',
                'P&L Distribution', 'Trading Activity by Hour'
            ]
        )
        
        # Trades over time
        trade_counts = df.groupby(df['timestamp'].dt.floor('H')).size().reset_index()
        trade_counts.columns = ['hour', 'count']
        
        fig.add_trace(
            go.Scatter(x=trade_counts['hour'], y=trade_counts['count'],
                      mode='lines+markers', name='Trades per Hour'),
            row=1, col=1
        )
        
        # Trade size distribution
        fig.add_trace(
            go.Histogram(x=df['volume'], name='Trade Size',
                        nbinsx=30, opacity=0.7),
            row=1, col=2
        )
        
        # P&L distribution
        if 'pnl' in df.columns:
            fig.add_trace(
                go.Histogram(x=df['pnl'], name='Trade P&L',
                            nbinsx=30, opacity=0.7),
                row=2, col=1
            )
        
        # Trading activity by hour
        hourly_activity = df.groupby('hour').size().reset_index()
        hourly_activity.columns = ['hour', 'trades']
        
        fig.add_trace(
            go.Bar(x=hourly_activity['hour'], y=hourly_activity['trades'],
                  name='Trades by Hour', marker_color='blue'),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title='Trade Analysis Dashboard',
            template=self.theme.value,
            width=self.default_width * 1.5,
            height=self.default_height * 1.2,
            showlegend=False
        )
        
        return fig
    
    def create_correlation_heatmap(self, 
                                  correlation_matrix: pd.DataFrame) -> go.Figure:
        """
        Create correlation heatmap
        
        Args:
            correlation_matrix: Correlation matrix DataFrame
            
        Returns:
            Plotly figure
        """
        
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.index,
            colorscale='RdBu',
            zmid=0,
            text=correlation_matrix.values,
            texttemplate="%{text:.2f}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title='Asset Correlation Matrix',
            template=self.theme.value,
            width=self.default_width,
            height=self.default_height
        )
        
        return fig
    
    def create_market_microstructure_chart(self, 
                                         book_snapshots: List[BookSnapshot]) -> go.Figure:
        """
        Create market microstructure analysis chart
        
        Args:
            book_snapshots: List of order book snapshots
            
        Returns:
            Plotly figure
        """
        
        if not book_snapshots:
            fig = go.Figure()
            fig.add_annotation(
                text="No market data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        # Extract data from snapshots
        timestamps = [snap.timestamp for snap in book_snapshots]
        spreads = []
        mid_prices = []
        
        for snap in book_snapshots:
            if snap.best_bid and snap.best_ask:
                spread = snap.best_ask[0] - snap.best_bid[0]
                mid_price = (snap.best_ask[0] + snap.best_bid[0]) / 2
                spreads.append(spread)
                mid_prices.append(mid_price)
            else:
                spreads.append(None)
                mid_prices.append(None)
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            subplot_titles=['Mid Price', 'Bid-Ask Spread', 'Spread Distribution'],
            vertical_spacing=0.1
        )
        
        # Mid price
        fig.add_trace(
            go.Scatter(x=timestamps, y=mid_prices, 
                      mode='lines', name='Mid Price',
                      line=dict(color='blue')),
            row=1, col=1
        )
        
        # Bid-ask spread
        fig.add_trace(
            go.Scatter(x=timestamps, y=spreads, 
                      mode='lines', name='Spread',
                      line=dict(color='red')),
            row=2, col=1
        )
        
        # Spread distribution
        valid_spreads = [s for s in spreads if s is not None]
        if valid_spreads:
            fig.add_trace(
                go.Histogram(x=valid_spreads, name='Spread Distribution',
                            nbinsx=30, opacity=0.7),
                row=3, col=1
            )
        
        # Update layout
        fig.update_layout(
            title='Market Microstructure Analysis',
            template=self.theme.value,
            width=self.default_width,
            height=self.default_height * 1.5,
            showlegend=False
        )
        
        return fig
    
    def save_chart(self, 
                   fig: go.Figure, 
                   filename: str, 
                   format: str = 'html') -> None:
        """
        Save chart to file
        
        Args:
            fig: Plotly figure to save
            filename: Output filename
            format: Output format ('html', 'png', 'pdf', 'svg')
        """
        
        try:
            if format == 'html':
                fig.write_html(filename)
            elif format == 'png':
                fig.write_image(filename, format='png')
            elif format == 'pdf':
                fig.write_image(filename, format='pdf')
            elif format == 'svg':
                fig.write_image(filename, format='svg')
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            self.logger.info(f"Chart saved to {filename}")
            
        except Exception as e:
            self.logger.error(f"Failed to save chart: {e}")
            raise
    
    def create_custom_chart(self, 
                           data: pd.DataFrame,
                           chart_config: Dict[str, Any]) -> go.Figure:
        """
        Create custom chart based on configuration
        
        Args:
            data: Data to plot
            chart_config: Chart configuration dictionary
            
        Returns:
            Plotly figure
        """
        
        chart_type = ChartType(chart_config.get('type', 'line'))
        title = chart_config.get('title', 'Custom Chart')
        x_col = chart_config.get('x_column', data.columns[0])
        y_col = chart_config.get('y_column', data.columns[1])
        
        fig = go.Figure()
        
        if chart_type == ChartType.LINE:
            fig.add_trace(go.Scatter(
                x=data[x_col], y=data[y_col],
                mode='lines', name=y_col
            ))
        
        elif chart_type == ChartType.BAR:
            fig.add_trace(go.Bar(
                x=data[x_col], y=data[y_col],
                name=y_col
            ))
        
        elif chart_type == ChartType.SCATTER:
            fig.add_trace(go.Scatter(
                x=data[x_col], y=data[y_col],
                mode='markers', name=y_col
            ))
        
        elif chart_type == ChartType.HISTOGRAM:
            fig.add_trace(go.Histogram(
                x=data[y_col], name=y_col
            ))
        
        # Update layout
        fig.update_layout(
            title=title,
            template=self.theme.value,
            width=self.default_width,
            height=self.default_height
        )
        
        return fig


# Utility functions
def create_chart_generator(theme: ChartTheme = ChartTheme.PROFESSIONAL) -> ChartGenerator:
    """Create a chart generator with specified theme"""
    return ChartGenerator(theme=theme)


def quick_plot(data: pd.DataFrame, 
               chart_type: ChartType = ChartType.LINE,
               title: str = "Quick Plot") -> go.Figure:
    """
    Quick plotting function for rapid visualization
    
    Args:
        data: Data to plot
        chart_type: Type of chart
        title: Chart title
        
    Returns:
        Plotly figure
    """
    generator = ChartGenerator()
    
    config = {
        'type': chart_type.value,
        'title': title,
        'x_column': data.columns[0],
        'y_column': data.columns[1] if len(data.columns) > 1 else data.columns[0]
    }
    
    return generator.create_custom_chart(data, config)