"""
Report Generation System for HFT Simulator

This module provides comprehensive reporting capabilities for generating
detailed analysis reports, performance summaries, and research documentation.

Educational Notes:
- Reports provide detailed analysis beyond real-time dashboards
- Different report types serve different audiences and purposes
- Automated reporting enables regular performance reviews
- Export capabilities allow sharing and archiving results
- Professional formatting enhances credibility and readability
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import io
from pathlib import Path

from src.utils.logger import get_logger
from src.visualization.charts import ChartGenerator, ChartType, ChartTheme
from src.performance.metrics import PerformanceAnalyzer, PerformanceMetrics
from src.performance.portfolio import Portfolio
from src.performance.risk_manager import RiskManager


class ReportType(Enum):
    """Types of reports available"""
    PERFORMANCE_SUMMARY = "performance_summary"
    RISK_ANALYSIS = "risk_analysis"
    TRADING_ACTIVITY = "trading_activity"
    MARKET_ANALYSIS = "market_analysis"
    STRATEGY_COMPARISON = "strategy_comparison"
    COMPREHENSIVE = "comprehensive"
    EXECUTIVE_SUMMARY = "executive_summary"
    RESEARCH_REPORT = "research_report"


class ReportFormat(Enum):
    """Report output formats"""
    HTML = "html"
    PDF = "pdf"
    JSON = "json"
    CSV = "csv"
    MARKDOWN = "markdown"


@dataclass
class ReportConfig:
    """Configuration for report generation"""
    
    # Content settings
    include_charts: bool = True
    include_tables: bool = True
    include_statistics: bool = True
    include_recommendations: bool = True
    
    # Formatting settings
    theme: ChartTheme = ChartTheme.PROFESSIONAL
    page_size: str = "A4"
    font_size: int = 11
    
    # Data settings
    analysis_period_days: int = 30
    benchmark_symbol: Optional[str] = None
    
    # Output settings
    output_directory: str = "reports"
    filename_prefix: str = "hft_report"
    include_timestamp: bool = True


class ReportGenerator:
    """
    Comprehensive report generation system
    
    This class generates detailed reports for various aspects of HFT
    trading performance, risk analysis, and market behavior.
    
    Key Features:
    - Multiple report types and formats
    - Professional formatting and styling
    - Interactive charts and visualizations
    - Statistical analysis and insights
    - Automated recommendations
    - Export capabilities
    - Template customization
    
    Educational Notes:
    - Reports provide deeper analysis than real-time dashboards
    - Different stakeholders need different types of reports
    - Regular reporting enables continuous improvement
    - Professional presentation enhances credibility
    - Automated insights help identify patterns and opportunities
    """
    
    def __init__(self, config: ReportConfig = None):
        """
        Initialize report generator
        
        Args:
            config: Report configuration
        """
        self.config = config or ReportConfig()
        self.logger = get_logger(__name__)
        
        # Chart generator for visualizations
        self.chart_generator = ChartGenerator(theme=self.config.theme)
        
        # Report templates
        self.templates = self._load_templates()
        
        # Ensure output directory exists
        Path(self.config.output_directory).mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Report generator initialized")
    
    def _load_templates(self) -> Dict[str, str]:
        """Load report templates"""
        # Basic HTML template
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>{title}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ text-align: center; margin-bottom: 30px; }}
                .section {{ margin-bottom: 30px; }}
                .metric {{ display: inline-block; margin: 10px; padding: 15px; 
                          border: 1px solid #ddd; border-radius: 5px; }}
                .chart {{ text-align: center; margin: 20px 0; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .positive {{ color: green; }}
                .negative {{ color: red; }}
                .warning {{ color: orange; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{title}</h1>
                <p>Generated on {timestamp}</p>
            </div>
            {content}
        </body>
        </html>
        """
        
        return {
            'html': html_template
        }
    
    def generate_performance_report(self, 
                                   portfolio: Portfolio,
                                   period_days: int = None) -> Dict[str, Any]:
        """
        Generate comprehensive performance report
        
        Args:
            portfolio: Portfolio to analyze
            period_days: Analysis period in days
            
        Returns:
            Report data dictionary
        """
        period_days = period_days or self.config.analysis_period_days
        
        # Get performance metrics
        metrics = portfolio.calculate_performance_metrics()
        summary = portfolio.get_portfolio_summary()
        positions = portfolio.get_position_breakdown()
        
        # Calculate additional statistics
        additional_stats = self._calculate_additional_performance_stats(portfolio, period_days)
        
        report_data = {
            'report_type': ReportType.PERFORMANCE_SUMMARY.value,
            'generation_time': pd.Timestamp.now().isoformat(),
            'analysis_period_days': period_days,
            'portfolio_summary': summary,
            'performance_metrics': metrics.to_dict(),
            'position_breakdown': positions.to_dict('records') if not positions.empty else [],
            'additional_statistics': additional_stats,
            'key_insights': self._generate_performance_insights(metrics, summary),
            'recommendations': self._generate_performance_recommendations(metrics, summary)
        }
        
        self.logger.info(f"Performance report generated for {portfolio.name}")
        return report_data
    
    def generate_risk_report(self, 
                            risk_manager: RiskManager,
                            period_days: int = None) -> Dict[str, Any]:
        """
        Generate comprehensive risk analysis report
        
        Args:
            risk_manager: Risk manager to analyze
            period_days: Analysis period in days
            
        Returns:
            Report data dictionary
        """
        period_days = period_days or self.config.analysis_period_days
        
        # Get risk data
        risk_summary = risk_manager.get_risk_summary()
        position_breakdown = risk_manager.get_position_risk_breakdown()
        recent_events = risk_manager.get_risk_events(hours=period_days * 24)
        
        # Stress test scenarios
        stress_scenarios = {
            'market_crash_10': {symbol: -0.10 for symbol in risk_manager.positions.keys()},
            'market_crash_20': {symbol: -0.20 for symbol in risk_manager.positions.keys()},
            'volatility_spike': {symbol: np.random.normal(0, 0.05) for symbol in risk_manager.positions.keys()}
        }
        stress_results = risk_manager.stress_test(stress_scenarios)
        
        report_data = {
            'report_type': ReportType.RISK_ANALYSIS.value,
            'generation_time': pd.Timestamp.now().isoformat(),
            'analysis_period_days': period_days,
            'risk_summary': risk_summary,
            'position_breakdown': position_breakdown.to_dict('records') if not position_breakdown.empty else [],
            'risk_events': [
                {
                    'timestamp': event.timestamp.isoformat(),
                    'risk_type': event.risk_type.value,
                    'level': event.risk_level.value,
                    'description': event.description,
                    'current_value': event.current_value,
                    'threshold': event.threshold
                }
                for event in recent_events
            ],
            'stress_test_results': stress_results,
            'risk_insights': self._generate_risk_insights(risk_summary, recent_events),
            'risk_recommendations': self._generate_risk_recommendations(risk_summary, recent_events)
        }
        
        self.logger.info("Risk analysis report generated")
        return report_data
    
    def generate_trading_report(self, 
                               portfolio: Portfolio,
                               trades: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate trading activity report
        
        Args:
            portfolio: Portfolio to analyze
            trades: List of trade data
            
        Returns:
            Report data dictionary
        """
        
        # Get trading data
        if trades is None:
            trades = [trade.__dict__ for trade in portfolio.all_trades] if hasattr(portfolio, 'all_trades') else []
        
        # Calculate trading statistics
        trading_stats = self._calculate_trading_statistics(trades)
        
        # Analyze trading patterns
        trading_patterns = self._analyze_trading_patterns(trades)
        
        report_data = {
            'report_type': ReportType.TRADING_ACTIVITY.value,
            'generation_time': pd.Timestamp.now().isoformat(),
            'trading_statistics': trading_stats,
            'trading_patterns': trading_patterns,
            'recent_trades': trades[-50:] if trades else [],  # Last 50 trades
            'trading_insights': self._generate_trading_insights(trading_stats, trading_patterns),
            'trading_recommendations': self._generate_trading_recommendations(trading_stats)
        }
        
        self.logger.info("Trading activity report generated")
        return report_data
    
    def generate_comprehensive_report(self, 
                                     portfolio: Portfolio,
                                     risk_manager: RiskManager = None,
                                     trades: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate comprehensive report combining all analyses
        
        Args:
            portfolio: Portfolio to analyze
            risk_manager: Risk manager to analyze
            trades: List of trade data
            
        Returns:
            Comprehensive report data
        """
        
        # Generate individual reports
        performance_report = self.generate_performance_report(portfolio)
        trading_report = self.generate_trading_report(portfolio, trades)
        
        risk_report = {}
        if risk_manager:
            risk_report = self.generate_risk_report(risk_manager)
        
        # Combine into comprehensive report
        comprehensive_data = {
            'report_type': ReportType.COMPREHENSIVE.value,
            'generation_time': pd.Timestamp.now().isoformat(),
            'executive_summary': self._generate_executive_summary(
                performance_report, risk_report, trading_report
            ),
            'performance_analysis': performance_report,
            'risk_analysis': risk_report,
            'trading_analysis': trading_report,
            'overall_recommendations': self._generate_overall_recommendations(
                performance_report, risk_report, trading_report
            )
        }
        
        self.logger.info("Comprehensive report generated")
        return comprehensive_data
    
    def _calculate_additional_performance_stats(self, 
                                              portfolio: Portfolio, 
                                              period_days: int) -> Dict[str, Any]:
        """Calculate additional performance statistics"""
        
# Get portfolio value history
        if hasattr(portfolio, 'value_history') and portfolio.value_history:
            values = [v for t, v in portfolio.value_history[-period_days * 24:]]  # Assuming hourly data
            
            if len(values) > 1:
                returns = np.diff(values) / values[:-1]
                
                stats = {
                    'total_return_pct': (values[-1] - values[0]) / values[0] * 100 if values[0] > 0 else 0,
                    'annualized_return': np.mean(returns) * 252 * 100,  # Assuming daily returns
                    'volatility': np.std(returns) * np.sqrt(252) * 100,
                    'best_day': np.max(returns) * 100 if len(returns) > 0 else 0,
                    'worst_day': np.min(returns) * 100 if len(returns) > 0 else 0,
                    'positive_days': len([r for r in returns if r > 0]) / len(returns) * 100 if returns else 0,
                    'average_daily_return': np.mean(returns) * 100 if returns else 0,
                    'skewness': float(pd.Series(returns).skew()) if len(returns) > 2 else 0,
                    'kurtosis': float(pd.Series(returns).kurtosis()) if len(returns) > 3 else 0
                }
            else:
                stats = {key: 0 for key in ['total_return_pct', 'annualized_return', 'volatility', 
                                          'best_day', 'worst_day', 'positive_days', 
                                          'average_daily_return', 'skewness', 'kurtosis']}
        else:
            stats = {key: 0 for key in ['total_return_pct', 'annualized_return', 'volatility', 
                                      'best_day', 'worst_day', 'positive_days', 
                                      'average_daily_return', 'skewness', 'kurtosis']}
        
        return stats
    
    def _calculate_trading_statistics(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate comprehensive trading statistics"""
        
        if not trades:
            return {
                'total_trades': 0,
                'total_volume': 0,
                'average_trade_size': 0,
                'largest_trade': 0,
                'smallest_trade': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'average_holding_time': 0,
                'trades_per_day': 0
            }
        
        df = pd.DataFrame(trades)
        
        # Basic statistics
        stats = {
            'total_trades': len(trades),
            'total_volume': df['volume'].sum() if 'volume' in df.columns else 0,
            'average_trade_size': df['volume'].mean() if 'volume' in df.columns else 0,
            'largest_trade': df['volume'].max() if 'volume' in df.columns else 0,
            'smallest_trade': df['volume'].min() if 'volume' in df.columns else 0,
        }
        
        # P&L statistics
        if 'pnl' in df.columns:
            winning_trades = df[df['pnl'] > 0]
            losing_trades = df[df['pnl'] < 0]
            
            stats.update({
                'win_rate': len(winning_trades) / len(df) * 100,
                'profit_factor': abs(winning_trades['pnl'].sum() / losing_trades['pnl'].sum()) if len(losing_trades) > 0 and losing_trades['pnl'].sum() != 0 else float('inf'),
                'average_win': winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0,
                'average_loss': losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0,
                'largest_win': winning_trades['pnl'].max() if len(winning_trades) > 0 else 0,
                'largest_loss': losing_trades['pnl'].min() if len(losing_trades) > 0 else 0
            })
        else:
            stats.update({
                'win_rate': 0,
                'profit_factor': 0,
                'average_win': 0,
                'average_loss': 0,
                'largest_win': 0,
                'largest_loss': 0
            })
        
        # Time-based statistics
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            time_span = (df['timestamp'].max() - df['timestamp'].min()).days
            stats['trades_per_day'] = len(df) / max(time_span, 1)
            
            # Trading hours analysis
            df['hour'] = df['timestamp'].dt.hour
            stats['most_active_hour'] = int(df['hour'].mode().iloc[0]) if not df['hour'].mode().empty else 0
            stats['trading_hours_spread'] = int(df['hour'].nunique())
        else:
            stats.update({
                'trades_per_day': 0,
                'most_active_hour': 0,
                'trading_hours_spread': 0
            })
        
        return stats
    
    def _analyze_trading_patterns(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze trading patterns and behaviors"""
        
        if not trades:
            return {'patterns': [], 'insights': []}
        
        df = pd.DataFrame(trades)
        patterns = {}
        
        # Volume patterns
        if 'volume' in df.columns:
            patterns['volume_trend'] = 'increasing' if df['volume'].corr(range(len(df))) > 0.1 else 'decreasing' if df['volume'].corr(range(len(df))) < -0.1 else 'stable'
            patterns['volume_volatility'] = float(df['volume'].std() / df['volume'].mean()) if df['volume'].mean() > 0 else 0
        
        # Time patterns
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            
            # Most active periods
            hourly_counts = df['hour'].value_counts()
            patterns['peak_trading_hours'] = hourly_counts.head(3).index.tolist()
            
            daily_counts = df['day_of_week'].value_counts()
            patterns['most_active_days'] = daily_counts.head(3).index.tolist()
        
        # Symbol patterns
        if 'symbol' in df.columns:
            symbol_counts = df['symbol'].value_counts()
            patterns['most_traded_symbols'] = symbol_counts.head(5).to_dict()
            patterns['symbol_concentration'] = symbol_counts.iloc[0] / len(df) * 100 if len(symbol_counts) > 0 else 0
        
        return patterns
    
    def _generate_performance_insights(self, 
                                     metrics: PerformanceMetrics, 
                                     summary: Dict[str, Any]) -> List[str]:
        """Generate performance insights"""
        
        insights = []
        
        # Return analysis
        if summary.get('return_pct', 0) > 10:
            insights.append("Strong positive returns achieved during the analysis period")
        elif summary.get('return_pct', 0) < -5:
            insights.append("Significant losses incurred - review strategy and risk management")
        
        # Sharpe ratio analysis
        if metrics.sharpe_ratio > 2:
            insights.append("Excellent risk-adjusted returns with Sharpe ratio above 2.0")
        elif metrics.sharpe_ratio < 0:
            insights.append("Negative Sharpe ratio indicates poor risk-adjusted performance")
        
        # Drawdown analysis
        if metrics.max_drawdown > 0.15:
            insights.append("High maximum drawdown suggests need for better risk controls")
        elif metrics.max_drawdown < 0.05:
            insights.append("Low drawdown indicates good risk management")
        
        # Volatility analysis
        if metrics.annualized_volatility > 0.3:
            insights.append("High volatility - consider position sizing adjustments")
        
        # Win rate analysis
        if metrics.win_rate > 0.6:
            insights.append("High win rate suggests effective trade selection")
        elif metrics.win_rate < 0.4:
            insights.append("Low win rate - review entry and exit criteria")
        
        return insights
    
    def _generate_risk_insights(self, 
                               risk_summary: Dict[str, Any], 
                               risk_events: List[Any]) -> List[str]:
        """Generate risk analysis insights"""
        
        insights = []
        
        # Drawdown insights
        current_dd = risk_summary.get('current_drawdown', 0)
        max_dd = risk_summary.get('max_drawdown', 0)
        
        if current_dd > 0.1:
            insights.append("Currently experiencing significant drawdown - consider reducing position sizes")
        
        if max_dd > 0.2:
            insights.append("Historical maximum drawdown is high - implement stricter risk limits")
        
        # Volatility insights
        volatility = risk_summary.get('portfolio_volatility', 0)
        if volatility > 0.25:
            insights.append("Portfolio volatility is elevated - diversification may help reduce risk")
        
        # Risk events insights
        if len(risk_events) > 10:
            insights.append("Frequent risk events suggest need for parameter adjustment")
        
        # Concentration insights
        concentration = risk_summary.get('concentration_risk', 0)
        if concentration > 0.3:
            insights.append("High portfolio concentration increases risk - consider diversification")
        
        return insights
    
    def _generate_trading_insights(self, 
                                  trading_stats: Dict[str, Any], 
                                  trading_patterns: Dict[str, Any]) -> List[str]:
        """Generate trading activity insights"""
        
        insights = []
        
        # Win rate insights
        win_rate = trading_stats.get('win_rate', 0)
        if win_rate > 60:
            insights.append("High win rate indicates effective trade selection")
        elif win_rate < 40:
            insights.append("Low win rate suggests need to improve entry/exit timing")
        
        # Profit factor insights
        profit_factor = trading_stats.get('profit_factor', 0)
        if profit_factor > 2:
            insights.append("Strong profit factor shows good risk/reward management")
        elif profit_factor < 1:
            insights.append("Profit factor below 1 indicates losses exceed profits")
        
        # Volume patterns
        volume_trend = trading_patterns.get('volume_trend', 'stable')
        if volume_trend == 'increasing':
            insights.append("Trading volume is increasing over time")
        elif volume_trend == 'decreasing':
            insights.append("Trading volume is decreasing - may indicate reduced opportunities")
        
        # Time patterns
        peak_hours = trading_patterns.get('peak_trading_hours', [])
        if peak_hours:
            insights.append(f"Most active trading occurs during hours: {peak_hours}")
        
        return insights
    
    def _generate_performance_recommendations(self, 
                                            metrics: PerformanceMetrics, 
                                            summary: Dict[str, Any]) -> List[str]:
        """Generate performance improvement recommendations"""
        
        recommendations = []
        
        # Sharpe ratio recommendations
        if metrics.sharpe_ratio < 1:
            recommendations.append("Consider improving risk-adjusted returns by optimizing position sizing")
        
        # Drawdown recommendations
        if metrics.max_drawdown > 0.1:
            recommendations.append("Implement stricter stop-loss rules to limit drawdowns")
        
        # Volatility recommendations
        if metrics.annualized_volatility > 0.25:
            recommendations.append("Reduce portfolio volatility through better diversification")
        
        # Return recommendations
        if summary.get('return_pct', 0) < 5:
            recommendations.append("Explore additional alpha sources to improve returns")
        
        return recommendations
    
    def _generate_risk_recommendations(self, 
                                     risk_summary: Dict[str, Any], 
                                     risk_events: List[Any]) -> List[str]:
        """Generate risk management recommendations"""
        
        recommendations = []
        
        # Drawdown recommendations
        if risk_summary.get('current_drawdown', 0) > 0.05:
            recommendations.append("Reduce position sizes until drawdown recovers")
        
        # Volatility recommendations
        if risk_summary.get('portfolio_volatility', 0) > 0.2:
            recommendations.append("Implement volatility targeting to maintain consistent risk levels")
        
        # Concentration recommendations
        if risk_summary.get('concentration_risk', 0) > 0.25:
            recommendations.append("Diversify holdings to reduce concentration risk")
        
        # Risk events recommendations
        if len(risk_events) > 5:
            recommendations.append("Review and tighten risk limits to reduce event frequency")
        
        return recommendations
    
    def _generate_trading_recommendations(self, trading_stats: Dict[str, Any]) -> List[str]:
        """Generate trading improvement recommendations"""
        
        recommendations = []
        
        # Win rate recommendations
        if trading_stats.get('win_rate', 0) < 50:
            recommendations.append("Improve trade selection criteria to increase win rate")
        
        # Profit factor recommendations
        if trading_stats.get('profit_factor', 0) < 1.5:
            recommendations.append("Focus on improving risk/reward ratio of trades")
        
        # Volume recommendations
        avg_size = trading_stats.get('average_trade_size', 0)
        if avg_size > 0:
            recommendations.append("Consider optimizing trade sizes based on market conditions")
        
        return recommendations
    
    def _generate_executive_summary(self, 
                                   performance_report: Dict[str, Any],
                                   risk_report: Dict[str, Any], 
                                   trading_report: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive summary"""
        
        # Extract key metrics
        perf_summary = performance_report.get('portfolio_summary', {})
        risk_summary = risk_report.get('risk_summary', {})
        trading_stats = trading_report.get('trading_statistics', {})
        
        summary = {
            'period_return': perf_summary.get('return_pct', 0),
            'total_trades': trading_stats.get('total_trades', 0),
            'win_rate': trading_stats.get('win_rate', 0),
            'max_drawdown': risk_summary.get('max_drawdown', 0),
            'sharpe_ratio': performance_report.get('performance_metrics', {}).get('sharpe_ratio', 0),
            'key_highlights': [],
            'main_concerns': [],
            'priority_actions': []
        }
        
        # Generate highlights
        if summary['period_return'] > 10:
            summary['key_highlights'].append("Strong positive returns achieved")
        if summary['win_rate'] > 60:
            summary['key_highlights'].append("High win rate demonstrates effective strategy")
        if summary['sharpe_ratio'] > 1.5:
            summary['key_highlights'].append("Excellent risk-adjusted performance")
        
        # Generate concerns
        if summary['max_drawdown'] > 0.15:
            summary['main_concerns'].append("High maximum drawdown indicates risk management issues")
        if summary['win_rate'] < 40:
            summary['main_concerns'].append("Low win rate suggests strategy refinement needed")
        
        # Generate priority actions
        if summary['max_drawdown'] > 0.1:
            summary['priority_actions'].append("Implement stricter risk controls")
        if summary['sharpe_ratio'] < 1:
            summary['priority_actions'].append("Improve risk-adjusted returns")
        
        return summary
    
    def _generate_overall_recommendations(self, 
                                        performance_report: Dict[str, Any],
                                        risk_report: Dict[str, Any], 
                                        trading_report: Dict[str, Any]) -> List[str]:
        """Generate overall strategic recommendations"""
        
        recommendations = []
        
        # Combine recommendations from all reports
        perf_recs = performance_report.get('recommendations', [])
        risk_recs = risk_report.get('risk_recommendations', [])
        trading_recs = trading_report.get('trading_recommendations', [])
        
        all_recs = perf_recs + risk_recs + trading_recs
        
        # Prioritize and deduplicate
        priority_recs = []
        for rec in all_recs:
            if 'risk' in rec.lower() or 'drawdown' in rec.lower():
                priority_recs.append(rec)
        
        # Add strategic recommendations
        recommendations.extend(priority_recs[:3])  # Top 3 priority items
        recommendations.append("Conduct regular strategy review and optimization")
        recommendations.append("Maintain detailed performance tracking and analysis")
        
        return recommendations[:5]  # Limit to 5 recommendations
    
    def export_report(self, 
                     report_data: Dict[str, Any], 
                     format: ReportFormat = ReportFormat.HTML,
                     filename: str = None) -> str:
        """
        Export report to specified format
        
        Args:
            report_data: Report data dictionary
            format: Output format
            filename: Output filename (auto-generated if None)
            
        Returns:
            Path to exported file
        """
        
        # Generate filename if not provided
        if filename is None:
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S") if self.config.include_timestamp else ""
            report_type = report_data.get('report_type', 'report')
            filename = f"{self.config.filename_prefix}_{report_type}_{timestamp}.{format.value}"
        
        filepath = Path(self.config.output_directory) / filename
        
        try:
            if format == ReportFormat.JSON:
                with open(filepath, 'w') as f:
                    json.dump(report_data, f, indent=2, default=str)
            
            elif format == ReportFormat.HTML:
                html_content = self._generate_html_report(report_data)
                with open(filepath, 'w') as f:
                    f.write(html_content)
            
            elif format == ReportFormat.CSV:
                # Export key metrics as CSV
                self._export_csv_report(report_data, filepath)
            
            elif format == ReportFormat.MARKDOWN:
                md_content = self._generate_markdown_report(report_data)
                with open(filepath, 'w') as f:
                    f.write(md_content)
            
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            self.logger.info(f"Report exported to {filepath}")
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"Failed to export report: {e}")
            raise
    
    def _generate_html_report(self, report_data: Dict[str, Any]) -> str:
        """Generate HTML report content"""
        
        title = f"HFT Simulator - {report_data.get('report_type', 'Report').replace('_', ' ').title()}"
        timestamp = report_data.get('generation_time', pd.Timestamp.now().isoformat())
        
        # Build content sections
        content_sections = []
        
        # Executive summary
        if 'executive_summary' in report_data:
            exec_summary = report_data['executive_summary']
            content_sections.append(f"""
            <div class="section">
                <h2>Executive Summary</h2>
                <div class="metric">Period Return: <span class="{'positive' if exec_summary.get('period_return', 0) > 0 else 'negative'}">{exec_summary.get('period_return', 0):.2f}%</span></div>
                <div class="metric">Total Trades: {exec_summary.get('total_trades', 0)}</div>
                <div class="metric">Win Rate: {exec_summary.get('win_rate', 0):.1f}%</div>
                <div class="metric">Max Drawdown: <span class="negative">{exec_summary.get('max_drawdown', 0):.2%}</span></div>
                <div class="metric">Sharpe Ratio: {exec_summary.get('sharpe_ratio', 0):.2f}</div>
            </div>
            """)
        
        # Performance metrics
        if 'performance_metrics' in report_data:
            metrics = report_data['performance_metrics']
            content_sections.append(f"""
            <div class="section">
                <h2>Performance Metrics</h2>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Total Return</td><td class="{'positive' if metrics.get('total_return', 0) > 0 else 'negative'}">{metrics.get('total_return', 0):.2%}</td></tr>
                    <tr><td>Sharpe Ratio</td><td>{metrics.get('sharpe_ratio', 0):.2f}</td></tr>
                    <tr><td>Maximum Drawdown</td><td class="negative">{metrics.get('max_drawdown', 0):.2%}</td></tr>
                    <tr><td>Volatility</td><td>{metrics.get('annualized_volatility', 0):.2%}</td></tr>
                    <tr><td>Win Rate</td><td>{metrics.get('win_rate', 0):.1%}</td></tr>
                </table>
            </div>
            """)
        
        # Key insights
        insights = []
        if 'key_insights' in report_data:
            insights.extend(report_data['key_insights'])
        if 'risk_insights' in report_data:
            insights.extend(report_data['risk_insights'])
        if 'trading_insights' in report_data:
            insights.extend(report_data['trading_insights'])
        
        if insights:
            insight_items = ''.join([f"<li>{insight}</li>" for insight in insights])
            content_sections.append(f"""
            <div class="section">
                <h2>Key Insights</h2>
                <ul>{insight_items}</ul>
            </div>
            """)
        
        # Recommendations
        recommendations = []
        if 'recommendations' in report_data:
            recommendations.extend(report_data['recommendations'])
        if 'risk_recommendations' in report_data:
            recommendations.extend(report_data['risk_recommendations'])
        if 'trading_recommendations' in report_data:
            recommendations.extend(report_data['trading_recommendations'])
        if 'overall_recommendations' in report_data:
            recommendations.extend(report_data['overall_recommendations'])
        
        if recommendations:
            rec_items = ''.join([f"<li>{rec}</li>" for rec in recommendations[:10]])  # Limit to 10
            content_sections.append(f"""
            <div class="section">
                <h2>Recommendations</h2>
                <ul>{rec_items}</ul>
            </div>
            """)
        
        # Combine all content
        content = ''.join(content_sections)
        
        # Use template
        html_report = self.templates['html'].format(
            title=title,
            timestamp=timestamp,
            content=content
        )
        
        return html_report
    
    def _generate_markdown_report(self, report_data: Dict[str, Any]) -> str:
        """Generate Markdown report content"""
        
        title = f"# HFT Simulator - {report_data.get('report_type', 'Report').replace('_', ' ').title()}\n\n"
        timestamp = f"*Generated on {report_data.get('generation_time', pd.Timestamp.now().isoformat())}*\n\n"
        
        content = [title, timestamp]
        
        # Executive summary
        if 'executive_summary' in report_data:
            exec_summary = report_data['executive_summary']
            content.append("## Executive Summary\n\n")
            content.append(f"- **Period Return:** {exec_summary.get('period_return', 0):.2f}%\n")
            content.append(f"- **Total Trades:** {exec_summary.get('total_trades', 0)}\n")
            content.append(f"- **Win Rate:** {exec_summary.get('win_rate', 0):.1f}%\n")
            content.append(f"- **Max Drawdown:** {exec_summary.get('max_drawdown', 0):.2%}\n")
            content.append(f"- **Sharpe Ratio:** {exec_summary.get('sharpe_ratio', 0):.2f}\n\n")
        
        # Key insights
        insights = []
        if 'key_insights' in report_data:
            insights.extend(report_data['key_insights'])
        if 'risk_insights' in report_data:
            insights.extend(report_data['risk_insights'])
        if 'trading_insights' in report_data:
            insights.extend(report_data['trading_insights'])
        
        if insights:
            content.append("## Key Insights\n\n")
            for insight in insights:
                content.append(f"- {insight}\n")
            content.append("\n")
        
        # Recommendations
        recommendations = []
        if 'recommendations' in report_data:
            recommendations.extend(report_data['recommendations'])
        if 'overall_recommendations' in report_data:
            recommendations.extend(report_data['overall_recommendations'])
        
        if recommendations:
            content.append("## Recommendations\n\n")
            for i, rec in enumerate(recommendations[:10], 1):
                content.append(f"{i}. {rec}\n")
            content.append("\n")
        
        return ''.join(content)
    
    def _export_csv_report(self, report_data: Dict[str, Any], filepath: Path) -> None:
        """Export key metrics as CSV"""
        
        # Extract key metrics for CSV export
        metrics_data = []
        
        if 'performance_metrics' in report_data:
            perf_metrics = report_data['performance_metrics']
            for key, value in perf_metrics.items():
                metrics_data.append({'Category': 'Performance', 'Metric': key, 'Value': value})
        
        if 'risk_summary' in report_data:
            risk_metrics = report_data['risk_summary']
            for key, value in risk_metrics.items():
                if isinstance(value, (int, float)):
                    metrics_data.append({'Category': 'Risk', 'Metric': key, 'Value': value})
        
        if 'trading_statistics' in report_data:
            trading_metrics = report_data['trading_statistics']
            for key, value in trading_metrics.items():
                metrics_data.append({'Category': 'Trading', 'Metric': key, 'Value': value})
        
        # Create DataFrame and export
        df = pd.DataFrame(metrics_data)
        df.to_csv(filepath, index=False)


# Utility functions
def create_report_generator(config: ReportConfig = None) -> ReportGenerator:
    """Create a report generator with specified configuration"""
    return ReportGenerator(config=config)


def generate_quick_report(portfolio: Portfolio, 
                         report_type: ReportType = ReportType.PERFORMANCE_SUMMARY) -> Dict[str, Any]:
    """
    Generate a quick report for immediate analysis
    
    Args:
        portfolio: Portfolio to analyze
        report_type: Type of report to generate
        
    Returns:
        Report data dictionary
    """
    generator = ReportGenerator()
    
    if report_type == ReportType.PERFORMANCE_SUMMARY:
        return generator.generate_performance_report(portfolio)
    elif report_type == ReportType.TRADING_ACTIVITY:
        return generator.generate_trading_report(portfolio)
    else:
        raise ValueError(f"Quick report not supported for type: {report_type}")