"""
Real-Time Risk Management System for HFT Simulator

This module provides comprehensive real-time risk management capabilities
including pre-trade checks, position monitoring, loss limits, and emergency
controls for live trading operations.

Key Features:
- Pre-trade risk validation
- Real-time position and P&L monitoring
- Dynamic risk limit enforcement
- Circuit breakers and emergency stops
- Volatility-based risk adjustments
- Compliance monitoring
- Risk reporting and alerts

Risk Controls:
- Position limits (per symbol, total exposure)
- Loss limits (daily, weekly, maximum drawdown)
- Concentration limits (sector, correlation)
- Volatility filters (market conditions)
- Liquidity requirements
- Order size limits
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Set, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging

from src.utils.logger import get_logger
from src.utils.constants import OrderSide, OrderType
from src.realtime.types import OrderRequest, Position, RiskLevel, ViolationType, RiskViolation, RiskValidationResult


@dataclass
class RiskLimit:
    """Definition of a risk limit"""
    
    limit_type: ViolationType
    limit_value: float
    warning_threshold: float  # Percentage of limit that triggers warning
    
    # Scope
    symbol: Optional[str] = None  # None means global limit
    strategy_id: Optional[str] = None
    
    # Timing
    time_window: Optional[timedelta] = None  # For time-based limits
    
    # Status
    enabled: bool = True
    last_checked: Optional[datetime] = None
    
    # Metadata
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MarketConditions:
    """Current market conditions for risk assessment"""
    
    # Volatility measures
    implied_volatility: Dict[str, float] = field(default_factory=dict)
    realized_volatility: Dict[str, float] = field(default_factory=dict)
    
    # Liquidity measures
    bid_ask_spreads: Dict[str, float] = field(default_factory=dict)
    market_depth: Dict[str, float] = field(default_factory=dict)
    
    # Market state
    market_hours: bool = True
    market_stress_level: float = 0.0  # 0-1 scale
    
    # Correlations
    correlation_matrix: Optional[pd.DataFrame] = None
    
    # Timestamps
    last_updated: datetime = field(default_factory=datetime.now)


class RiskMonitor:
    """
    Monitors risk metrics and triggers alerts
    """
    
    def __init__(self):
        self.logger = get_logger(f"{self.__class__.__name__}")
        
        # Alert callbacks
        self.alert_callbacks: List[Callable[[RiskViolation], None]] = []
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_task: Optional[asyncio.Task] = None
        
        # Historical data for trend analysis
        self.risk_history: List[Dict[str, Any]] = []
        self.violation_history: List[RiskViolation] = []
        
    def add_alert_callback(self, callback: Callable[[RiskViolation], None]) -> None:
        """Add callback for risk alerts"""
        self.alert_callbacks.append(callback)
        self.logger.info("Added risk alert callback")
    
    async def start_monitoring(self, risk_manager: 'RealTimeRiskManager') -> None:
        """Start continuous risk monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop(risk_manager))
        self.logger.info("Risk monitoring started")
    
    async def stop_monitoring(self) -> None:
        """Stop risk monitoring"""
        self.monitoring_active = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Risk monitoring stopped")
    
    async def _monitoring_loop(self, risk_manager: 'RealTimeRiskManager') -> None:
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Check all risk metrics
                violations = await risk_manager.check_all_limits()
                
                # Process violations
                for violation in violations:
                    await self._handle_violation(violation)
                
                # Update risk history
                risk_metrics = risk_manager.get_risk_metrics()
                self.risk_history.append({
                    'timestamp': datetime.now(),
                    'metrics': risk_metrics
                })
                
                # Keep history bounded
                if len(self.risk_history) > 1000:
                    self.risk_history.pop(0)
                
                await asyncio.sleep(1)  # Check every second
                
            except Exception as e:
                self.logger.error(f"Error in risk monitoring: {e}")
                await asyncio.sleep(5)
    
    async def _handle_violation(self, violation: RiskViolation) -> None:
        """Handle risk violation"""
        self.violation_history.append(violation)
        
        # Notify callbacks
        for callback in self.alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(violation)
                else:
                    callback(violation)
            except Exception as e:
                self.logger.error(f"Error in alert callback: {e}")
        
        # Log violation
        self.logger.warning(
            f"Risk violation: {violation.violation_type.value} - "
            f"{violation.current_value} exceeds {violation.limit_value}"
        )


class RealTimeRiskManager:
    """
    Main real-time risk management system
    
    Provides comprehensive risk management including pre-trade validation,
    real-time monitoring, and automatic risk controls.
    """
    
    def __init__(self):
        self.logger = get_logger(f"{self.__class__.__name__}")
        
        # Risk limits
        self.limits: Dict[str, RiskLimit] = {}
        self.global_limits: Dict[ViolationType, RiskLimit] = {}
        
        # Current state
        self.positions: Dict[str, Position] = {}
        self.market_conditions = MarketConditions()
        
        # Risk metrics
        self.current_exposure = 0.0
        self.daily_pnl = 0.0
        self.max_drawdown = 0.0
        self.risk_score = 0.0
        
        # Emergency controls
        self.emergency_stop_active = False
        self.trading_halted = False
        self.halt_reasons: Set[str] = set()
        
        # Monitoring
        self.risk_monitor = RiskMonitor()
        
        # Statistics
        self.stats = {
            'orders_blocked': 0,
            'violations_detected': 0,
            'emergency_stops': 0,
            'risk_checks_performed': 0,
            'last_check': None
        }
        
        # Initialize default limits
        self._setup_default_limits()
    
    def _setup_default_limits(self) -> None:
        """Setup default risk limits"""
        
        # Position limits
        self.add_global_limit(
            ViolationType.POSITION_LIMIT,
            limit_value=1000000.0,  # $1M max position
            warning_threshold=0.8,
            description="Maximum position size per symbol"
        )
        
        # Loss limits
        self.add_global_limit(
            ViolationType.LOSS_LIMIT,
            limit_value=50000.0,  # $50K daily loss limit
            warning_threshold=0.8,
            description="Daily loss limit"
        )
        
        # Exposure limits
        self.add_global_limit(
            ViolationType.EXPOSURE_LIMIT,
            limit_value=5000000.0,  # $5M total exposure
            warning_threshold=0.8,
            description="Total portfolio exposure limit"
        )
        
        # Order size limits
        self.add_global_limit(
            ViolationType.ORDER_SIZE_LIMIT,
            limit_value=100000.0,  # $100K max order
            warning_threshold=0.9,
            description="Maximum single order size"
        )
        
        # Drawdown limits
        self.add_global_limit(
            ViolationType.DRAWDOWN_LIMIT,
            limit_value=0.1,  # 10% max drawdown
            warning_threshold=0.8,
            description="Maximum drawdown from peak"
        )
    
    def add_global_limit(self, 
                        violation_type: ViolationType,
                        limit_value: float,
                        warning_threshold: float,
                        description: str = "") -> None:
        """Add global risk limit"""
        
        limit = RiskLimit(
            limit_type=violation_type,
            limit_value=limit_value,
            warning_threshold=warning_threshold,
            description=description
        )
        
        self.global_limits[violation_type] = limit
        self.logger.info(f"Added global limit: {violation_type.value} = {limit_value}")
    
    def add_symbol_limit(self,
                        symbol: str,
                        violation_type: ViolationType,
                        limit_value: float,
                        warning_threshold: float,
                        description: str = "") -> None:
        """Add symbol-specific risk limit"""
        
        limit = RiskLimit(
            limit_type=violation_type,
            limit_value=limit_value,
            warning_threshold=warning_threshold,
            symbol=symbol,
            description=description
        )
        
        limit_key = f"{symbol}_{violation_type.value}"
        self.limits[limit_key] = limit
        self.logger.info(f"Added symbol limit: {symbol} {violation_type.value} = {limit_value}")
    
    async def validate_order(self, 
                           order_request: OrderRequest,
                           current_positions: Dict[str, Position]) -> RiskValidationResult:
        """
        Validate order against all risk limits
        
        Args:
            order_request: Order to validate
            current_positions: Current portfolio positions
            
        Returns:
            Validation result with approval status and any violations
        """
        self.stats['risk_checks_performed'] += 1
        self.stats['last_check'] = datetime.now()
        
        violations = []
        warnings = []
        
        # Update positions
        self.positions = current_positions
        
        # Check if trading is halted
        if self.trading_halted:
            violations.append(RiskViolation(
                violation_type=ViolationType.POSITION_LIMIT,
                risk_level=RiskLevel.CRITICAL,
                current_value=0,
                limit_value=0,
                excess_amount=0,
                message="Trading is currently halted",
                action_taken="Order blocked"
            ))
        
        # Check emergency stop
        if self.emergency_stop_active:
            violations.append(RiskViolation(
                violation_type=ViolationType.POSITION_LIMIT,
                risk_level=RiskLevel.CRITICAL,
                current_value=0,
                limit_value=0,
                excess_amount=0,
                message="Emergency stop is active",
                action_taken="Order blocked"
            ))
        
        # Order size check
        order_value = (order_request.price or 0) * order_request.quantity
        size_violation = self._check_order_size_limit(order_value)
        if size_violation:
            violations.append(size_violation)
        
        # Position limit check
        position_violation = await self._check_position_limits(order_request)
        if position_violation:
            violations.append(position_violation)
        
        # Exposure limit check
        exposure_violation = await self._check_exposure_limits(order_request)
        if exposure_violation:
            violations.append(exposure_violation)
        
        # Loss limit check
        loss_violation = await self._check_loss_limits()
        if loss_violation:
            violations.append(loss_violation)
        
        # Volatility check
        volatility_violation = await self._check_volatility_limits(order_request)
        if volatility_violation:
            violations.append(volatility_violation)
        
        # Liquidity check
        liquidity_violation = await self._check_liquidity_requirements(order_request)
        if liquidity_violation:
            violations.append(liquidity_violation)
        
        # Calculate risk score
        risk_score = self._calculate_risk_score(order_request, violations, warnings)
        
        # Determine approval
        approved = len(violations) == 0
        
        if not approved:
            self.stats['orders_blocked'] += 1
            
        # Emit violations to the risk monitor for callback processing
        for violation in violations:
            # Add to violation history and trigger callbacks
            await self.risk_monitor._handle_violation(violation)
        
        return RiskValidationResult(
            approved=approved,
            violations=violations,
            warnings=warnings,
            risk_score=risk_score,
            recommendation=self._generate_recommendation(violations, warnings)
        )
    
    def _check_order_size_limit(self, order_value: float) -> Optional[RiskViolation]:
        """Check order size against limits"""
        limit = self.global_limits.get(ViolationType.ORDER_SIZE_LIMIT)
        if not limit or not limit.enabled:
            return None
        
        if order_value > limit.limit_value:
            return RiskViolation(
                violation_type=ViolationType.ORDER_SIZE_LIMIT,
                risk_level=RiskLevel.HIGH,
                current_value=order_value,
                limit_value=limit.limit_value,
                excess_amount=order_value - limit.limit_value,
                message=f"Order size ${order_value:,.2f} exceeds limit ${limit.limit_value:,.2f}",
                action_taken="Order blocked"
            )
        
        return None
    
    async def _check_position_limits(self, order_request: OrderRequest) -> Optional[RiskViolation]:
        """Check position limits"""
        symbol = order_request.symbol
        
        # Get current position
        current_position = self.positions.get(symbol)
        current_quantity = current_position.quantity if current_position else 0.0
        
        # Calculate new position after order
        if order_request.side == OrderSide.BUY:
            new_quantity = current_quantity + order_request.quantity
        else:
            new_quantity = current_quantity - order_request.quantity
        
        # Check symbol-specific limit
        symbol_limit_key = f"{symbol}_{ViolationType.POSITION_LIMIT.value}"
        if symbol_limit_key in self.limits:
            limit = self.limits[symbol_limit_key]
            if abs(new_quantity) > limit.limit_value:
                return RiskViolation(
                    violation_type=ViolationType.POSITION_LIMIT,
                    risk_level=RiskLevel.HIGH,
                    current_value=abs(new_quantity),
                    limit_value=limit.limit_value,
                    excess_amount=abs(new_quantity) - limit.limit_value,
                    symbol=symbol,
                    message=f"Position limit exceeded for {symbol}",
                    action_taken="Order blocked"
                )
        
        # Check global position limit
        global_limit = self.global_limits.get(ViolationType.POSITION_LIMIT)
        if global_limit and global_limit.enabled:
            position_value = abs(new_quantity) * (order_request.price or 0)
            if position_value > global_limit.limit_value:
                return RiskViolation(
                    violation_type=ViolationType.POSITION_LIMIT,
                    risk_level=RiskLevel.HIGH,
                    current_value=position_value,
                    limit_value=global_limit.limit_value,
                    excess_amount=position_value - global_limit.limit_value,
                    symbol=symbol,
                    message=f"Global position limit exceeded",
                    action_taken="Order blocked"
                )
        
        return None
    
    async def _check_exposure_limits(self, order_request: OrderRequest) -> Optional[RiskViolation]:
        """Check total exposure limits"""
        limit = self.global_limits.get(ViolationType.EXPOSURE_LIMIT)
        if not limit or not limit.enabled:
            return None
        
        # Calculate current exposure
        current_exposure = sum(
            abs(pos.quantity * pos.average_price) 
            for pos in self.positions.values()
        )
        
        # Add order exposure
        order_exposure = order_request.quantity * (order_request.price or 0)
        new_exposure = current_exposure + order_exposure
        
        if new_exposure > limit.limit_value:
            return RiskViolation(
                violation_type=ViolationType.EXPOSURE_LIMIT,
                risk_level=RiskLevel.HIGH,
                current_value=new_exposure,
                limit_value=limit.limit_value,
                excess_amount=new_exposure - limit.limit_value,
                message=f"Total exposure limit exceeded",
                action_taken="Order blocked"
            )
        
        return None
    
    async def _check_loss_limits(self) -> Optional[RiskViolation]:
        """Check loss limits"""
        limit = self.global_limits.get(ViolationType.LOSS_LIMIT)
        if not limit or not limit.enabled:
            return None
        
        # Calculate current daily P&L (simplified)
        current_pnl = sum(pos.unrealized_pnl + pos.realized_pnl for pos in self.positions.values())
        
        if current_pnl < -limit.limit_value:
            return RiskViolation(
                violation_type=ViolationType.LOSS_LIMIT,
                risk_level=RiskLevel.CRITICAL,
                current_value=abs(current_pnl),
                limit_value=limit.limit_value,
                excess_amount=abs(current_pnl) - limit.limit_value,
                message=f"Daily loss limit exceeded: ${current_pnl:,.2f}",
                action_taken="Trading halted"
            )
        
        return None
    
    async def _check_volatility_limits(self, order_request: OrderRequest) -> Optional[RiskViolation]:
        """Check volatility-based limits"""
        symbol = order_request.symbol
        
        # Get volatility from market conditions
        volatility = self.market_conditions.realized_volatility.get(symbol, 0.0)
        
        # Simple volatility check (would be more sophisticated in practice)
        if volatility > 0.5:  # 50% volatility threshold
            return RiskViolation(
                violation_type=ViolationType.VOLATILITY_LIMIT,
                risk_level=RiskLevel.MEDIUM,
                current_value=volatility,
                limit_value=0.5,
                excess_amount=volatility - 0.5,
                symbol=symbol,
                message=f"High volatility detected: {volatility:.1%}",
                action_taken="Order flagged"
            )
        
        return None
    
    async def _check_liquidity_requirements(self, order_request: OrderRequest) -> Optional[RiskViolation]:
        """Check liquidity requirements"""
        symbol = order_request.symbol
        
        # Get spread from market conditions
        spread = self.market_conditions.bid_ask_spreads.get(symbol, 0.0)
        
        # Simple liquidity check
        if spread > 0.01:  # 1% spread threshold
            return RiskViolation(
                violation_type=ViolationType.LIQUIDITY_LIMIT,
                risk_level=RiskLevel.MEDIUM,
                current_value=spread,
                limit_value=0.01,
                excess_amount=spread - 0.01,
                symbol=symbol,
                message=f"Wide spread detected: {spread:.1%}",
                action_taken="Order flagged"
            )
        
        return None
    
    def _calculate_risk_score(self, 
                            order_request: OrderRequest,
                            violations: List[RiskViolation],
                            warnings: List[RiskViolation]) -> float:
        """Calculate overall risk score (0-100)"""
        base_score = 0.0
        
        # Add points for violations
        for violation in violations:
            if violation.risk_level == RiskLevel.CRITICAL:
                base_score += 40
            elif violation.risk_level == RiskLevel.HIGH:
                base_score += 25
            elif violation.risk_level == RiskLevel.MEDIUM:
                base_score += 15
            else:
                base_score += 5
        
        # Add points for warnings
        for warning in warnings:
            base_score += 5
        
        # Cap at 100
        return min(base_score, 100.0)
    
    def _generate_recommendation(self, 
                               violations: List[RiskViolation],
                               warnings: List[RiskViolation]) -> str:
        """Generate risk recommendation"""
        if not violations and not warnings:
            return "Order approved - low risk"
        
        if violations:
            critical_count = sum(1 for v in violations if v.risk_level == RiskLevel.CRITICAL)
            if critical_count > 0:
                return "Order rejected - critical risk violations detected"
            else:
                return "Order rejected - risk limit violations detected"
        
        if warnings:
            return "Order approved with warnings - monitor closely"
        
        return "Order approved"
    
    async def check_all_limits(self) -> List[RiskViolation]:
        """Check all current positions against limits"""
        violations = []
        
        # Check each position
        for symbol, position in self.positions.items():
            # Position size check
            position_value = abs(position.quantity * position.average_price)
            
            # Check symbol-specific limits
            symbol_limit_key = f"{symbol}_{ViolationType.POSITION_LIMIT.value}"
            if symbol_limit_key in self.limits:
                limit = self.limits[symbol_limit_key]
                if position_value > limit.limit_value:
                    violations.append(RiskViolation(
                        violation_type=ViolationType.POSITION_LIMIT,
                        risk_level=RiskLevel.HIGH,
                        current_value=position_value,
                        limit_value=limit.limit_value,
                        excess_amount=position_value - limit.limit_value,
                        symbol=symbol,
                        message=f"Position limit exceeded for {symbol}"
                    ))
        
        # Check global limits
        total_exposure = sum(
            abs(pos.quantity * pos.average_price) 
            for pos in self.positions.values()
        )
        
        exposure_limit = self.global_limits.get(ViolationType.EXPOSURE_LIMIT)
        if exposure_limit and total_exposure > exposure_limit.limit_value:
            violations.append(RiskViolation(
                violation_type=ViolationType.EXPOSURE_LIMIT,
                risk_level=RiskLevel.HIGH,
                current_value=total_exposure,
                limit_value=exposure_limit.limit_value,
                excess_amount=total_exposure - exposure_limit.limit_value,
                message="Total exposure limit exceeded"
            ))
        
        if violations:
            self.stats['violations_detected'] += len(violations)
        
        return violations
    
    def activate_emergency_stop(self, reason: str) -> None:
        """Activate emergency stop"""
        self.emergency_stop_active = True
        self.halt_reasons.add(reason)
        self.stats['emergency_stops'] += 1
        
        self.logger.critical(f"EMERGENCY STOP ACTIVATED: {reason}")
    
    def deactivate_emergency_stop(self, reason: str) -> None:
        """Deactivate emergency stop"""
        self.halt_reasons.discard(reason)
        
        if not self.halt_reasons:
            self.emergency_stop_active = False
            self.logger.info("Emergency stop deactivated")
    
    def halt_trading(self, reason: str) -> None:
        """Halt all trading"""
        self.trading_halted = True
        self.halt_reasons.add(reason)
        
        self.logger.critical(f"TRADING HALTED: {reason}")
    
    def resume_trading(self, reason: str) -> None:
        """Resume trading"""
        self.halt_reasons.discard(reason)
        
        if not self.halt_reasons:
            self.trading_halted = False
            self.logger.info("Trading resumed")
    
    def update_market_conditions(self, conditions: MarketConditions) -> None:
        """Update market conditions for risk assessment"""
        self.market_conditions = conditions
        self.logger.debug("Market conditions updated")
    
    def get_risk_metrics(self) -> Dict[str, Any]:
        """Get current risk metrics"""
        total_exposure = sum(
            abs(pos.quantity * pos.average_price) 
            for pos in self.positions.values()
        )
        
        total_pnl = sum(
            pos.unrealized_pnl + pos.realized_pnl 
            for pos in self.positions.values()
        )
        
        return {
            'total_exposure': total_exposure,
            'total_pnl': total_pnl,
            'position_count': len([p for p in self.positions.values() if p.quantity != 0]),
            'emergency_stop_active': self.emergency_stop_active,
            'trading_halted': self.trading_halted,
            'risk_score': self.risk_score,
            'halt_reasons': list(self.halt_reasons),
            'limits_count': len(self.limits) + len(self.global_limits),
            'stats': self.stats
        }
    
    async def start_monitoring(self) -> None:
        """Start risk monitoring"""
        await self.risk_monitor.start_monitoring(self)
    
    async def stop_monitoring(self) -> None:
        """Stop risk monitoring"""
        await self.risk_monitor.stop_monitoring()
    
    def generate_risk_report(self) -> Dict[str, Any]:
        """Generate comprehensive risk report for real-time system"""
        
        # Get current risk metrics
        current_metrics = self.get_risk_metrics()
        
        # Get violation history from monitor
        recent_violations = self.risk_monitor.violation_history[-50:] if hasattr(self.risk_monitor, 'violation_history') else []
        
        report = {
            'report_type': 'real_time_risk_analysis',
            'generation_time': datetime.now().isoformat(),
            'system_status': {
                'emergency_stop_active': self.emergency_stop_active,
                'trading_halted': self.trading_halted,
                'halt_reasons': list(self.halt_reasons),
            },
            'current_metrics': current_metrics,
            'global_limits': [
                {
                    'violation_type': limit.limit_type.value,
                    'limit_value': limit.limit_value,
                    'warning_threshold': limit.warning_threshold,
                    'enabled': limit.enabled,
                    'description': limit.description,
                }
                for limit in self.global_limits.values()
            ],
            'symbol_limits': [
                {
                    'key': key,
                    'violation_type': limit.limit_type.value,
                    'symbol': limit.symbol,
                    'limit_value': limit.limit_value,
                    'warning_threshold': limit.warning_threshold,
                    'enabled': limit.enabled,
                    'description': limit.description,
                }
                for key, limit in self.limits.items()
            ],
            'recent_violations': [
                {
                    'violation_type': violation.violation_type.value,
                    'risk_level': violation.risk_level.value,
                    'current_value': violation.current_value,
                    'limit_value': violation.limit_value,
                    'excess_amount': violation.excess_amount,
                    'symbol': violation.symbol,
                    'message': violation.message,
                    'action_taken': violation.action_taken,
                }
                for violation in recent_violations
            ],
            'statistics': self.stats,
            'positions_summary': {
                'total_positions': len(self.positions),
                'active_positions': len([p for p in self.positions.values() if p.quantity != 0]),
                'total_exposure': sum(
                    abs(pos.quantity * pos.average_price) 
                    for pos in self.positions.values()
                ),
            }
        }
        
        return report
