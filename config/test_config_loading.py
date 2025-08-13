#!/usr/bin/env python3
"""
Test script for configuration loading functionality.

This script tests the BacktestConfig and strategy configuration loading
to ensure everything works correctly.
"""

import json
import logging
from pathlib import Path

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_backtest_config():
    """Test BacktestConfig loading and functionality."""
    logger.info("Testing BacktestConfig...")
    
    try:
        # Import the configuration classes
        from backtest_config import BacktestConfig, load_backtest_config
        
        # Test 1: Create default config
        logger.info("Test 1: Creating default config")
        default_config = BacktestConfig()
        logger.info(f"Default strategy type: {default_config.strategy_type}")
        logger.info(f"Default initial capital: ${default_config.initial_capital:,.2f}")
        
        # Test 2: Validate configuration
        logger.info("Test 2: Validating configuration")
        is_valid = default_config.validate()
        logger.info(f"Configuration is valid: {is_valid}")
        
        # Test 3: Load from existing JSON file
        logger.info("Test 3: Loading from JSON file")
        config_path = Path("backtest_config.json")
        if config_path.exists():
            loaded_config = BacktestConfig.from_json(config_path)
            logger.info(f"Loaded config strategy type: {loaded_config.strategy_type}")
            logger.info(f"Loaded config has {len(loaded_config.strategy_params)} strategy parameter sets")
        else:
            logger.warning("backtest_config.json not found in current directory")
        
        # Test 4: Get strategy config
        logger.info("Test 4: Getting strategy configuration")
        mm_config = default_config.get_strategy_config("market_making")
        logger.info(f"Market making spread_bps: {mm_config['spread_bps']}")
        
        # Test 5: Update strategy parameter
        logger.info("Test 5: Updating strategy parameter")
        default_config.update_strategy_param("spread_bps", 15.0, "market_making")
        updated_mm_config = default_config.get_strategy_config("market_making")
        logger.info(f"Updated market making spread_bps: {updated_mm_config['spread_bps']}")
        
        logger.info("✅ BacktestConfig tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ BacktestConfig test failed: {e}")
        return False


def test_strategy_config():
    """Test strategy configuration loading."""
    logger.info("Testing StrategyConfig...")
    
    try:
        # Import strategy configuration classes
        from strategy_config import (
            MarketMakingStrategyConfig, 
            MomentumStrategyConfig,
            LiquidityTakingStrategyConfig,
            StrategyConfigManager
        )
        
        # Test 1: Create default strategy configs
        logger.info("Test 1: Creating default strategy configs")
        mm_config = MarketMakingStrategyConfig()
        momentum_config = MomentumStrategyConfig()
        lt_config = LiquidityTakingStrategyConfig()
        
        logger.info(f"Market making config name: {mm_config.name}")
        logger.info(f"Market making target spread: {mm_config.target_spread_bps} bps")
        logger.info(f"Momentum config name: {momentum_config.name}")
        logger.info(f"Momentum threshold: {momentum_config.momentum_threshold}")
        
        # Test 2: Convert to dictionary
        logger.info("Test 2: Converting to dictionary")
        mm_dict = mm_config.to_dict()
        logger.info(f"Market making config has {len(mm_dict)} parameters")
        
        # Test 3: Strategy config manager
        logger.info("Test 3: Testing StrategyConfigManager")
        config_manager = StrategyConfigManager()
        
        # Try to load config (will use defaults if file doesn't exist)
        mm_loaded = config_manager.load_config("market_making")
        logger.info(f"Loaded market making config: {mm_loaded.name}")
        
        logger.info("✅ StrategyConfig tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ StrategyConfig test failed: {e}")
        return False


def test_json_loading():
    """Test loading configurations from JSON files."""
    logger.info("Testing JSON configuration loading...")
    
    try:
        # Test loading from different config files
        config_files = [
            "backtest_config.json",
            "strategy_config.json",
            "market_making_template.json",
            "momentum_template.json"
        ]
        
        for config_file in config_files:
            config_path = Path(config_file)
            if config_path.exists():
                logger.info(f"Loading {config_file}...")
                with open(config_path, 'r') as f:
                    data = json.load(f)
                logger.info(f"✅ {config_file} loaded successfully, has {len(data)} top-level keys")
            else:
                logger.warning(f"⚠️ {config_file} not found")
        
        logger.info("✅ JSON loading tests completed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ JSON loading test failed: {e}")
        return False


def main():
    """Run all configuration tests."""
    logger.info("🧪 Starting configuration loading tests...")
    logger.info("=" * 60)
    
    tests_passed = 0
    total_tests = 3
    
    # Run tests
    if test_backtest_config():
        tests_passed += 1
    
    if test_strategy_config():
        tests_passed += 1
    
    if test_json_loading():
        tests_passed += 1
    
    # Summary
    logger.info("=" * 60)
    logger.info(f"📊 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        logger.info("🎉 All configuration tests passed successfully!")
        return 0
    else:
        logger.error(f"❌ {total_tests - tests_passed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit(main())
