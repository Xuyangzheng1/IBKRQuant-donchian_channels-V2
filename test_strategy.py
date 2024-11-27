# test_strategy.py
import unittest
import time
import pandas as pd
import numpy as np
from mock_tws import MockTWS
from improved_donchian_strategy import ImprovedDonchianStrategy
import logging

class TestDonchianStrategy(unittest.TestCase):
    def setUp(self):
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        
        # 初始化模拟TWS
        self.mock_tws = MockTWS()
        
        # 初始化策略
        self.strategy = ImprovedDonchianStrategy(
            symbol='TSLA',
            capital=100000,
            period=20,
            base_tranches=3,
            alert_threshold=0.004,
            max_capital_per_trade=50000
        )
        
        # 替换策略中的TWS对象
        self.strategy.bot = self.mock_tws
        
        # 连接模拟TWS
        self.mock_tws.connect("127.0.0.1", 7497, 1)
        #==============================
    # def test_backtest(self):
    #     """测试回测功能"""
    #     result = self.strategy.backtest(
    #         start_date='2024-11-22',
    #         end_date='2024-11-26',
    #         symbol='TSLA'
    #     )
        
    #     self.assertIsNotNone(result)
    #     self.assertGreaterEqual(result['total_trades'], 0)
        #==============================================
    def test_position_calculation(self):
        """测试持仓量计算"""
        # 测试正常价格
        position_size = self.strategy.calculate_position_size(100.0)
        self.assertGreater(position_size, 0)
        self.assertLessEqual(position_size * 100.0, self.strategy.max_capital_per_trade)
        
        # 测试最小交易量
        position_size = self.strategy.calculate_position_size(5000.0)
        self.assertGreaterEqual(position_size, 10)

        # 测试极端情况
        position_size = self.strategy.calculate_position_size(1.0)
        self.assertLessEqual(position_size * 1.0, self.strategy.max_capital_per_trade)

        # 测试高价情况
        position_size = self.strategy.calculate_position_size(10000.0)
        self.assertGreaterEqual(position_size, 10)
    def generate_test_data(self):
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1min')
        data = {
            'Open': np.random.uniform(100, 110, 100),
            'High': np.random.uniform(105, 115, 100),
            'Low': np.random.uniform(95, 105, 100),
            'Close': np.random.uniform(100, 110, 100),
            'Volume': np.random.uniform(1000, 2000, 100)
        }
        return pd.DataFrame(data, index=dates)

    def test_trade_execution(self):
        """测试交易执行"""
        # 确保初始状态正确
        self.assertEqual(self.mock_tws.positions['TSLA'], 0)
        
        # 设置最后交易时间为None以允许交易
        self.strategy.last_trade_time = None
        
        # 执行买入交易
        result = self.strategy.execute_trade(
            "BUY",
            price=100.0,
            upper_channel=110.0,
            lower_channel=90.0
        )
        
        # 验证交易结果
        self.assertTrue(result)
        self.assertTrue(len(self.mock_tws.orders) > 0)
        self.assertEqual(self.mock_tws.orders[-1]['action'], "BUY")
        self.assertTrue(self.mock_tws.positions['TSLA'] > 0)

    def test_position_management(self):
        """测试仓位管理"""
        # 设置测试模式
        self.strategy._test_mode = True
        
        # 确保初始状态正确
        self.assertEqual(self.mock_tws.positions['TSLA'], 0)
        self.strategy.current_position_size = 0
        
        # 执行买入交易
        buy_result = self.strategy.execute_trade(
            "BUY",
            price=100.0,
            upper_channel=110.0,
            lower_channel=90.0
        )
        
        # 验证买入结果
        self.assertTrue(buy_result)
        self.assertEqual(self.mock_tws.positions['TSLA'], 
                        self.strategy.current_position_size)
        initial_position = self.mock_tws.positions['TSLA']
        
        # 执行卖出交易
        sell_result = self.strategy.execute_trade(
            "SELL",
            price=110.0,
            upper_channel=120.0,
            lower_channel=100.0
        )
        
        # 验证卖出结果
        self.assertTrue(sell_result)
        self.assertLess(self.mock_tws.positions['TSLA'], initial_position)

    def test_channel_calculation(self):
        """测试通道计算"""
        df = self.generate_test_data()
        df_with_channels = self.strategy.get_donchian_channels(df)
        
        # 验证通道数据存在
        self.assertTrue('upper_channel' in df_with_channels.columns)
        self.assertTrue('lower_channel' in df_with_channels.columns)
        self.assertTrue('middle_channel' in df_with_channels.columns)

    def test_threshold_adjustment(self):
        """测试阈值调整"""
        df = self.generate_test_data()
        df_with_channels = self.strategy.get_donchian_channels(df)
        
        current_price = df_with_channels['Close'].iloc[-1]
        upper = df_with_channels['upper_channel'].iloc[-1]
        lower = df_with_channels['lower_channel'].iloc[-1]
        
        # 测试动态阈值调整
        channel_position = self.strategy.is_near_channel(
            current_price,
            upper,
            lower,
            df_with_channels
        )
        self.assertIn(channel_position, [None, "UPPER", "LOWER"])

def main():
    unittest.main()

if __name__ == '__main__':
    main()