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
           symbol='aapl',           # 交易标的
           capital=100000,          # 初始资金
           period=20,               # 通道周期
           base_tranches=3,         # 分批数
           alert_threshold=0.004,   # 触发阈值
           max_capital_per_trade=50000  # 最大交易额
       )
        
        # 替换策略中的TWS对象
        self.strategy.bot = self.mock_tws
        
        # 连接模拟TWS
        self.mock_tws.connect("127.0.0.1", 7497, 1)
        #==============================
    def test_backtest(self):
       """测试回测功能"""
       from datetime import datetime, timedelta
       
       # 计算回测时间范围
       end_date = datetime.now()
       start_date = end_date - timedelta(days=4)
       
       # 使用策略初始化时的参数进行回测
       result = self.strategy.backtest(
           start_date=start_date.strftime('%Y-%m-%d'),
           end_date=end_date.strftime('%Y-%m-%d')
       )
       
       self.assertIsNotNone(result)
       self.assertGreaterEqual(result['total_trades'], 0)

def main():
   unittest.main()

if __name__ == '__main__':
   main()

#     === 回测结果 ===
# INFO:DonchianStrategy_TSLA:总交易次数: 35
# INFO:DonchianStrategy_TSLA:盈利交易: 28
# INFO:DonchianStrategy_TSLA:胜率: 80.00%
# INFO:DonchianStrategy_TSLA:总盈亏: $6.17
#?买入股数