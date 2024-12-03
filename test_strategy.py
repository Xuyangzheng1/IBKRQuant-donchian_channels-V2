# test_strategy.py
import unittest
import time
import pandas as pd
import numpy as np
from mock_tws import MockTWS
from improved_donchian_strategy import DonchianStrategy
import logging
import backtrader as bt
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
# from paste import DonchianStrategy  # 导入原策略文件


class DonchianStrategyWrapper(bt.Strategy):
    """Backtrader策略包装器，用于调用原始策略"""
    
    params = (
        ('donchian_strategy', None),  # 原始策略实例
    )
    
    def __init__(self):
        # 初始化原始策略
        self.strategy = self.params.donchian_strategy
        
        # 设置Backtrader数据源
        self.upper_channel = bt.indicators.Highest(self.data.high, period=self.strategy.period)
        self.lower_channel = bt.indicators.Lowest(self.data.low, period=self.strategy.period)
        self.middle_channel = (self.upper_channel + self.lower_channel) / 2
        
        # 交易记录
        self.trades = []
        self.last_trade_time = None
        self.min_trade_interval = timedelta(minutes=5)  # 保持原策略的交易间隔

    def next(self):
        """每个bar的策略执行"""
        # 获取当前时间
        current_time = bt.num2date(self.data.datetime[0])
        
        # 只在交易时段内执行策略
        if not self.is_trading_time(current_time):
            return
            
        # 检查交易间隔
        if self.last_trade_time and (current_time - self.last_trade_time) < self.min_trade_interval:
            return
        
        # 准备当前数据
        current_data = pd.DataFrame({
            'High': [self.data.high[0]],
            'Low': [self.data.low[0]],
            'Close': [self.data.close[0]],
            'Open': [self.data.open[0]],
            'Volume': [self.data.volume[0]]
        })
        
        # 计算通道数据
        current_data = self.strategy.calculate_channels(current_data)
        
        # 获取当前价格和通道值
        price = self.data.close[0]
        upper = self.upper_channel[0]
        lower = self.lower_channel[0]
        
        # 使用原策略的check_channel_position方法检查信号
        channel_pos = self.strategy.check_channel_position(price, upper, lower, current_data)
        
        # 交易执行
        position = self.getposition()
        
        # 检查是否接近收盘时间需要平仓
        if self.should_close_position(current_time) and position.size != 0:
            self.close()
            self.last_trade_time = current_time
            self.trades.append({
                'time': current_time,
                'action': 'SELL',
                'price': price,
                'quantity': position.size,
                'reason': 'MARKET_CLOSE'
            })
            return
        
        # 常规交易逻辑
        if channel_pos == "UPPER" and position.size > 0:
            # 触及上轨，平仓
            self.close()
            self.last_trade_time = current_time
            self.trades.append({
                'time': current_time,
                'action': 'SELL',
                'price': price,
                'quantity': position.size,
                'reason': 'SIGNAL'
            })
            
        elif channel_pos == "LOWER" and not position:
            # 触及下轨，开仓
            size = self.strategy.calculate_position_size(price)
            if size >= 10:
                self.buy(size=size)
                self.last_trade_time = current_time
                self.trades.append({
                    'time': current_time,
                    'action': 'BUY',
                    'price': price,
                    'quantity': size,
                    'reason': 'SIGNAL'
                })

    def is_trading_time(self, dt):
        """检查是否在交易时间内"""
        # 转换为美东时间
        et_time = dt.hour
        
        # 美股常规交易时间：9:30 - 16:00 ET
        if et_time < 9 or et_time >= 16:
            return False
        if et_time == 9 and dt.minute < 30:
            return False
            
        return True
        
    def should_close_position(self, dt):
        """检查是否需要收盘平仓"""
        et_time = dt.hour
        
        # 设定收盘前的安全时间窗口（15:55 - 16:00 ET）
        if et_time == 15 and dt.minute >= 55:
            return True
            
        return False

def get_minute_data(symbol, start_date, end_date):
    """获取分钟级数据"""
    try:
        # 从yfinance获取分钟级数据
        stock = yf.Ticker(symbol)
        hist = stock.history(start=start_date, end=end_date, interval='1m')
        
        if hist.empty:
            print("未能获取到数据")
            return None
            
        # 检查数据是否有效
        if len(hist) < 2:  # 至少需要两个数据点
            print("数据点数量不足")
            return None
            
        print(f"获取到 {len(hist)} 条数据")
        
        # 确保数据包含所需的所有列
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_columns:
            if col not in hist.columns:
                print(f"数据中缺少 {col} 列")
                return None
                
        # 去除任何包含 NaN 的行
        hist = hist.dropna()
        
        if hist.empty:
            print("清理 NaN 后数据为空")
            return None
            
        # 重置索引，使日期成为一列而不是索引
        hist = hist.reset_index()
        
        # 转换为Backtrader数据源
        data = bt.feeds.PandasData(
            dataname=hist,
            datetime='Datetime',  # 使用列名而不是索引
            open='Open',
            high='High',
            low='Low',
            close='Close',
            volume='Volume',
            openinterest=-1,
            timeframe=bt.TimeFrame.Minutes
        )
        
        return data
        
    except Exception as e:
        print(f"获取数据错误: {str(e)}")
        return None

def analyze_results(cerebro, strat, initial_capital):
    """分析和打印回测结果"""
    try:
        final_value = cerebro.broker.getvalue()
        pnl = final_value - initial_capital
        roi = (pnl / initial_capital) * 100
        
        sharpe = strat.analyzers.sharpe.get_analysis()
        drawdown = strat.analyzers.drawdown.get_analysis()
        trades = strat.analyzers.trades.get_analysis()
        
        print('\n=== 回测结果 ===')
        print(f'期末资金: ${final_value:,.2f}')
        print(f'总收益: ${pnl:,.2f} ({roi:.2f}%)')
        
        # 夏普比率可能为None
        sharpe_ratio = sharpe.get('sharperatio', 0.0)
        if sharpe_ratio is not None:
            print(f'夏普比率: {sharpe_ratio:.2f}')
        else:
            print('夏普比率: N/A')
            
        # 最大回撤可能为None
        max_drawdown = drawdown.get('max', {}).get('drawdown', 0.0)
        if max_drawdown is not None:
            print(f'最大回撤: {max_drawdown:.2f}%')
        else:
            print('最大回撤: N/A')
        
        # 交易统计
        if trades:
            total_trades = trades.get('total', {}).get('total', 0)
            won_trades = trades.get('won', {}).get('total', 0)
            lost_trades = trades.get('lost', {}).get('total', 0)
            
            print('\n=== 交易统计 ===')
            print(f'总交易次数: {total_trades}')
            print(f'盈利交易: {won_trades}')
            print(f'亏损交易: {lost_trades}')
            
            if total_trades > 0:
                win_rate = (won_trades / total_trades * 100)
                print(f'胜率: {win_rate:.2f}%')
                
            # 盈亏数据
            if 'won' in trades and 'lost' in trades:
                won_data = trades['won'].get('pnl', {})
                lost_data = trades['lost'].get('pnl', {})
                
                avg_win = won_data.get('average', 0) if won_data else 0
                avg_loss = lost_data.get('average', 0) if lost_data else 0
                
                if avg_loss != 0:
                    profit_factor = abs(avg_win / avg_loss)
                    print(f'平均盈利: ${avg_win:,.2f}')
                    print(f'平均亏损: ${avg_loss:,.2f}')
                    print(f'盈亏比: {profit_factor:.2f}')
        
    except Exception as e:
        print(f"分析结果处理错误: {str(e)}")

def run_backtest(symbol='TSLA', initial_capital=100000):
    """运行回测"""
    
    # 获取日期范围
    end_date = datetime.now()
    start_date = end_date - timedelta(days=1)  # 使用较短时间范围
    
    print(f'\n开始回测 {symbol}...')
    print(f'时间范围: {start_date.strftime("%Y-%m-%d")} 到 {end_date.strftime("%Y-%m-%d")}')
    print(f'初始资金: ${initial_capital:,.2f}')
    
    # 创建原始策略实例
    strategy = DonchianStrategy(
        symbol=symbol,
        period=20,
        capital=initial_capital,
        alert_threshold=0.004,
        max_capital_per_trade=50000,
        log_dir='backtest_logs',
        csv_dir='backtest_results'
    )
    
    # 创建 Cerebro 引擎
    cerebro = bt.Cerebro()
    
    # 获取分钟级数据
    data = get_minute_data(symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    if data is None:
        print("无法获取分钟级数据")
        return None
    
    # 添加数据
    cerebro.adddata(data)
    
    # 设置初始资金
    cerebro.broker.setcash(initial_capital)
    
    # 设置手续费
    cerebro.broker.setcommission(commission=0.001)
    
    # 添加策略包装器
    cerebro.addstrategy(DonchianStrategyWrapper, donchian_strategy=strategy)
    
    # 添加分析器
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='returns')
    
    try:
        # 运行回测
        results = cerebro.run()
        strat = results[0]
        
        # 分析结果
        analyze_results(cerebro, strat, initial_capital)
        
        # 绘制图表
        cerebro.plot(style='candlestick', volume=True)
        
        return {
            'strategy': strat,
            'trades': strat.trades,
            'analyzers': {
                'sharpe': strat.analyzers.sharpe.get_analysis(),
                'drawdown': strat.analyzers.drawdown.get_analysis(),
                'trades': strat.analyzers.trades.get_analysis(),
                'returns': strat.analyzers.returns.get_analysis()
            }
        }
    except Exception as e:
        print(f"回测执行错误: {str(e)}")
        return None
if __name__ == "__main__":
    # 运行回测示例
    results = run_backtest('tsla', 100000)
    
    # 保存交易记录
    if results and results['trades']:
        trades_df = pd.DataFrame(results['trades'])
        trades_df.to_csv('backtest_trades.csv', index=False)