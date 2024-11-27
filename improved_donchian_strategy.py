


import yfinance as yf
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import Order
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import threading
import time
import logging
import os

class Trade:
    def __init__(self, action, price, quantity, timestamp, channel_price):
        self.action = action
        self.price = price
        self.quantity = quantity
        self.timestamp = timestamp
        self.channel_price = channel_price

class TradingBot(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.positions = {'TSLA': 0}  # 初始化仓位字典
        self.current_price = None
        self.order_id = 0
        self.market_depth = {}
        
    def error(self, reqId, errorCode, errorString):
        logging.error(f'Error {errorCode}: {errorString}')
        
    def nextValidId(self, orderId):
        self.order_id = orderId
        logging.info("Connected to TWS")
        
    def position(self, account, contract, pos, avgCost):
        self.positions[contract.symbol] = pos
        
    def tickPrice(self, reqId, tickType, price, attrib):
        if tickType == 4:
            self.current_price = price
            
    def marketDepth(self, reqId, position, operation, side, price, size):
        if reqId not in self.market_depth:
            self.market_depth[reqId] = {'bids': [], 'asks': []}
        
        depth_side = 'bids' if side == 0 else 'asks'
        
        if operation == 0:  # Insert
            self.market_depth[reqId][depth_side].append({'price': price, 'size': size})
        elif operation == 1:  # Update
            if position < len(self.market_depth[reqId][depth_side]):
                self.market_depth[reqId][depth_side][position] = {'price': price, 'size': size}
        elif operation == 2:  # Delete
            if position < len(self.market_depth[reqId][depth_side]):
                del self.market_depth[reqId][depth_side][position]

class ImprovedDonchianStrategy:
    def __init__(self, symbol, capital=100000, period=20, base_tranches=3, 
                 alert_threshold=0.004, max_capital_per_trade=50000):
        # 基础参数
        self.symbol = symbol
        self.capital = capital
        self.period = period
        self.base_tranches = base_tranches
        self.alert_threshold = alert_threshold
        self.max_capital_per_trade = max_capital_per_trade
        
        # 交易相关
        self.position = 0
        self.current_position_size = 0
        self.total_cost_basis = 0
        self.realized_pnl = 0
        self.last_trade_time = None
        self.min_trade_interval = 300
        
        # 交易记录
        self.trades = []
        self.daily_trades = []
        self.trade_history = []
        
        # 日志设置
        self.log_dir = r"F:\shares\twspy\trading_logs\trading_logs"
        self.csv_dir = r"F:\shares\twspy\trading_logs\trading_results"
        for directory in [self.log_dir, self.csv_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
        
        # 初始化交易接口和日志
        self.bot = TradingBot()
        self.logger = self._setup_logger()
        self.csv_filename = self._setup_csv_file()
        
    def calculate_position_size(self, price):
        """计算交易量"""
        try:
            # 确保每次交易不超过最大资金限制
            max_shares = int(self.max_capital_per_trade / price)
            # 确保最小交易数量
            return max(10, max_shares)
        except Exception as e:
            self.logger.error(f"计算交易量错误: {str(e)}")
            return 0
    def verify_position(self):
        """验证当前仓位信息"""
        try:
            # 从IB获取实际仓位
            ib_position = self.bot.positions.get(self.symbol, 0)
            
            # 检查是否与我们的记录一致
            if ib_position != self.current_position_size:
                self.logger.warning(f"仓位不一致! IB仓位: {ib_position}, 本地记录: {self.current_position_size}")
                # 以IB仓位为准
                self.current_position_size = ib_position
                
            # 确保没有负仓位
            if self.current_position_size < 0:
                self.logger.error(f"检测到负仓位: {self.current_position_size}，立即修正")
                self.current_position_size = 0
                
            return self.current_position_size
            
        except Exception as e:
            self.logger.error(f"验证仓位错误: {str(e)}")
            return 0

    def _setup_logger(self):
        """设置日志系统"""
        logger = logging.getLogger(f'DonchianStrategy_{self.symbol}')
        logger.setLevel(logging.INFO)
        
        timestamp = datetime.now().strftime('%Y%m%d')
        filename = os.path.join(self.log_dir, f'donchian_{self.symbol}_{timestamp}.log')
        
        fh = logging.FileHandler(filename)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
        return logger

    def _setup_csv_file(self):
        """设置CSV文件"""
        timestamp = datetime.now().strftime('%Y%m%d')
        filename = os.path.join(self.csv_dir, f'trades_{self.symbol}_{timestamp}.csv')
        
        if not os.path.exists(filename):
            headers = [
                'Timestamp', 'Action', 'Price', 'Quantity', 'Channel_Price',
                'Position_Size', 'Avg_Cost', 'Realized_PnL', 'Unrealized_PnL',
                'Upper_Channel', 'Lower_Channel', 'Current_Price', 'Trade_Value'
            ]
            pd.DataFrame(columns=headers).to_csv(filename, index=False)
            
        return filename

    def calculate_baseline_channel_width(self, df):
        """计算基准通道宽度（前N天的平均宽度）"""
        try:
            # 计算每个时间点的通道宽度
            df['channel_width'] = (df['upper_channel'] - df['lower_channel']) / df['Close']
            
            # 获取最近三天的数据
            lookback_periods = self.period * 3
            recent_widths = df['channel_width'].tail(lookback_periods)
            
            # 计算平均宽度
            baseline_width = recent_widths.mean()
            self.logger.info(f"基准通道宽度: {baseline_width*100:.2f}%")
            
            return baseline_width
            
        except Exception as e:
            self.logger.error(f"计算基准通道宽度错误: {str(e)}")
            return 0.03

    def adjust_threshold(self, current_width, baseline_width):
        """根据当前通道宽度相对于基准宽度调整阈值"""
        try:
            # 计算宽度比率
            width_ratio = current_width / baseline_width
            
            # 基础阈值
            base_threshold = self.alert_threshold
            
            if width_ratio < 0.8:  # 当前通道比基准窄20%以上
                # 降低阈值，让信号更容易触发
                adjusted_threshold = base_threshold * 0.8
                self.logger.info(f"通道较窄，降低阈值至: {adjusted_threshold*100:.3f}%")
            elif width_ratio > 1.2:  # 当前通道比基准宽20%以上
                # 提高阈值，让信号更难触发
                adjusted_threshold = base_threshold * 1.2
                self.logger.info(f"通道较宽，提高阈值至: {adjusted_threshold*100:.3f}%")
            else:
                # 保持基础阈值
                adjusted_threshold = base_threshold
                self.logger.info(f"通道正常，使用基础阈值: {adjusted_threshold*100:.3f}%")
                
            return adjusted_threshold
            
        except Exception as e:
            self.logger.error(f"调整阈值错误: {str(e)}")
            return self.alert_threshold

    def is_near_channel(self, price, upper, lower, df):
        """改进后的通道位置检查"""
        try:
            # 计算当前通道宽度
            current_width = (upper - lower) / price
            
            # 计算基准通道宽度
            baseline_width = self.calculate_baseline_channel_width(df)
            
            # 动态调整阈值
            adjusted_threshold = self.adjust_threshold(current_width, baseline_width)
            
            # 计算与通道的距离
            upper_dist = abs(price - upper) / upper
            lower_dist = abs(price - lower) / lower
            
            # 记录详细信息
            self.logger.info(f"\n=== 通道分析 ===")
            self.logger.info(f"当前通道宽度: {current_width*100:.2f}%")
            self.logger.info(f"基准通道宽度: {baseline_width*100:.2f}%")
            self.logger.info(f"调整后阈值: {adjusted_threshold*100:.3f}%")
            self.logger.info(f"与上轨距离: {upper_dist*100:.3f}%")
            self.logger.info(f"与下轨距离: {lower_dist*100:.3f}%")
            
            # 使用调整后的阈值判断
            if upper_dist <= adjusted_threshold:
                return "UPPER"
            elif lower_dist <= adjusted_threshold:
                return "LOWER"
                    
            return None
                
        except Exception as e:
            self.logger.error(f"检查通道位置错误: {str(e)}")
            return None

    def save_trade_to_csv(self, trade, current_data):
        try:
            position_size = self.current_position_size
            avg_cost = self.total_cost_basis / position_size if position_size > 0 else 0
            unrealized_pnl = position_size * (current_data['current_price'] - avg_cost) if position_size > 0 else 0
            
            trade_record = {
                'Timestamp': trade.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'Action': trade.action,
                'Price': trade.price,
                'Quantity': trade.quantity,
                'Channel_Price': trade.channel_price,
                'Position_Size': position_size,
                'Avg_Cost': avg_cost,
                'Realized_PnL': self.realized_pnl,
                'Unrealized_PnL': unrealized_pnl,
                'Upper_Channel': current_data.get('upper_channel', 0),
                'Lower_Channel': current_data.get('lower_channel', 0),
                'Current_Price': current_data['current_price'],
                'Trade_Value': trade.quantity * trade.price
            }
            pd.DataFrame([trade_record]).to_csv(self.csv_filename, mode='a', header=False, index=False)
            
        except Exception as e:
            self.logger.error(f"保存交易记录到CSV时发生错误: {str(e)}")
            self.logger.error(f"Debug - Error details: {e.__class__.__name__}")

    def log_trade(self, action, price, quantity, channel_price):
        """记录交易"""
        try:
            # 创建交易对象
            trade = Trade(action, price, quantity, datetime.now(), channel_price)
            self.trades.append(trade)
            self.daily_trades.append(trade)
            
            # 获取当前市场数据
            current_data = {
                'current_price': price,
                'upper_channel': channel_price if action == "SELL" else 0,
                'lower_channel': channel_price if action == "BUY" else 0
            }
            
            # 保存到CSV
            self.save_trade_to_csv(trade, current_data)
            
            # 记录到日志
            self.logger.info(f"执行交易: {action} {quantity}股 @ ${price:.2f}")
            self.logger.info(f"触发通道价格: ${channel_price:.2f}")
            self.logger.info(f"当前持仓: {self.current_position_size}")
            if self.current_position_size > 0:
                self.logger.info(f"平均成本: ${self.total_cost_basis/self.current_position_size:.2f}")
            self.logger.info(f"已实现盈亏: ${self.realized_pnl:.2f}")
            
        except Exception as e:
            self.logger.error(f"记录交易时发生错误: {str(e)}")

    def get_market_data(self):
        """获取市场数据"""
        try:
            stock = yf.Ticker(self.symbol)
            end = datetime.now()
            start = end - timedelta(days=5)
            df = stock.history(start=start, end=end, interval='1m')
            if df.empty:
                raise ValueError("No data received")
            return df
        except Exception as e:
            self.logger.error(f"获取市场数据错误: {str(e)}")
            raise

    def get_donchian_channels(self, df):
        """计算唐奇安通道"""
        try:
            df = df.copy()
            df['upper_channel'] = df['High'].rolling(window=self.period).max()
            df['lower_channel'] = df['Low'].rolling(window=self.period).min()
            df['middle_channel'] = (df['upper_channel'] + df['lower_channel']) / 2
            return df
        except Exception as e:
            self.logger.error(f"计算通道错误: {str(e)}")
            return df

    def can_trade(self):
        """检查是否可以交易"""
        # 在测试环境中不检查交易间隔
        if hasattr(self, '_test_mode') and self._test_mode:
            return True
        if not self.last_trade_time:
            return True
        time_passed = (datetime.now() - self.last_trade_time).seconds
        can_trade = time_passed > self.min_trade_interval
        
        if not can_trade:
            self.logger.info(f"距离上次交易仅过去{time_passed}秒，需要等待{self.min_trade_interval-time_passed}秒")
            
        return can_trade

    def execute_tranche(self, action, position_size, price):
        """执行单批次交易"""
        try:
            contract = Contract()
            contract.symbol = self.symbol
            contract.secType = "STK"
            contract.exchange = "SMART"
            contract.currency = "USD"
            contract.primaryExchange = "NASDAQ"
            
            order = Order()
            order.action = action
            order.orderType = "MKT"
            order.totalQuantity = position_size
            order.eTradeOnly = False
            order.firmQuoteOnly = False
            
            self.logger.info(f"准备执行订单 - {action} {order.totalQuantity}股")
            
            self.bot.placeOrder(self.bot.order_id, contract, order)
            self.bot.order_id += 1
            
            return order.totalQuantity
                
        except Exception as e:
            self.logger.error(f"执行批次交易错误: {str(e)}")
            return 0

    def execute_trade(self, action, price, upper_channel, lower_channel):
        """执行交易"""
        try:
            if not self.can_trade():
                return False
            
            # 计算交易量
            position_size = self.calculate_position_size(price)
            
            if position_size < 10:
                self.logger.info("交易量过小，取消交易")
                return False
            
            # 执行交易
            filled = self.execute_tranche(action, position_size, price)
            if filled:
                self.update_position_info(action, filled, price)
                self.log_trade(action, price, filled, 
                             upper_channel if action == "SELL" else lower_channel)
                self.last_trade_time = datetime.now()
            
            # 交易完成后验证仓位
            self.verify_position()
            return filled > 0
                
        except Exception as e:
            self.logger.error(f"执行交易错误: {str(e)}")
            self.verify_position()
            return False

    def update_position_info(self, action, filled_quantity, price):
        """更新持仓信息"""
        try:
            if action == "BUY":
                new_position = self.current_position_size + filled_quantity
                self.total_cost_basis += filled_quantity * price
            else:  # SELL
                new_position = self.current_position_size - filled_quantity
                if self.current_position_size > 0:
                    avg_cost = self.total_cost_basis / self.current_position_size
                    realized_pnl = filled_quantity * (price - avg_cost)
                    self.realized_pnl += realized_pnl
                    
            # 更新仓位
            self.current_position_size = new_position
            if new_position == 0:
                self.total_cost_basis = 0
                
            # 验证更新后的仓位
            actual_position = self.verify_position()
            if actual_position != self.current_position_size:
                self.logger.error(f"仓位更新错误: 本地={self.current_position_size}, IB={actual_position}")
                self.current_position_size = actual_position
                
        except Exception as e:
            self.logger.error(f"更新持仓信息错误: {str(e)}")
            self.verify_position()

    def log_daily_summary(self):
        """记录每日交易摘要"""
        self.logger.info("\n=== 每日交易摘要 ===")
        self.logger.info(f"日期: {datetime.now().strftime('%Y-%m-%d')}")
        self.logger.info(f"交易次数: {len(self.daily_trades)}")
        
        daily_pnl = sum([
            trade.quantity * trade.price if trade.action == "SELL" else -trade.quantity * trade.price 
            for trade in self.daily_trades
        ])
        
        self.logger.info(f"当日交易:")
        for trade in self.daily_trades:
            self.logger.info(
                f"时间: {trade.timestamp.strftime('%H:%M:%S')}, "
                f"动作: {trade.action}, "
                f"数量: {trade.quantity}, "
                f"价格: ${trade.price:.2f}, "
                f"通道价格: ${trade.channel_price:.2f}"
            )
        
        self.logger.info(f"当日盈亏: ${daily_pnl:.2f}")
        self.logger.info(f"当前持仓: {self.current_position_size}")
        if self.current_position_size > 0:
            self.logger.info(f"平均成本: ${self.total_cost_basis/self.current_position_size:.2f}")
        self.logger.info(f"总实现盈亏: ${self.realized_pnl:.2f}")
        self.logger.info("==================")
        self.export_daily_summary_to_csv()
        # 清空当日交易记录
        self.daily_trades = []

    def export_daily_summary_to_csv(self):
        """导出每日交易摘要到CSV"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d')
            summary_filename = os.path.join(self.csv_dir, f'daily_summary_{self.symbol}_{timestamp}.csv')
            
            daily_stats = {
                'Date': datetime.now().strftime('%Y-%m-%d'),
                'Total_Trades': len(self.daily_trades),
                'Buy_Trades': sum(1 for t in self.daily_trades if t.action == "BUY"),
                'Sell_Trades': sum(1 for t in self.daily_trades if t.action == "SELL"),
                'Total_Volume': sum(t.quantity for t in self.daily_trades),
                'Total_Value': sum(t.quantity * t.price for t in self.daily_trades),
                'Final_Position': self.current_position_size,
                'Realized_PnL': self.realized_pnl
            }
            
            pd.DataFrame([daily_stats]).to_csv(
                summary_filename,
                mode='a',
                header=not os.path.exists(summary_filename),
                index=False
            )
            
            self.logger.info(f"每日交易摘要已保存到: {summary_filename}")
            
        except Exception as e:
            self.logger.error(f"导出每日摘要时发生错误: {str(e)}")

    
    # 在 ImprovedDonchianStrategy 类中添加
    # 在 ImprovedDonchianStrategy 类中

    def backtest(self, start_date=None, end_date=None, initial_position=0):
        """回测功能"""
        try:
            # 使用策略初始化时的参数
            symbol = self.symbol
            test_capital = self.capital        # 初始资金
            test_period = self.period          # 通道周期
            max_trade = self.max_capital_per_trade  # 单次最大交易金额
            
            # 获取测试数据
            stock = yf.Ticker(symbol)
            
            if start_date and end_date:
                hist = stock.history(start=start_date, end=end_date, interval='1m')
            else:
                hist = stock.history(period='4d', interval='1m')
                
            if hist.empty:
                self.logger.error("未能获取历史数据")
                return None
                
            self.logger.info(f"回测时间范围: {hist.index[0]} 至 {hist.index[-1]}")
            self.logger.info(f"数据点数量: {len(hist)}")
            
            # 使用初始化参数
            self.logger.info(f"回测标的: {symbol}")
            self.logger.info(f"初始资金: ${test_capital}")
            self.logger.info(f"通道周期: {test_period}")
            self.logger.info(f"单次最大交易额: ${max_trade}")

            # 初始化回测数据
            available_capital = test_capital  # 可用资金
            total_trades = 0
            winning_trades = 0
            total_pnl = 0
            position = initial_position
            entry_price = 0
            trade_size = 0
            trade_records = []
            
            # 使用test_period计算通道
            hist['upper_channel'] = hist['High'].rolling(window=test_period).max()
            hist['lower_channel'] = hist['Low'].rolling(window=test_period).min()
            
            # 遍历数据
            for i in range(len(hist)):
                row = hist.iloc[i]
                price = row['Close']
                upper = row['upper_channel']
                lower = row['lower_channel']
                timestamp = hist.index[i]
                
                # 检查是否触及通道
                lookback_data = hist.iloc[max(0, i-test_period):i+1]  # 使用test_period
                channel_position = self.is_near_channel(price, upper, lower, lookback_data)
                    
                if channel_position == "LOWER" and position == 0:
                    # 计算可买数量（考虑资金限制）
                    max_shares = min(
                        int(available_capital / price),  # 可用资金能买的数量
                        int(max_trade / price)          # 单次交易限制的数量
                    )
                    trade_size = max_shares
                    
                    if trade_size > 0:
                        position = 1
                        entry_price = price
                        available_capital -= (trade_size * price)  # 更新可用资金
                        total_trades += 1
                        
                        trade_records.append({
                            'time': timestamp,
                            'action': 'BUY',
                            'price': price,
                            'size': trade_size,
                            'available_capital': available_capital,
                            'reason': f'价格(${price:.2f})接近下轨(${lower:.2f})',
                            'channel_width': f'{(upper-lower)/price*100:.2f}%'
                        })
                        
                        self.logger.info(
                            f"买入 - 时间: {timestamp}, "
                            f"数量: {trade_size}股, "
                            f"价格: ${price:.2f}, "
                            f"可用资金: ${available_capital:.2f}, "
                            f"原因: 价格接近下轨(${lower:.2f}), "
                            f"通道宽度: {(upper-lower)/price*100:.2f}%"
                        )
                    
                elif channel_position == "UPPER" and position == 1:
                    position = 0
                    pnl = (price - entry_price) * trade_size
                    total_pnl += pnl
                    available_capital += (trade_size * price)  # 更新可用资金
                    
                    if pnl > 0:
                        winning_trades += 1
                        
                    trade_records.append({
                        'time': timestamp,
                        'action': 'SELL',
                        'price': price,
                        'size': trade_size,
                        'pnl': pnl,
                        'available_capital': available_capital,
                        'reason': f'价格(${price:.2f})接近上轨(${upper:.2f})',
                        'channel_width': f'{(upper-lower)/price*100:.2f}%'
                    })
                    
                    self.logger.info(
                        f"卖出 - 时间: {timestamp}, "
                        f"数量: {trade_size}股, "
                        f"价格: ${price:.2f}, "
                        f"盈亏: ${pnl:.2f}, "
                        f"可用资金: ${available_capital:.2f}, "
                        f"原因: 价格接近上轨(${upper:.2f}), "
                        f"通道宽度: {(upper-lower)/price*100:.2f}%"
                    )
            
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            
            self.logger.info("\n=== 回测结果 ===")
            self.logger.info(f"初始资金: ${test_capital}")
            self.logger.info(f"最终资金: ${available_capital}")
            self.logger.info(f"总盈亏: ${total_pnl:.2f}")
            self.logger.info(f"总交易次数: {total_trades}")
            self.logger.info(f"盈利交易: {winning_trades}")
            self.logger.info(f"胜率: {win_rate:.2f}%")
            
            return {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'final_capital': available_capital,
                'trade_records': trade_records
            }
            
        except Exception as e:
            self.logger.error(f"回测过程中发生错误: {str(e)}")
            raise
    
    def run_strategy(self):
        """运行策略"""
        try:
            self.logger.info("\n=== 策略启动 ===")
            self.logger.info(f"交易品种: {self.symbol}")
            self.logger.info(f"初始资金: ${self.capital:,}")
            
            # 连接交易接口
            self.bot.connect("127.0.0.1", 7497, 1)
            
            api_thread = threading.Thread(target=self.bot.run)
            api_thread.daemon = True
            api_thread.start()
            
            time.sleep(1)
            last_trading_day = None
            
            while True:
                try:
                    now = datetime.now()
                    current_trading_day = now.date()
                    
                    if last_trading_day != current_trading_day:
                        if last_trading_day is not None:
                            self.log_daily_summary()
                        last_trading_day = current_trading_day
                        self.logger.info(f"\n=== 新交易日开始: {current_trading_day} ===")
                    
                    # 获取市场数据
                    df = self.get_market_data()
                    df = self.get_donchian_channels(df)
                    latest = df.iloc[-1]
                    
                    current_price = latest['Close']
                    upper_channel = latest['upper_channel']
                    lower_channel = latest['lower_channel']
                    
                    # 传入完整df用于计算基准宽度
                    channel_position = self.is_near_channel(
                        current_price,
                        upper_channel,
                        lower_channel,
                        df
                    )
                    
                    # 记录状态
                    self.log_status(current_price, upper_channel, lower_channel)
                    
                    # 交易逻辑
                    if channel_position == "UPPER" and self.current_position_size > 0:
                        # 触及上轨，平仓
                        self.logger.info("\n检测到上轨突破信号 - 准备卖出")
                        if self.execute_trade("SELL", current_price, upper_channel, lower_channel):
                            self.position = 0
                    
                    elif channel_position == "LOWER":
                        # 触及下轨，考虑开仓或加仓
                        self.logger.info("\n检测到下轨支撑信号 - 准备买入")
                        if self.execute_trade("BUY", current_price, upper_channel, lower_channel):
                            self.position = 1
                    
                    # 收盘检查
                    if now.hour == 16 and now.minute == 0:
                        self.handle_market_close(current_price, upper_channel, lower_channel)
                    
                    time.sleep(5)
                    
                except KeyboardInterrupt:
                    self.logger.info("\n策略手动停止")
                    break
                    
                except Exception as e:
                    self.logger.error(f"策略运行错误: {str(e)}")
                    time.sleep(5)
                    continue
                    
        except Exception as e:
            self.logger.error(f"严重错误: {str(e)}")
        finally:
            if hasattr(self, 'bot'):
                self.bot.disconnect()

    def log_status(self, current_price, upper_channel, lower_channel):
        """记录当前状态"""
        self.logger.info(f"\n=== {datetime.now().strftime('%H:%M:%S')} 策略状态 ===")
        self.logger.info(f"当前价格: ${current_price:.2f}")
        self.logger.info(f"上轨: ${upper_channel:.2f}")
        self.logger.info(f"下轨: ${lower_channel:.2f}")
        self.logger.info(f"持仓数量: {self.current_position_size}")
        
        if self.current_position_size > 0:
            avg_cost = self.total_cost_basis / self.current_position_size
            unrealized_pnl = self.current_position_size * (current_price - avg_cost)
            self.logger.info(f"平均成本: ${avg_cost:.2f}")
            self.logger.info(f"未实现盈亏: ${unrealized_pnl:.2f}")
        self.logger.info(f"已实现盈亏: ${self.realized_pnl:.2f}")

    def handle_market_close(self, current_price, upper_channel, lower_channel):
        """处理收盘时的逻辑"""
        self.log_daily_summary()
        if self.current_position_size > 0:
            self.logger.info("收盘前平仓")
            self.execute_trade("SELL", current_price, upper_channel, lower_channel)

def main():
    config = {
        'symbol': 'TSLA',
        'capital': 10000,
        'period': 20,
        'base_tranches': 3,
        'alert_threshold': 0.002,
        'max_capital_per_trade': 50000
    }
    
    strategy = ImprovedDonchianStrategy(**config)
    
    try:
        strategy.run_strategy()
    except KeyboardInterrupt:
        strategy.logger.info("\n程序手动停止")
    except Exception as e:
        strategy.logger.error(f"\n程序运行错误: {str(e)}")
    finally:
        if hasattr(strategy, 'bot'):
            strategy.bot.disconnect()

if __name__ == "__main__":
    main()