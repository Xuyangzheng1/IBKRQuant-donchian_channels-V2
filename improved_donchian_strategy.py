
from decimal import Decimal
import random
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
import logging
from datetime import datetime
import time
#通道越窄，阈值越高还是越低？
#操作后是否有间隔？（测试环境中不检查间隔）

#应对连续下跌情况
#持仓显示
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
        self.positions = {}
        self.current_price = None
        self.order_id = 0
        self.connected = False
        self.position_received = False
        self.pnl_single_received = False  # 新增
        self.logger = logging.getLogger(__name__)
        self.stock_pnls = {}  # 新增
        self.account = None
        
    def error(self, reqId, errorCode, errorString):
        logging.error(f'Error {errorCode}: {errorString}')
        if errorCode == 102:  # 复制产品代码错误
            if reqId in self.pnl_request_map:
                symbol = self.pnl_request_map[reqId]
                self.logger.error(f"获取{symbol}的PnL信息失败，可能需要订阅市场数据")
        elif errorCode == 502:
            self.connected = False
            
    def nextValidId(self, orderId):
        self.order_id = orderId
        self._connected = True  # 使用新的变量名
        logging.info("Connected to TWS")
        self.reqPositions()
    def pnlSingle(self, reqId: int, pos: int, dailyPnL: float, unrealizedPnL: float, realizedPnL: float, value: float):
        """接收单个股票的盈亏回调"""
        if reqId in self.pnl_request_map:
            symbol = self.pnl_request_map[reqId]
            self.stock_pnls[symbol] = {
                'position': pos,
                'daily_pnl': dailyPnL,
                'unrealized_pnl': unrealizedPnL,
                'realized_pnl': realizedPnL,
                'value': value
            }
            self.pnl_single_received = True
            
            self.logger.info(f"{symbol} PnL更新")
            self.logger.info(f"当日盈亏: ${dailyPnL:.2f}")
            self.logger.info(f"未实现盈亏: ${unrealizedPnL:.2f}")
            self.logger.info(f"已实现盈亏: ${realizedPnL:.2f}")

    def reqStockPnL(self, symbol: str):
        """请求特定股票的PnL"""
        try:
            if symbol in self.positions and self.account:
                contract = self.positions[symbol]['contract']
                if not contract.conId:
                    self.logger.error(f"无法获取{symbol}的合约ID")
                    return
                    
                # 先订阅市场数据
                self.reqMktData(self.order_id, contract, "", False, False, [])
                time.sleep(0.5)  # 等待市场数据
                    
                req_id = self.order_id + 1
                self.pnl_request_map[req_id] = symbol
                
                # 请求特定股票的PnL
                self.reqPnLSingle(req_id, self.account, "", contract.conId)
                
        except Exception as e:
            self.logger.error(f"请求股票PnL时出错: {str(e)}")
            
    def getStockPnL(self, symbol: str):
        """获取特定股票的盈亏信息"""
        return self.stock_pnls.get(symbol, {
            'daily_pnl': 0,
            'unrealized_pnl': 0,
            'realized_pnl': 0,
            'value': 0
        })   
    def position(self, account: str, contract: Contract, pos: Decimal, avgCost: float):
        """接收持仓回调"""
        symbol = contract.symbol
        self.logger.info(f"Position received - Symbol: {symbol}, Position: {pos}, AvgCost: {avgCost}")
        
        self.positions[symbol] = {
            'position': float(pos),
            'avg_cost': float(avgCost),
            'account': account,
            'contract': contract,  # 新增：保存合约信息
            'last_update': datetime.now()
        }
        
        if account:
            self.account = account
    def positionEnd(self):
        """持仓数据接收完成"""
        self.position_received = True
        self.logger.info("Position data end")
        self.pnl_request_map = {}  # 新增
    def pnl(self, reqId: int, dailyPnL: float, unrealizedPnL: float, realizedPnL: float):
        """接收盈亏回调"""
        self.daily_pnl = dailyPnL
        self.unrealized_pnl = unrealizedPnL
        self.realized_pnl = realizedPnL
        self.pnl_received = True
        
        self.logger.info(f"PnL更新")
        self.logger.info(f"当日盈亏: ${dailyPnL:.2f}")
        self.logger.info(f"未实现盈亏: ${unrealizedPnL:.2f}")
        self.logger.info(f"已实现盈亏: ${realizedPnL:.2f}")
        
    def getPnL(self):
        """获取最新的盈亏信息"""
        return {
            'daily_pnl': self.daily_pnl,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl
        }        
    def isConnected(self):  # 添加方法来检查连接状态
        """返回连接状态"""
        return self.isConnected   
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


class Position:
    def __init__(self, symbol, logger=None):
        self.symbol = symbol
        self.quantity = 0
        self.avg_cost = 0
        self.total_cost = 0
        self.realized_pnl = 0
        self.unrealized_pnl = 0
        self.last_update_time = None
        self.logger = logger or logging.getLogger(__name__)
        self.trades_today = []
        
    def update_from_ib(self, quantity, avg_cost):
        """从IB更新持仓信息"""
        try:
            old_quantity = self.quantity
            self.quantity = quantity
            if quantity > 0:
                self.avg_cost = avg_cost
                self.total_cost = quantity * avg_cost
            else:
                self.avg_cost = 0
                self.total_cost = 0
                
            self.last_update_time = datetime.now()
            
            self.logger.info(f"持仓更新 - {self.symbol}")
            self.logger.info(f"数量: {old_quantity} -> {self.quantity}")
            self.logger.info(f"平均成本: ${self.avg_cost:.2f}")
            self.logger.info(f"总成本: ${self.total_cost:.2f}")
            
        except Exception as e:
            self.logger.error(f"更新持仓信息时出错: {str(e)}")
            
    def update_market_price(self, current_price):
        """更新市场价格相关信息"""
        try:
            if self.quantity > 0:
                self.unrealized_pnl = self.quantity * (current_price - self.avg_cost)
            else:
                self.unrealized_pnl = 0
                
        except Exception as e:
            self.logger.error(f"更新市场价格信息时出错: {str(e)}")
            
    def record_trade(self, action, quantity, price, timestamp=None):
        """记录交易"""
        try:
            timestamp = timestamp or datetime.now()
            
            if action == "SELL":
                if self.quantity > 0:
                    # 计算已实现盈亏
                    trade_pnl = quantity * (price - self.avg_cost)
                    self.realized_pnl += trade_pnl
                    
                    # 更新持仓
                    self.quantity -= quantity
                    if self.quantity <= 0:
                        self.quantity = 0
                        self.avg_cost = 0
                        self.total_cost = 0
                    else:
                        self.total_cost = self.quantity * self.avg_cost
                        
            elif action == "BUY":
                # 更新平均成本和总成本
                total_cost = self.total_cost + (quantity * price)
                self.quantity += quantity
                self.avg_cost = total_cost / self.quantity
                self.total_cost = total_cost
                
            # 记录交易
            trade = {
                'timestamp': timestamp,
                'action': action,
                'quantity': quantity,
                'price': price,
                'position_after': self.quantity,
                'avg_cost_after': self.avg_cost,
                'realized_pnl': self.realized_pnl
            }
            self.trades_today.append(trade)
            
            # 记录日志
            self.logger.info(f"\n=== 交易记录 ===")
            self.logger.info(f"时间: {timestamp}")
            self.logger.info(f"动作: {action}")
            self.logger.info(f"数量: {quantity}")
            self.logger.info(f"价格: ${price:.2f}")
            self.logger.info(f"交易后持仓: {self.quantity}")
            self.logger.info(f"交易后均价: ${self.avg_cost:.2f}")
            if action == "SELL":
                self.logger.info(f"已实现盈亏: ${self.realized_pnl:.2f}")
                
        except Exception as e:
            self.logger.error(f"记录交易时出错: {str(e)}")
            
    def get_daily_summary(self):
        """获取每日交易摘要"""
        return {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'symbol': self.symbol,
            'total_trades': len(self.trades_today),
            'current_position': self.quantity,
            'avg_cost': self.avg_cost,
            'realized_pnl': self.realized_pnl,
            'unrealized_pnl': self.unrealized_pnl,
            'total_pnl': self.realized_pnl + self.unrealized_pnl
        }
        
    def reset_daily_trades(self):
        """重置每日交易记录"""
        self.trades_today = []
        
    def verify_position(self):
        """验证当前仓位信息"""
        try:
            # 等待直到收到持仓数据
            wait_count = 0
            while not self.bot.position_received and wait_count < 10:
                time.sleep(0.5)
                wait_count += 1
                
            position_info = self.bot.positions.get(self.symbol, {})
            if position_info:
                ib_position = position_info.get('position', 0)
                avg_cost = position_info.get('avg_cost', 0)
                
                if ib_position != self.current_position_size:
                    self.logger.warning(f"仓位不一致! IB仓位: {ib_position}, 本地记录: {self.current_position_size}")
                    self.current_position_size = ib_position
                    self.total_cost_basis = ib_position * avg_cost
                    
            return self.current_position_size
            
        except Exception as e:
            self.logger.error(f"验证仓位错误: {str(e)}")
            return 0
        
    def get_position_info(self):
        """获取当前持仓信息"""
        return {
            'symbol': self.symbol,
            'quantity': self.quantity,
            'avg_cost': self.avg_cost,
            'total_cost': self.total_cost,
            'realized_pnl': self.realized_pnl,
            'unrealized_pnl': self.unrealized_pnl,
            'last_update': self.last_update_time
        }

class PositionManager:
    def __init__(self, bot, logger=None):
        self.bot = bot
        self.logger = logger or logging.getLogger(__name__)
        self.positions = {}
        self.last_verification_time = None
        self.verification_interval = 300  # 5分钟验证一次
        
    def initialize_position(self, symbol):
        """初始化品种的持仓跟踪"""
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol, self.logger)
            
    def update_from_ib(self):
        """从IB更新所有持仓信息"""
        try:
            # 请求持仓更新
            self.bot.reqPositions()
            
            # 等待持仓数据
            wait_count = 0
            while not self.bot.position_received and wait_count < 10:
                time.sleep(0.5)
                wait_count += 1
                
            if not self.bot.position_received:
                self.logger.warning("未能获取持仓数据")
                return False
                
            # 更新持仓信息
            for symbol, pos_data in self.bot.positions.items():
                if symbol in self.positions:
                    quantity = pos_data.get('position', 0)
                    avg_cost = pos_data.get('avg_cost', 0)
                    self.positions[symbol].update_from_ib(quantity, avg_cost)
                    
            self.last_verification_time = datetime.now()
            return True
            
        except Exception as e:
            self.logger.error(f"更新IB持仓信息时出错: {str(e)}")
            return False
            
    def verify_positions(self, force=False):
        """验证所有持仓信息"""
        now = datetime.now()
        if not force and self.last_verification_time and \
           (now - self.last_verification_time).seconds < self.verification_interval:
            return
            
        self.update_from_ib()
        
    def record_trade(self, symbol, action, quantity, price):
        """记录交易"""
        if symbol in self.positions:
            self.positions[symbol].record_trade(action, quantity, price)
            
    def get_position(self, symbol):
        """获取指定品种的持仓信息"""
        return self.positions.get(symbol)
        
    def get_all_positions(self):
        """获取所有持仓信息"""
        return {symbol: pos.get_position_info() for symbol, pos in self.positions.items()}
    
class ImprovedDonchianStrategy:
    def __init__(self, symbol, capital=100000, period=20, base_tranches=3, 
                 alert_threshold=0.004, max_capital_per_trade=50000):
        print("开始初始化策略...")  # 调试日志
        
        # 基础参数
        self.symbol = symbol
        self.capital = capital
        self.period = period
        self.base_tranches = base_tranches
        self.alert_threshold = alert_threshold
        self.max_capital_per_trade = max_capital_per_trade
        self.last_trading_day = None  # 初始化交易日变量
        
        print(f"基础参数设置完成 - 交易品种: {symbol}")  # 调试日志
        
        # 设置日志目录
        self.log_dir = r"F:\shares\twspy\trading_logs\trading_logs"
        self.csv_dir = r"F:\shares\twspy\trading_logs\trading_results"
        
        # 确保目录存在
        for directory in [self.log_dir, self.csv_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f"创建目录: {directory}")  # 调试日志
        
        # 初始化交易状态
        self.position = 0
        self.current_position_size = 0
        self.total_cost_basis = 0
        self.realized_pnl = 0
        self.last_trade_time = None
        self.min_trade_interval = 300
        self.trades = []
        self.daily_trades = []
        
        print("开始设置日志系统...")  # 调试日志
        
        # 设置基础日志配置
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # 初始化组件
        self.logger = self._setup_logger()
        print("日志系统设置完成")  # 调试日志
        
        self.csv_filename = self._setup_csv_file()
        print("CSV文件设置完成")  # 调试日志
        
        print("初始化交易接口...")  # 调试日志
        self.bot = TradingBot()
        print("交易接口初始化完成")  # 调试日志
        
        print("初始化持仓管理...")  # 调试日志
        self.position_manager = PositionManager(self.bot, self.logger)
        self.position_manager.initialize_position(symbol)
        print("持仓管理初始化完成")  # 调试日志
        
        print("策略初始化完成!")  # 调试日志
        
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
        self.position_manager.verify_positions()
        position = self.position_manager.get_position(self.symbol)
        return position.quantity if position else 0

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
    def update_position_info(self, action, filled_quantity, price):
        """更新持仓信息"""
        try:
            old_position = self.current_position_size
            old_cost_basis = self.total_cost_basis
            
            if action == "BUY":
                self.current_position_size += filled_quantity
                self.total_cost_basis += (filled_quantity * price)
            else:  # SELL
                if old_position > 0:
                    # 计算已实现盈亏
                    avg_cost = old_cost_basis / old_position
                    realized_pnl = filled_quantity * (price - avg_cost)
                    self.realized_pnl += realized_pnl
                    self.logger.info(f"交易实现盈亏: ${realized_pnl:.2f}")
                    
                self.current_position_size -= filled_quantity
                if self.current_position_size > 0:
                    self.total_cost_basis = (old_cost_basis / old_position) * self.current_position_size
                else:
                    self.total_cost_basis = 0
                    
            self.logger.info(f"持仓更新 - 数量: {self.current_position_size}, 总成本: ${self.total_cost_basis:.2f}")
            if self.current_position_size > 0:
                avg_cost = self.total_cost_basis / self.current_position_size
                self.logger.info(f"新平均成本: ${avg_cost:.2f}")
            
        except Exception as e:
            self.logger.error(f"更新持仓信息错误: {str(e)}")
            self.verify_position()
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
                # 提高阈值，更严格，避免震荡假突破
                adjusted_threshold = base_threshold * 1.2
                self.logger.info(f"通道较窄，提高阈值至: {adjusted_threshold*100:.3f}%，避免震荡假突破")
                
            elif width_ratio > 1.2:  # 当前通道比基准宽20%以上
                # 降低阈值，更宽松，提前捕捉机会
                adjusted_threshold = base_threshold * 0.8
                self.logger.info(f"通道较宽，降低阈值至: {adjusted_threshold*100:.3f}%，提前捕捉机会")
                
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
        #在测试环境中不检查交易间隔
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
        try:
            # [原有的初始化和数据获取代码保持不变，直到信号记录部分]...
            symbol = self.symbol            
            test_capital = self.capital     
            test_period = self.period       
            max_trade = self.max_capital_per_trade  
            
            stock = yf.Ticker(symbol)
            if start_date and end_date:
                hist = stock.history(start=start_date, end=end_date, interval='1m')
            else:
                hist = stock.history(period='4d', interval='1m')
                
            if hist.empty:
                self.logger.error("未能获取历史数据")
                return None
                    
            hist['upper_channel'] = hist['High'].rolling(window=test_period).max()
            hist['lower_channel'] = hist['Low'].rolling(window=test_period).min()
            baseline_width = self.calculate_baseline_channel_width(hist)
            
            # 记录基本信息
            self.logger.info(f"回测时间范围: {hist.index[0]} 至 {hist.index[-1]}")
            self.logger.info(f"数据点数量: {len(hist)}")
            self.logger.info(f"回测标的: {symbol}")
            self.logger.info(f"初始资金: ${test_capital}")
            self.logger.info(f"通道周期: {test_period}")
            self.logger.info(f"单次最大交易额: ${max_trade}")
            self.logger.info(f"基准通道宽度: {baseline_width*100:.2f}%")

            available_capital = test_capital
            total_trades = 0
            winning_trades = 0
            total_pnl = 0
            position_size = initial_position
            entry_price = 0
            trade_records = []
            all_signals = []
            
            for i in range(len(hist)):
                row = hist.iloc[i]
                price = row['Close']
                upper = row['upper_channel']
                lower = row['lower_channel']
                timestamp = hist.index[i]
                
                current_width = (upper - lower) / price
                upper_dist = abs(price - upper) / upper * 100
                lower_dist = abs(price - lower) / lower * 100
                width_ratio = current_width / baseline_width

                if width_ratio < 0.8:
                    channel_state = "窄通道"
                    adjusted_threshold = self.alert_threshold * 1.2
                elif width_ratio > 1.2:
                    channel_state = "宽通道"
                    adjusted_threshold = self.alert_threshold * 0.8
                else:
                    channel_state = "正常通道"
                    adjusted_threshold = self.alert_threshold

                lookback_data = hist.iloc[max(0, i-test_period):i+1]
                channel_position = self.is_near_channel(price, upper, lower, lookback_data)
                
                unrealized_pnl = (price - entry_price) * position_size if position_size > 0 else 0
                
                # 记录下轨信号
                if lower_dist <= (adjusted_threshold * 100):
                    signal = {
                        'time': timestamp,
                        'price': price,
                        'trigger_type': 'NEAR_LOWER',
                        'channel_state': channel_state,
                        'channel_width': f"{current_width*100:.2f}%",
                        'baseline_width': baseline_width * 100,
                        'upper_dist': upper_dist,
                        'lower_dist': lower_dist,
                        'position_size': position_size,
                        'upper_price': upper,
                        'lower_price': lower,
                        'available_capital': available_capital
                    }

                    if position_size == 0:
                        if channel_position == "LOWER":
                            shares_by_capital = int(available_capital / price)
                            shares_by_limit = int(max_trade / price)
                            max_shares = min(shares_by_capital, shares_by_limit)
                            
                            if max_shares >= 10:
                                position_size = max_shares
                                entry_price = price
                                available_capital -= (position_size * price)
                                total_trades += 1

                                signal.update({
                                    'action': 'BUY',
                                    'size': position_size,
                                    'reason': f'价格(${price:.2f})接近下轨(${lower:.2f})且无持仓，执行买入'
                                })
                                trade_records.append(signal)
                            else:
                                signal.update({
                                    'action': 'NO_ACTION',
                                    'reason': (f'虽然接近下轨(${lower:.2f})但资金条件不满足:\n'
                                            f'- 可用资金: ${available_capital:.2f}\n'
                                            f'- 资金允许买入: {shares_by_capital}股\n'
                                            f'- 限额允许买入: {shares_by_limit}股\n'
                                            f'- 最终可买: {max_shares}股 < 最小交易量10股\n'
                                            f'- 通道状态: {channel_state}\n'
                                            f'- 通道宽度比: {width_ratio:.2f}')
                                })
                        else:
                            signal.update({
                                'action': 'NO_ACTION',
                                'reason': (f'虽然价格(${price:.2f})接近下轨(${lower:.2f})但通道条件未触发:\n'
                                        f'- 通道状态: {channel_state}\n'
                                        f'- 通道宽度比: {width_ratio:.2f}\n'
                                        f'- 触发阈值: {adjusted_threshold:.4f}\n'
                                        f'- 实际距离: {lower_dist:.4f}%')
                            })
                    else:
                        signal.update({
                            'action': 'NO_ACTION',
                            'reason': (f'虽然接近下轨(${lower:.2f})但已有持仓:\n'
                                    f'- 当前持仓: {position_size}股\n'
                                    f'- 持仓均价: ${entry_price:.2f}\n'
                                    f'- 当前浮动盈亏: ${unrealized_pnl:.2f}\n'
                                    f'- 通道状态: {channel_state}\n'
                                    f'- 通道宽度比: {width_ratio:.2f}')
                        })
                        
                    all_signals.append(signal)

                # 记录上轨信号
                if upper_dist <= (adjusted_threshold * 100):
                    signal = {
                        'time': timestamp,
                        'price': price,
                        'trigger_type': 'NEAR_UPPER',
                        'channel_state': channel_state,
                        'channel_width': f"{current_width*100:.2f}%",
                        'baseline_width': baseline_width * 100,
                        'upper_dist': upper_dist,
                        'lower_dist': lower_dist,
                        'position_size': position_size,
                        'upper_price': upper,
                        'lower_price': lower,
                        'available_capital': available_capital
                    }

                    if position_size > 0:
                        if channel_position == "UPPER":
                            realized_pnl = (price - entry_price) * position_size
                            total_pnl += realized_pnl
                            available_capital += (position_size * price)
                            
                            if realized_pnl > 0:
                                winning_trades += 1
                            
                            signal.update({
                                'action': 'SELL',
                                'size': position_size,
                                'pnl': realized_pnl,
                                'reason': f'价格(${price:.2f})接近上轨(${upper:.2f})且有持仓{position_size}股，执行卖出'
                            })
                            trade_records.append(signal)
                            position_size = 0
                            entry_price = 0
                        else:
                            signal.update({
                                'action': 'NO_ACTION',
                                'reason': (f'虽然价格(${price:.2f})接近上轨(${upper:.2f})但通道条件未触发:\n'
                                        f'- 通道状态: {channel_state}\n'
                                        f'- 通道宽度比: {width_ratio:.2f}\n'
                                        f'- 触发阈值: {adjusted_threshold:.4f}\n'
                                        f'- 实际距离: {upper_dist:.4f}%')
                            })
                    else:
                        signal.update({
                            'action': 'NO_ACTION',
                            'reason': (f'虽然接近上轨(${upper:.2f})但无持仓:\n'
                                    f'- 通道状态: {channel_state}\n'
                                    f'- 通道宽度比: {width_ratio:.2f}\n'
                                    f'- 触发阈值: {adjusted_threshold:.4f}')
                        })
                        
                    all_signals.append(signal)

            # [结果计算和返回部分保持不变]...
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
                'trade_records': trade_records,
                'all_signals': all_signals
            }
            
        except Exception as e:
            self.logger.error(f"回测过程中发生错误: {str(e)}")
            raise
    # def backtest(self, start_date=None, end_date=None, initial_position=0):
    #     try:
    #         # 使用策略初始化时的参数
    #         symbol = self.symbol            
    #         test_capital = self.capital     
    #         test_period = self.period       
    #         max_trade = self.max_capital_per_trade  
            
    #         # 获取数据
    #         stock = yf.Ticker(symbol)
    #         if start_date and end_date:
    #             hist = stock.history(start=start_date, end=end_date, interval='1m')
    #         else:
    #             hist = stock.history(period='4d', interval='1m')
                
    #         if hist.empty:
    #             self.logger.error("未能获取历史数据")
    #             return None
                
    #         # 计算通道和基准宽度    
    #         hist['upper_channel'] = hist['High'].rolling(window=test_period).max()
    #         hist['lower_channel'] = hist['Low'].rolling(window=test_period).min()
    #         baseline_width = self.calculate_baseline_channel_width(hist)
            
    #         # 记录基本信息
    #         self.logger.info(f"回测时间范围: {hist.index[0]} 至 {hist.index[-1]}")
    #         self.logger.info(f"数据点数量: {len(hist)}")
    #         self.logger.info(f"回测标的: {symbol}")
    #         self.logger.info(f"初始资金: ${test_capital}")
    #         self.logger.info(f"通道周期: {test_period}")
    #         self.logger.info(f"单次最大交易额: ${max_trade}")
    #         self.logger.info(f"基准通道宽度: {baseline_width*100:.2f}%")

    #         # 初始化回测数据
    #         available_capital = test_capital
    #         total_trades = 0
    #         winning_trades = 0
    #         total_pnl = 0
    #         position = initial_position
    #         entry_price = 0
    #         trade_size = 0
    #         trade_records = []
            
    #         # 遍历数据
    #         for i in range(len(hist)):
    #             row = hist.iloc[i]
    #             price = row['Close']
    #             upper = row['upper_channel']
    #             lower = row['lower_channel']
    #             timestamp = hist.index[i]
                
    #             # 检查是否触及通道
    #             lookback_data = hist.iloc[max(0, i-test_period):i+1]
    #             channel_position = self.is_near_channel(price, upper, lower, lookback_data)
                
    #             if channel_position == "LOWER" and position == 0:
    #                 # 计算买入数量
    #                 max_shares = min(
    #                     int(available_capital / price),
    #                     int(max_trade / price)
    #                 )
    #                 trade_size = max_shares
                    
    #                 if trade_size > 0:
    #                     # 更新持仓
    #                     position = 1
    #                     entry_price = price
    #                     available_capital -= (trade_size * price)
    #                     total_trades += 1
                        
    #                     # 记录交易
    #                     trade = Trade("BUY", price, trade_size, timestamp, lower)
    #                     current_data = {
    #                         'current_price': price,
    #                         'upper_channel': upper,
    #                         'lower_channel': lower
    #                     }
    #                     self.save_trade_to_csv(trade, current_data)
                        
    #                     # 获取其他指标
    #                     upper_dist = abs(price - upper) / upper
    #                     lower_dist = abs(price - lower) / lower
    #                     adjusted_threshold = self.adjust_threshold(
    #                         (upper-lower)/price, baseline_width)
                        
    #                     trade_records.append({
    #                         'time': timestamp,
    #                         'action': 'BUY',
    #                         'price': price,
    #                         'size': trade_size,
    #                         'upper_price': upper,
    #                         'lower_price': lower,
    #                         'upper_dist': upper_dist * 100,
    #                         'lower_dist': lower_dist * 100,
    #                         'available_capital': available_capital,
    #                         'reason': f'价格(${price:.2f})接近下轨(${lower:.2f})',
    #                         'channel_width': f'{(upper-lower)/price*100:.2f}%',
    #                         'baseline_width': baseline_width * 100,
    #                         'adjusted_threshold': adjusted_threshold * 100,
    #                         'can_trade': self.can_trade()
    #                     })
                        
    #                     self.logger.info(
    #                         f"买入 - 时间: {timestamp}, "
    #                         f"数量: {trade_size}股, "
    #                         f"价格: ${price:.2f}, "
    #                         f"可用资金: ${available_capital:.2f}, "
    #                         f"原因: 价格接近下轨(${lower:.2f}), "
    #                         f"通道宽度: {(upper-lower)/price*100:.2f}%"
    #                     )
                        
    #             elif channel_position == "UPPER" and position == 1:
    #                 # 计算卖出收益
    #                 position = 0
    #                 pnl = (price - entry_price) * trade_size
    #                 total_pnl += pnl
    #                 available_capital += (trade_size * price)
                    
    #                 if pnl > 0:
    #                     winning_trades += 1
                    
    #                 # 记录交易    
    #                 trade = Trade("SELL", price, trade_size, timestamp, upper)
    #                 current_data = {
    #                     'current_price': price,
    #                     'upper_channel': upper,
    #                     'lower_channel': lower
    #                 }
    #                 self.save_trade_to_csv(trade, current_data)
                    
    #                 # 获取其他指标
    #                 upper_dist = abs(price - upper) / upper
    #                 lower_dist = abs(price - lower) / lower
    #                 adjusted_threshold = self.adjust_threshold(
    #                     (upper-lower)/price, baseline_width)

    #                 trade_records.append({
    #                     'time': timestamp,
    #                     'action': 'SELL',
    #                     'price': price,
    #                     'size': trade_size,
    #                     'pnl': pnl,
    #                     'upper_price': upper,
    #                     'lower_price': lower,
    #                     'upper_dist': upper_dist * 100,
    #                     'lower_dist': lower_dist * 100,
    #                     'available_capital': available_capital,
    #                     'reason': f'价格(${price:.2f})接近上轨(${upper:.2f})',
    #                     'channel_width': f'{(upper-lower)/price*100:.2f}%',
    #                     'baseline_width': baseline_width * 100,
    #                     'adjusted_threshold': adjusted_threshold * 100,
    #                     'can_trade': self.can_trade()
    #                 })
                    
    #                 self.logger.info(
    #                     f"卖出 - 时间: {timestamp}, "
    #                     f"数量: {trade_size}股, "
    #                     f"价格: ${price:.2f}, "
    #                     f"盈亏: ${pnl:.2f}, "
    #                     f"可用资金: ${available_capital:.2f}, "
    #                     f"原因: 价格接近上轨(${upper:.2f}), "
    #                     f"通道宽度: {(upper-lower)/price*100:.2f}%"
    #                 )
            
    #         # 计算回测结果
    #         win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            
    #         self.logger.info("\n=== 回测结果 ===")
    #         self.logger.info(f"初始资金: ${test_capital}")
    #         self.logger.info(f"最终资金: ${available_capital}")
    #         self.logger.info(f"总盈亏: ${total_pnl:.2f}")
    #         self.logger.info(f"总交易次数: {total_trades}")
    #         self.logger.info(f"盈利交易: {winning_trades}")
    #         self.logger.info(f"胜率: {win_rate:.2f}%")
            
    #         return {
    #             'total_trades': total_trades,
    #             'winning_trades': winning_trades,
    #             'win_rate': win_rate,
    #             'total_pnl': total_pnl,
    #             'final_capital': available_capital,
    #             'trade_records': trade_records
    #         }
            
    #     except Exception as e:
    #         self.logger.error(f"回测过程中发生错误: {str(e)}")
    #         raise
    
    def run_strategy(self):
        """运行策略"""
        try:
            self.logger.info("\n=== 策略启动 ===")
            self.logger.info(f"交易品种: {self.symbol}")
            self.logger.info(f"初始资金: ${self.capital:,}")
            
            # 连接到TWS
            self.bot.connect("127.0.0.1", 7497, clientId=random.randint(1, 10000))
            
            # 创建并启动API线程
            api_thread = threading.Thread(target=lambda: self.bot.run(), daemon=True)
            api_thread.start()
            time.sleep(1)
            
            # 更新持仓信息
            self.logger.info("请求持仓数据...")
            self.bot.reqPositions()
            
            # 等待持仓数据
            wait_count = 0
            while not self.bot.position_received and wait_count < 10:
                time.sleep(1)
                wait_count += 1
                self.logger.info(f"等待数据... {wait_count}/10")
            
            # 更新持仓信息
            position_info = self.bot.positions.get(self.symbol, {})
            if position_info:
                self.current_position_size = position_info.get('position', 0)
                avg_cost = position_info.get('avg_cost', 0)
                self.total_cost_basis = self.current_position_size * avg_cost
                self.logger.info(f"当前持仓: {self.current_position_size}")
                self.logger.info(f"平均成本: ${avg_cost:.2f}")
                self.logger.info(f"总成本: ${self.total_cost_basis:.2f}")
                
                # 请求并获取特定股票的盈亏信息
                if self.bot.account:
                    self.bot.reqStockPnL(self.symbol)
                    time.sleep(1)  # 等待数据返回
                    pnl_info = self.bot.getStockPnL(self.symbol)
                    self.logger.info(f"当日盈亏: ${pnl_info['daily_pnl']:.2f}")
                    self.logger.info(f"未实现盈亏: ${pnl_info['unrealized_pnl']:.2f}")
                    self.logger.info(f"已实现盈亏: ${pnl_info['realized_pnl']:.2f}")
            else:
                self.current_position_size = 0
                self.total_cost_basis = 0
                self.logger.info("无持仓")
            
            # 主循环
            while True:
                try:
                    now = datetime.now()
                    current_trading_day = now.date()
                    
                    if self.last_trading_day != current_trading_day:
                        if self.last_trading_day is not None:
                            self.log_daily_summary()
                        self.last_trading_day = current_trading_day
                        self.logger.info(f"\n=== 新交易日开始: {current_trading_day} ===")
                    
                    # 获取市场数据
                    df = self.get_market_data()
                    if df is not None and not df.empty:
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
                        
                        elif channel_position == "LOWER" and self.current_position_size == 0:
                            # 触及下轨，考虑开仓或加仓
                            self.logger.info("\n检测到下轨支撑信号 - 准备买入")
                            if self.execute_trade("BUY", current_price, upper_channel, lower_channel):
                                self.position = 1
                        
                        # 更新特定股票的PnL
                        if self.bot.account:
                            self.bot.reqStockPnL(self.symbol)
                            
                        # 收盘检查
                        if now.hour == 16 and now.minute == 0:
                            self.handle_market_close(current_price, upper_channel, lower_channel)
                    
                    time.sleep(5)
                    
                except Exception as e:
                    self.logger.error(f"策略运行错误: {str(e)}")
                    time.sleep(5)
                    continue
                    
        except Exception as e:
            self.logger.error(f"策略执行出错: {str(e)}")
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
            # 使用 getStockPnL 替代 getPnL
            pnl_info = self.bot.getStockPnL(self.symbol)
            self.logger.info(f"当日盈亏: ${pnl_info['daily_pnl']:.2f}")
            self.logger.info(f"未实现盈亏: ${pnl_info['unrealized_pnl']:.2f}")
            self.logger.info(f"已实现盈亏: ${pnl_info['realized_pnl']:.2f}")

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