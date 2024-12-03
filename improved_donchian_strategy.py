from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional, Dict, List, Any
import pandas as pd
from tabulate import tabulate  # 正确方式
import numpy as np
from ib_insync import *
import logging
import os
import json
import yfinance as yf
import random
from datetime import datetime, time
from zoneinfo import ZoneInfo  # Python 3.9+
@dataclass
class Trade:
    """交易记录"""
    action: str  # BUY or SELL
    price: float
    quantity: int
    timestamp: datetime
    channel_price: float

@dataclass
class Position:
    """持仓信息"""
    symbol: str
    quantity: int = 0
    avg_cost: float = 0
    total_cost: float = 0
    realized_pnl: float = 0
    unrealized_pnl: float = 0
    daily_pnl: float = 0  # 添加当日盈亏字段
    last_update: Optional[datetime] = None
    trades_today: List[Dict] = None
    
    def __post_init__(self):
        if self.trades_today is None:
            self.trades_today = []
            
    def update(self, quantity: int, avg_cost: float):
        """更新持仓信息"""
        self.quantity = quantity
        if quantity > 0:
            self.avg_cost = avg_cost
            self.total_cost = quantity * avg_cost
        else:
            self.avg_cost = 0
            self.total_cost = 0
        self.last_update = datetime.now()
        
    def update_market_price(self, current_price: float):
        """更新市场价格相关信息"""
        if self.quantity > 0:
            self.unrealized_pnl = self.quantity * (current_price - self.avg_cost)
        else:
            self.unrealized_pnl = 0
            
    def record_trade(self, action: str, quantity: int, price: float):
        """记录交易"""
        if action == "SELL" and self.quantity > 0:
            trade_pnl = quantity * (price - self.avg_cost)
            self.realized_pnl += trade_pnl
            self.quantity -= quantity
            if self.quantity <= 0:
                self.reset()
            else:
                self.total_cost = self.quantity * self.avg_cost
        elif action == "BUY":
            total_cost = self.total_cost + (quantity * price)
            self.quantity += quantity
            self.avg_cost = total_cost / self.quantity if self.quantity > 0 else 0
            self.total_cost = total_cost
            
        self.trades_today.append({
            'timestamp': datetime.now(),
            'action': action,
            'quantity': quantity,
            'price': price,
            'position_after': self.quantity,
            'avg_cost_after': self.avg_cost,
            'realized_pnl': self.realized_pnl
        })
        
    def reset(self):
        """重置持仓"""
        self.quantity = 0
        self.avg_cost = 0
        self.total_cost = 0
        
    def get_summary(self):
        """获取持仓摘要"""
        return {
            'symbol': self.symbol,
            'quantity': self.quantity,
            'avg_cost': self.avg_cost,
            'total_cost': self.total_cost,
            'realized_pnl': self.realized_pnl,
            'unrealized_pnl': self.unrealized_pnl,
            'last_update': self.last_update,
            'trades_today': len(self.trades_today)
        }
class AccountMonitor:
    def __init__(self, host='127.0.0.1', port=7497, client_id=None):
        self.host = host
        self.port = port
        self.client_id = client_id or random.randint(1, 10000)
        self.ib = IB()
        self.logger = self._setup_logger()
        
    def _setup_logger(self):
        logger = logging.getLogger('AccountMonitor')
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def connect(self):
        """连接到TWS"""
        try:
            if not self.ib.isConnected():
                self.logger.info("正在连接到交易系统...")
                self.ib.connect(self.host, self.port, clientId=self.client_id)
                self.ib.sleep(1)  # 等待连接建立
            return True
        except Exception as e:
            self.logger.error(f"连接失败: {str(e)}")
            return False

    def disconnect(self):
        """断开连接"""
        if self.ib.isConnected():
            self.ib.disconnect()
            self.logger.info("已断开连接")

    def get_position_info(self):
        """获取持仓信息"""
        self.logger.info("正在获取数据...")
        try:
            if not self.connect():
                return None

            # 获取账户
            account = self.ib.wrapper.accounts[0]
            
            # 请求账户整体的PnL
            account_pnl = self.ib.reqPnL(account)
            self.ib.sleep(1)
            
            # 获取当前持仓
            positions = self.ib.positions()
            if not positions:
                self.logger.info("没有持仓")
                return None
                
            pnls = {}
            
            # 获取每个持仓的PnL
            for position in positions:
                contract = position.contract
                pnl = self.ib.reqPnLSingle(account, "", contract.conId)
                self.ib.sleep(1)
                
                if pnl:
                    market_price = pnl.value / pnl.position if pnl.position != 0 else 0
                    pnls[contract.symbol] = {
                        'position': pnl.position,
                        'avgCost': position.avgCost,
                        'market_price': market_price,
                        'value': pnl.value,
                        'dailyPnL': pnl.dailyPnL,
                        'unrealizedPnL': pnl.unrealizedPnL,
                    }

            # 转换为DataFrame
            positions_data = [
                {
                    'Symbol': symbol,
                    '持仓': data['position'],
                    '成本价': round(data['avgCost'], 2),
                    '市价': round(data['market_price'], 2),
                    '持仓市值': round(data['value'], 2),
                    '未实现盈亏': round(data['unrealizedPnL'], 2),
                    '当日盈亏': round(data['dailyPnL'], 2)
                }
                for symbol, data in pnls.items()
            ]
            
            df = pd.DataFrame(positions_data)
            
            # 计算汇总数据
            summary = {
                'daily_total': sum(data['dailyPnL'] for data in pnls.values()),
                'unrealized_total': sum(data['unrealizedPnL'] for data in pnls.values()),
                'realized_pnl': account_pnl.realizedPnL if account_pnl else 0
            }
            
            return df, summary
            
        except Exception as e:
            self.logger.error(f"获取数据失败: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None, None
        finally:
            self.disconnect()

    def print_position_report(self):
        """打印持仓报告"""
        df, summary = self.get_position_info()
        if df is not None and not df.empty:
            print("\n当前持仓及盈亏状况：")
            print(tabulate(df, headers='keys', tablefmt='pretty', showindex=False,
                         floatfmt=('.0f', '.2f', '.2f', '.2f', '.2f', '.2f')))
            
            print("\n汇总信息：")
            print(f"当日总盈亏: ${summary['daily_total']:,.2f}")
            print(f"未实现盈亏: ${summary['unrealized_total']:,.2f}")
            print(f"已实现盈亏: ${summary['realized_pnl']:,.2f}")
            
        return df, summary
class Logger:
    """日志管理"""
    def __init__(self, name: str, log_dir: str):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        timestamp = datetime.now().strftime('%Y%m%d')
        log_file = os.path.join(log_dir, f'{name}_{timestamp}.log')
        
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        
    def info(self, msg: str):
        self.logger.info(msg)
        
    def error(self, msg: str):
        self.logger.error(msg)
        
    def warning(self, msg: str):
        self.logger.warning(msg)

class RiskControl:
    def __init__(self, max_position_value=1000000, max_daily_loss=-50000):
        self.max_position_value = max_position_value
        self.max_daily_loss = max_daily_loss
        self.daily_pnl = 0
        self.last_reset = datetime.now().date()
        
    def reset_daily_pnl(self):
        current_date = datetime.now().date()
        if current_date > self.last_reset:
            self.daily_pnl = 0
            self.last_reset = current_date
            
    def update_pnl(self, pnl_change):
        self.reset_daily_pnl()
        self.daily_pnl += pnl_change
        
    def can_trade(self, position_value, potential_loss=0):
        self.reset_daily_pnl()
        
        if position_value > self.max_position_value:
            return False, "超过最大持仓限制"
            
        if self.daily_pnl + potential_loss < self.max_daily_loss:
            return False, "触及每日止损限制"
            
        return True, ""
    
    
    
class PositionManager:
    def __init__(self, ib: IB, logger: Logger):
        self.ib = ib
        self.logger = logger
        self.positions: Dict[str, Position] = {}
        self.last_update = None
        self.update_interval = timedelta(minutes=5)
        
    def initialize_position(self, symbol: str):
        """初始化品种的持仓"""
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol)
            
    def update_positions(self):
        """更新所有持仓信息"""
        if (not self.last_update or 
            datetime.now() - self.last_update > self.update_interval):
            
            try:
                account = self.ib.managedAccounts()[0]
                portfolio = self.ib.portfolio(account=account)
                
                for item in portfolio:
                    symbol = item.contract.symbol
                    if symbol in self.positions:
                        self.positions[symbol].update(
                            quantity=int(item.position),
                            avg_cost=float(item.averageCost)
                        )
                        
                        # 更新未实现盈亏和当日盈亏
                        self.positions[symbol].unrealized_pnl = float(item.unrealizedPNL)
                        self.positions[symbol].daily_pnl = float(item.dailyPnL) if hasattr(item, 'dailyPnL') else 0
                        
                self.last_update = datetime.now()
                
            except Exception as e:
                self.logger.error(f"更新持仓信息错误: {str(e)}")
            
    def record_trade(self, symbol: str, action: str, quantity: int, price: float):
        """记录交易"""
        if symbol in self.positions:
            self.positions[symbol].record_trade(action, quantity, price)
            
    def get_position(self, symbol: str) -> Optional[Position]:
        """获取指定品种的持仓"""
        return self.positions.get(symbol)
        
    def get_all_positions(self) -> Dict[str, dict]:
        """获取所有持仓信息"""
        return {symbol: pos.get_summary() for symbol, pos in self.positions.items()}
        
    def export_daily_summary(self, csv_dir: str):
        """导出每日交易摘要"""
        timestamp = datetime.now().strftime('%Y%m%d')
        
        for symbol, position in self.positions.items():
            summary_file = os.path.join(csv_dir, f'daily_summary_{symbol}_{timestamp}.csv')
            
            summary = {
                'Date': datetime.now().strftime('%Y-%m-%d'),
                'Symbol': symbol,
                'Final_Position': position.quantity,
                'Avg_Cost': position.avg_cost,
                'Realized_PnL': position.realized_pnl,
                'Unrealized_PnL': position.unrealized_pnl,
                'Total_Trades': len(position.trades_today)
            }
            
            pd.DataFrame([summary]).to_csv(
                summary_file,
                mode='a',
                header=not os.path.exists(summary_file),
                index=False
            )
            
        # 清空每日交易记录
        for position in self.positions.values():
            position.trades_today = []
            
            
class DonchianStrategy:
    def __init__(self, 
             symbol: str, 
             period: int = 20,
             capital: float = 100000,
             alert_threshold: float = 0.004,
             max_capital_per_trade: float = 50000,
             log_dir: str = "trading_logs",
             csv_dir: str = "trading_results"):
    
        # 设置日志
        logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(f'Donchian_{symbol}')
        
        self.logger.info("开始初始化策略...")
        self.account_monitor = AccountMonitor()
        # 基础参数
        self.symbol = symbol
        self.period = period
        self.capital = capital
        self.alert_threshold = alert_threshold
        self.max_capital_per_trade = max_capital_per_trade
        self.log_dir = log_dir
        self.csv_dir = csv_dir

        # 确保输出目录存在
        for directory in [log_dir, csv_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)

        try:
            # 初始化IB连接
            self.ib = IB()
            self.ib.connect('127.0.0.1', 7497, clientId=random.randint(1, 1000))
            self.logger.info("TWS连接成功")

            # 设置合约
            self.contract = Stock(symbol, 'SMART', 'USD')
            self.ib.qualifyContracts(self.contract)
            self.logger.info(f"合约信息: {self.contract}")

            # 获取账户
            self.account = self.ib.managedAccounts()[0]
            self.logger.info(f"使用账户: {self.account}")

            # 初始化持仓管理
            self.position_manager = PositionManager(self.ib, self.logger)
            self.position_manager.initialize_position(symbol)

            # 风险控制
            self.risk_control = RiskControl()
            
            # 交易控制
            self.last_trade_time = None
            self.min_trade_interval = timedelta(minutes=5)
            self.last_trading_day = None

            # 同步初始持仓
            portfolio = self.ib.portfolio(account=self.account)
            for item in portfolio:
                if item.contract.symbol == symbol:
                    self.logger.info(f"当前持仓: {item.position}股, 均价: ${item.averageCost:.2f}")
                    break

            self.logger.info("策略初始化完成")

        except Exception as e:
            self.logger.error(f"初始化错误: {str(e)}")
            if hasattr(self, 'ib') and self.ib.isConnected():
                self.ib.disconnect()
            raise
        
        # 确保输出目录存在
        for directory in [log_dir, csv_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)

    def calculate_channels(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算唐奇安通道"""
        df['upper_channel'] = df['High'].rolling(window=self.period).max()
        df['lower_channel'] = df['Low'].rolling(window=self.period).min()
        df['middle_channel'] = (df['upper_channel'] + df['lower_channel']) / 2
        df['channel_width'] = (df['upper_channel'] - df['lower_channel']) / df['Close']
        return df
    def calculate_baseline_width(self, df: pd.DataFrame) -> float:
        """计算基准通道宽度"""
        lookback = self.period * 3
        recent_widths = df['channel_width'].tail(lookback)
        baseline = recent_widths.mean()
        self.logger.info(f"基准通道宽度: {baseline*100:.2f}%")
        return baseline
        
    def adjust_threshold(self, current_width: float, baseline_width: float) -> float:
        """调整阈值"""
        width_ratio = current_width / baseline_width
        if width_ratio < 0.8:
            threshold = self.alert_threshold * 1.2
            self.logger.info(f"通道较窄，提高阈值至: {threshold*100:.3f}%")
        elif width_ratio > 1.2:
            threshold = self.alert_threshold * 0.8
            self.logger.info(f"通道较宽，降低阈值至: {threshold*100:.3f}%")
        else:
            threshold = self.alert_threshold
            self.logger.info(f"通道正常，使用基础阈值: {threshold*100:.3f}%")
        return threshold
        
    def check_channel_position(self, price: float, upper: float, lower: float, 
                            df: pd.DataFrame) -> Optional[str]:
        """检查价格相对于通道的位置"""
        current_width = (upper - lower) / price
        baseline_width = self.calculate_baseline_width(df)
        threshold = self.adjust_threshold(current_width, baseline_width)
        
        upper_dist = abs(price - upper) / upper
        lower_dist = abs(price - lower) / lower
        
        self.logger.info("\n=== 通道分析 ===")
        self.logger.info(f"当前通道宽度: {current_width*100:.2f}%")
        self.logger.info(f"调整后阈值: {threshold*100:.3f}%")
        self.logger.info(f"与上轨距离: {upper_dist*100:.3f}%")
        self.logger.info(f"与下轨距离: {lower_dist*100:.3f}%")
        
        if upper_dist <= threshold:
            return "UPPER"
        elif lower_dist <= threshold:
            return "LOWER"
        return None

    def calculate_position_size(self, price: float) -> int:
        """计算交易数量"""
        max_shares = int(min(self.capital, self.max_capital_per_trade) / price)
        return max(10, max_shares)

    def can_trade(self) -> bool:
        """检查是否可以交易"""
        if not self.last_trade_time:
            return True
        return datetime.now() - self.last_trade_time >= self.min_trade_interval

    def execute_trade(self, action: str, price: float, channel_price: float) -> bool:
        """执行交易"""
        if not self.can_trade():
            return False
            
        position_size = self.calculate_position_size(price)
        if position_size < 10:
            self.logger.info("交易量过小，取消交易")
            return False
            
        # 风险检查
        position_value = position_size * price
        can_trade, reason = self.risk_control.can_trade(position_value)
        if not can_trade:
            self.logger.warning(f"风险控制阻止交易: {reason}")
            return False
        
        order = MarketOrder(action, position_size)
        trade = self.ib.placeOrder(self.contract, order)
        
        # 等待订单完成
        while not trade.isDone():
            self.ib.sleep(1)
            
        if trade.orderStatus.status == 'Filled':
            filled = trade.orderStatus.filled
            fill_price = trade.orderStatus.avgFillPrice
            
            self.position_manager.record_trade(
                self.symbol, action, filled, fill_price)
            self.last_trade_time = datetime.now()
            
            self.logger.info(f"交易执行成功: {action} {filled}股 @ ${fill_price:.2f}")
            return True
            
        return False
    def should_close_position(self, now: datetime) -> bool:
        """判断是否应该收盘平仓"""
        # 获取当前是否是夏令时
        is_dst = self._is_dst(now)
        
        # 设定收盘前的安全时间窗口
        if is_dst:
            CLOSING_START = time(3, 55)  # 夏令时 北京时间 03:55
            CLOSING_END = time(4, 0)     # 夏令时 北京时间 04:00
        else:
            CLOSING_START = time(4, 55)  # 冬令时 北京时间 04:55
            CLOSING_END = time(5, 0)     # 冬令时 北京时间 05:00
        
        current_time = now.time()
        return CLOSING_START <= current_time <= CLOSING_END

    def _is_dst(self, now: datetime) -> bool:
        """判断当前是否是夏令时"""
        # 美国夏令时：3月第二个周日开始，11月第一个周日结束
        year = now.year
        
        # 3月第二个周日
        dst_start = datetime(year, 3, 1)
        while dst_start.weekday() != 6:  # 找到第一个周日
            dst_start = dst_start + timedelta(days=1)
        dst_start = dst_start + timedelta(days=7)  # 第二个周日
        
        # 11月第一个周日
        dst_end = datetime(year, 11, 1)
        while dst_end.weekday() != 6:  # 找到第一个周日
            dst_end = dst_end + timedelta(days=1)
        
        return dst_start <= now < dst_end
    def log_status(self):
        """记录当前状态"""
        position = self.position_manager.get_position(self.symbol)
        if not position:
            return
            
        self.logger.info(f"\n=== {datetime.now().strftime('%H:%M:%S')} 策略状态 ===")
        self.logger.info(f"持仓数量: {position.quantity}")
        self.logger.info(f"平均成本: ${position.avg_cost:.2f}")
        self.logger.info(f"当日盈亏: ${position.daily_pnl:.2f}")  # 添加当日盈亏显示
        self.logger.info(f"未实现盈亏: ${position.unrealized_pnl:.2f}")
        self.logger.info(f"已实现盈亏: ${position.realized_pnl:.2f}")
    
    def _create_signal_record(self, price, upper, lower, channel_pos, position_size, available_capital):
        """创建信号记录"""
        return {
            'time': datetime.now(),
            'price': price,
            'upper': upper,
            'lower': lower,
            'channel_position': channel_pos,
            'position_size': position_size,
            'available_capital': available_capital,
            'width_ratio': (upper - lower) / price
        }

    def _execute_backtest_sell(self, price, position_size, entry_price, timestamp):
        """回测卖出"""
        pnl = position_size * (price - entry_price)
        return {
            'time': timestamp,
            'action': 'SELL',
            'price': price,
            'quantity': position_size,
            'value': position_size * price,
            'pnl': pnl
        }

    def _execute_backtest_buy(self, price, available_capital, timestamp):
        """回测买入"""
        max_shares = int(min(available_capital, self.max_capital_per_trade) / price)
        if max_shares < 10:
            return None
            
        return {
            'time': timestamp,
            'action': 'BUY',
            'price': price,
            'quantity': max_shares,
            'value': max_shares * price
        }

    def _generate_backtest_results(self, initial_capital, final_capital, total_trades, 
                                winning_trades, total_pnl, trade_records, signals):
        """生成回测结果"""
        return {
            'initial_capital': initial_capital,
            'final_capital': final_capital,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'win_rate': (winning_trades / total_trades * 100) if total_trades > 0 else 0,
            'total_pnl': total_pnl,
            'trade_records': trade_records,
            'all_signals': signals
        }

    def _export_backtest_results(self, results):
        """导出回测结果"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 保存交易记录
        trades_df = pd.DataFrame(results['trade_records'])
        trades_df.to_csv(f'{self.csv_dir}/backtest_trades_{timestamp}.csv', index=False)
        
        # 保存信号记录
        signals_df = pd.DataFrame(results['all_signals'])
        signals_df.to_csv(f'{self.csv_dir}/backtest_signals_{timestamp}.csv', index=False)
        
        # 保存汇总
        summary = {
            'Initial Capital': results['initial_capital'],
            'Final Capital': results['final_capital'],
            'Total PnL': results['total_pnl'],
            'Total Trades': results['total_trades'],
            'Winning Trades': results['winning_trades'],
            'Win Rate': results['win_rate']
        }
        
        with open(f'{self.csv_dir}/backtest_summary_{timestamp}.json', 'w') as f:
            json.dump(summary, f, indent=4)
            #used yfinance get data==================================================
    # def run_strategy(self):
    #     """运行策略"""
    #     try:
    #         self.logger.info("\n=== 策略启动 ===")
    #         self.logger.info(f"交易品种: {self.symbol}")
            
    #         # 确保TWS连接
    #         if not self.ib.isConnected():
    #             self.ib.connect('127.0.0.1', 7497, clientId=random.randint(1, 1000))
                
    #         while True:
    #             try:
    #                 now = datetime.now()
    #                 current_trading_day = now.date()
                    
    #                 # 每日重置
    #                 if self.last_trading_day != current_trading_day:
    #                     if self.last_trading_day is not None:
    #                         self.position_manager.export_daily_summary(self.csv_dir)
    #                     self.last_trading_day = current_trading_day
    #                     self.logger.info(f"\n=== 新交易日: {current_trading_day} ===")
                    
    #                 # 从yfinance获取数据
    #                 stock = yf.Ticker(self.symbol)
    #                 df = stock.history(period='2d', interval='1m')
                    
    #                 if not df.empty:
    #                     # 计算通道
    #                     df = self.calculate_channels(df)
    #                     latest = df.iloc[-1]
                        
    #                     price = latest['Close']
    #                     upper = latest['upper_channel']
    #                     lower = latest['lower_channel']
                        
    #                     # 更新持仓
    #                     self.position_manager.update_positions()
    #                     position = self.position_manager.get_position(self.symbol)
                        
    #                     # 检查收盘时间
    #                     if self.should_close_position(now):
    #                         self.logger.info("市场即将收盘，开始平仓操作...")
    #                         self.handle_market_close()
    #                         continue
                        
    #                     # 检查通道位置
    #                     channel_pos = self.check_channel_position(price, upper, lower, df)
                        
    #                     # 记录状态
    #                     self.log_status()
                        
    #                     # 交易逻辑
    #                     if channel_pos == "UPPER" and position and position.quantity > 0:
    #                         self.execute_trade("SELL", price, upper)
    #                     elif channel_pos == "LOWER" and (not position or position.quantity == 0):
    #                         self.execute_trade("BUY", price, lower)
                    
    #                 time.sleep(5)  # 可以缩短等待时间提高频率
                    
    #             except Exception as e:
    #                 self.logger.error(f"策略运行错误: {str(e)}")
    #                 time.sleep(2)
                    
    #     except Exception as e:
    #         self.logger.error(f"策略执行错误: {str(e)}")
    #     finally:
    #         if self.ib.isConnected():
    #             self.ib.disconnect()

     #used yfinance get data==================================================
    def run_strategy(self):
        """运行策略"""
        try:
            # 策略启动时打印持仓状况
            self.logger.info("正在获取账户信息...")
            self.account_monitor.print_position_report()
            
            self.logger.info(f"\n=== 策略启动 - {self.symbol} ===")
            self.position_manager.update_positions()
            
            while True:
                try:
                    now = datetime.now()
                    current_trading_day = now.date()
                    
                    # 处理新交易日
                    if self.last_trading_day != current_trading_day:
                        if self.last_trading_day is not None:
                            self.position_manager.export_daily_summary(self.csv_dir)
                        self.last_trading_day = current_trading_day
                        self.logger.info(f"\n=== 新交易日: {current_trading_day} ===")
                        # 新交易日开始时打印持仓状况
                        self.logger.info("正在获取账户信息...")
                        self.account_monitor.print_position_report()
                    
                    # 获取市场数据
                    self.logger.info("正在获取市场数据...")


                    bars = self.ib.reqHistoricalData(
                        self.contract,
                        endDateTime='',
                        durationStr='2 D',
                        barSizeSetting='1 min',
                        whatToShow='TRADES',
                        useRTH=True
                    )
                    
                    if bars:
                        # 转换数据
                        df = util.df(bars)
                        self.logger.info(f"获取数据点数: {len(df)}")
                        self.logger.info(f"第一个时间点: {df.index[0]}")
                        self.logger.info(f"最后一个时间点: {df.index[-1]}")
                        self.logger.info(f"数据间隔: {df.index[1] - df.index[0]}")
                        # 重命名列
                        df = df.rename(columns={
                            'high': 'High',
                            'low': 'Low',
                            'close': 'Close',
                            'open': 'Open',
                            'volume': 'Volume'
                        })
                        
                        df = self.calculate_channels(df)
                        latest = df.iloc[-1]
                        
                        price = latest['Close']
                        upper = latest['upper_channel']
                        lower = latest['lower_channel']
                        
                        # 更新持仓
                        self.position_manager.update_positions()
                        position = self.position_manager.get_position(self.symbol)
                        
                        # 检查是否到了收盘时间
                        if self.should_close_position(now):
                            self.logger.info("市场即将收盘，开始平仓操作...")
                            self.handle_market_close()
                            continue  # 收盘时不执行其他交易
                        
                        # 检查通道位置
                        channel_pos = self.check_channel_position(price, upper, lower, df)
                        
                        # 记录状态
                        self.log_status()
                        
                        # 交易逻辑
                        trade_executed = False
                        if channel_pos == "UPPER" and position and position.quantity > 0:
                            trade_executed = self.execute_trade("SELL", price, upper)
                        elif channel_pos == "LOWER" and (not position or position.quantity == 0):
                            trade_executed = self.execute_trade("BUY", price, lower)
                        
                        # 如果有交易执行，打印最新持仓状况
                        if trade_executed:
                            self.logger.info("交易执行后，正在获取最新账户信息...")
                            self.account_monitor.print_position_report()
                    
                    self.ib.sleep(5)
                    
                except Exception as e:
                    self.logger.error(f"策略运行错误: {str(e)}")
                    self.ib.sleep(2)
                    
        except Exception as e:
            self.logger.error(f"策略执行错误: {str(e)}")
        finally:
            if self.ib.isConnected():
                self.ib.disconnect()

    def handle_market_close(self):
        """处理收盘"""
        self.position_manager.export_daily_summary(self.csv_dir)
        position = self.position_manager.get_position(self.symbol)
        if position and position.quantity != 0:  # 只要有仓位就处理
            self.logger.info("收盘前平仓")
            price = self.ib.reqMktData(self.contract).last
            if price:
                if position.quantity > 0:
                    self.execute_trade("SELL", price, 0)
                else:  # quantity < 0, 说明是空仓
                    self.execute_trade("BUY", price, 0)

    def backtest(self, start_date=None, end_date=None, initial_position=0):
        """回测策略"""
        try:
            symbol = self.symbol
            test_capital = self.capital
            test_period = self.period
            
            # 获取历史数据
            stock = yf.Ticker(symbol)
            if start_date and end_date:
                hist = stock.history(start=start_date, end=end_date, interval='1m')
            else:
                hist = stock.history(period='4d', interval='1m')
                
            if hist.empty:
                self.logger.error("未能获取历史数据")
                return None
                
            # 计算通道
            hist = self.calculate_channels(hist)
            baseline_width = self.calculate_baseline_width(hist)
            
            # 初始化回测变量
            available_capital = test_capital
            total_trades = 0
            winning_trades = 0
            total_pnl = 0
            position_size = initial_position
            entry_price = 0
            trade_records = []
            all_signals = []
            
            # 回测循环
            for i in range(len(hist)):
                row = hist.iloc[i]
                self._process_backtest_bar(
                    row, hist.iloc[:i+1],
                    available_capital, position_size, entry_price,
                    total_trades, winning_trades, total_pnl,
                    trade_records, all_signals
                )
                
            results = self._generate_backtest_results(
                test_capital, available_capital,
                total_trades, winning_trades, total_pnl,
                trade_records, all_signals
            )
            
            self._export_backtest_results(results)
            return results
            
        except Exception as e:
            self.logger.error(f"回测错误: {str(e)}")
            raise

    def _process_backtest_bar(self, row, hist_slice, available_capital, position_size, 
                            entry_price, total_trades, winning_trades, total_pnl,
                            trade_records, all_signals):
        """处理回测数据条"""
        price = row['Close']
        upper = row['upper_channel']
        lower = row['lower_channel']
        timestamp = row.name
        
        channel_pos = self.check_channel_position(price, upper, lower, hist_slice)
        signal = self._create_signal_record(price, upper, lower, channel_pos, 
                                        position_size, available_capital)
        
        if channel_pos == "UPPER" and position_size > 0:
            trade_result = self._execute_backtest_sell(
                price, position_size, entry_price, timestamp)
            if trade_result:
                position_size = 0
                available_capital += trade_result['value']
                total_pnl += trade_result['pnl']
                if trade_result['pnl'] > 0:
                    winning_trades += 1
                trade_records.append(trade_result)
                
        elif channel_pos == "LOWER" and position_size == 0:
            trade_result = self._execute_backtest_buy(
                price, available_capital, timestamp)
            if trade_result:
                position_size = trade_result['quantity']
                entry_price = price
                available_capital -= trade_result['value']
                total_trades += 1
                trade_records.append(trade_result)
        
        all_signals.append(signal)

    def optimize_parameters(self, param_ranges, start_date, end_date):
        """优化策略参数"""
        results = []
        
        for period in range(*param_ranges['period']):
            for threshold in np.arange(*param_ranges['threshold']):
                self.period = period
                self.alert_threshold = threshold
                
                backtest_result = self.backtest(start_date, end_date)
                if backtest_result:
                    results.append({
                        'period': period,
                        'threshold': threshold,
                        'pnl': backtest_result['total_pnl'],
                        'win_rate': backtest_result['win_rate'],
                        'trades': backtest_result['total_trades']
                    })
                    
        results_df = pd.DataFrame(results)
        best_pnl = results_df.loc[results_df['pnl'].idxmax()]
        best_win_rate = results_df.loc[results_df['win_rate'].idxmax()]
        
        return {
            'all_results': results_df,
            'best_pnl': best_pnl.to_dict(),
            'best_win_rate': best_win_rate.to_dict()
        }
        
def main():
    config = {
        'symbol': 'aapl',
        'period': 20,
        'capital': 100000,
        'alert_threshold': 0.004,
        'max_capital_per_trade': 50000,
        'log_dir': 'trading_logs',
        'csv_dir': 'trading_results'
    }
    
    strategy = None
    try:
        strategy = DonchianStrategy(**config)
        print("策略初始化完成，开始运行...")
        strategy.run_strategy()
        
    except KeyboardInterrupt:
        print("\n收到停止信号，正在关闭...")
    except Exception as e:
        print(f"\n发生错误: {str(e)}")
    finally:
        if strategy and hasattr(strategy, 'ib') and strategy.ib.isConnected():
            strategy.ib.disconnect()
            print("已断开TWS连接")

if __name__ == "__main__":
    main()