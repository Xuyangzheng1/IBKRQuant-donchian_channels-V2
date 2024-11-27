# mock_tws.py
from datetime import datetime
import time

class MockContract:
    def __init__(self, symbol):
        self.symbol = symbol
        self.secType = "STK"
        self.exchange = "SMART"
        self.currency = "USD"
        self.primaryExchange = "NASDAQ"

class MockOrder:
    def __init__(self, action, totalQuantity):
        self.action = action
        self.orderType = "MKT"
        self.totalQuantity = totalQuantity
        self.eTradeOnly = False
        self.firmQuoteOnly = False

class MockTWS(object):
    def __init__(self):
        self.positions = {'TSLA': 0}  # 初始化仓位字典
        self.current_price = 100.0
        self.order_id = 0
        self.market_depth = {
            1: {
                'bids': [{'price': 99.5, 'size': 100}],
                'asks': [{'price': 100.5, 'size': 100}]
            }
        }
        self.orders = []
        self.connected = False

    def connect(self, host, port, client_id):
        self.connected = True
        self.order_id = 1000  # 设置初始订单ID
        return True

    def disconnect(self):
        self.connected = False

    def run(self):
        pass

    def placeOrder(self, orderId, contract, order):
        # 创建订单记录
        order_record = {
            'orderId': orderId,
            'symbol': contract.symbol,
            'action': order.action,
            'quantity': order.totalQuantity,
            'time': datetime.now()
        }
        
        # 更新仓位
        if order.action == "BUY":
            self.positions[contract.symbol] = self.positions.get(contract.symbol, 0) + order.totalQuantity
        elif order.action == "SELL":
            self.positions[contract.symbol] = self.positions.get(contract.symbol, 0) - order.totalQuantity
        
        self.orders.append(order_record)
        self.order_id = orderId + 1
        return True

    def reqMktDepth(self, reqId, contract, numRows):
        return self.market_depth.get(reqId, None)

    def position(self, account, contract, pos, avgCost):
        self.positions[contract.symbol] = pos