from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
import time
import threading

class TWSPositionApp(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.positions = []
        self.done = False
    
    def position(self, account, contract, pos, avg_cost):
        position_data = {
            "账户": account,
            "合约": f"{contract.symbol} {contract.secType}",
            "持仓量": pos,
            "持仓成本": avg_cost
        }
        self.positions.append(position_data)
    
    def positionEnd(self):
        self.done = True

def main():
    app = TWSPositionApp()
    app.connect("127.0.0.1", 7497, 0)  # TWS端口一般为7497
    
    # 启动消息循环
    api_thread = threading.Thread(target=app.run, daemon=True)
    api_thread.start()
    
    time.sleep(1)  # 等待连接建立
    
    app.reqPositions()  # 请求持仓数据
    
    # 等待数据接收完成
    timeout = 10
    start_time = time.time()
    while not app.done and time.time() - start_time < timeout:
        time.sleep(0.1)
    
    # 打印持仓信息
    for pos in app.positions:
        print(f"账户: {pos['账户']}")
        print(f"合约: {pos['合约']}")
        print(f"持仓量: {pos['持仓量']}")
        print(f"持仓成本: {pos['持仓成本']}")
        print("-" * 50)
    
    app.disconnect()

if __name__ == "__main__":
    main()