from ib_insync import *
import pandas as pd
from tabulate import tabulate

def get_pnl():
    ib = IB()
    try:
        # 连接到TWS
        ib.connect('127.0.0.1', 7497, clientId=1)
        ib.sleep(1)
        
        # 获取账户
        account = ib.wrapper.accounts[0]
        
        # 获取当前持仓
        positions = ib.positions()
        pnls = {}
        
        # 调试信息：查看PnLSingle对象的属性
        for position in positions:
            contract = position.contract
            pnl = ib.reqPnLSingle(account, "", contract.conId)
            ib.sleep(1)
            
            if pnl:
                print(f"\nPnLSingle object for {contract.symbol}:")
                for attr in dir(pnl):
                    if not attr.startswith('_'):  # 只显示非私有属性
                        print(f"{attr}: {getattr(pnl, attr)}")
                
                # 调试信息：查看Position对象的属性
                print(f"\nPosition object for {contract.symbol}:")
                for attr in dir(position):
                    if not attr.startswith('_'):
                        print(f"{attr}: {getattr(position, attr)}")
                
                break  # 只打印第一个持仓的信息即可
                
    except Exception as e:
        print(f"发生错误: {str(e)}")
    finally:
        ib.disconnect()

if __name__ == '__main__':
    print("正在获取数据...")
    get_pnl()