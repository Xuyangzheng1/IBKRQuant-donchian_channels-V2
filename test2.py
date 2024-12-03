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
        
        # 请求账户整体的PnL
        account_pnl = ib.reqPnL(account)
        ib.sleep(1)  # 等待数据返回
        
        # 获取当前持仓
        positions = ib.positions()
        pnls = {}
        
        # 获取每个持仓的PnL
        for position in positions:
            contract = position.contract
            pnl = ib.reqPnLSingle(account, "", contract.conId)
            ib.sleep(1)
            
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
        
        if pnls:
            positions_data = []
            for symbol, data in pnls.items():
                positions_data.append({
                    'Symbol': symbol,
                    '持仓': data['position'],
                    '成本价': round(data['avgCost'], 2),
                    '市价': round(data['market_price'], 2),
                    '持仓市值': round(data['value'], 2),
                    '未实现盈亏': round(data['unrealizedPnL'], 2),
                    '当日盈亏': round(data['dailyPnL'], 2)
                })
                
            df = pd.DataFrame(positions_data)
            
            # 格式化显示
            print("\n当前持仓及盈亏状况：")
            print(tabulate(df, headers='keys', tablefmt='pretty', showindex=False,
                         floatfmt=('.0f', '.2f', '.2f', '.2f', '.2f', '.2f')))
            
            # 计算并显示汇总信息
            daily_total = sum(data['dailyPnL'] for data in pnls.values())
            unrealized_total = sum(data['unrealizedPnL'] for data in pnls.values())
            
            print(f"\n汇总信息：")
            print(f"当日总盈亏: ${daily_total:,.2f}")
            print(f"未实现盈亏: ${unrealized_total:,.2f}")
            if account_pnl:
                print(f"已实现盈亏: ${account_pnl.realizedPnL:,.2f}")
            
            return df
            
        else:
            print("没有找到持仓信息")
            return None
            
    except Exception as e:
        print(f"发生错误: {str(e)}")
        import traceback
        print(traceback.format_exc())
    finally:
        ib.disconnect()

if __name__ == '__main__':
    print("正在获取数据...")
    df = get_pnl()