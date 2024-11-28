import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta
import yfinance as yf
from improved_donchian_strategy import ImprovedDonchianStrategy
import pandas as pd

def plot_backtest_results():
   st.title('唐奇安通道策略回测可视化')

   # 初始化策略
   strategy = ImprovedDonchianStrategy(
       symbol='TSLA',
       capital=100000,
       period=20,
       base_tranches=3,
       alert_threshold=0.002,
       max_capital_per_trade=30000
   )

   # 获取回测时间范围
   end_date = datetime.now()
   start_date = end_date - timedelta(days=2)

   # 获取数据
   df = yf.download(strategy.symbol, start=start_date, end=end_date, interval='1m')
   df = strategy.get_donchian_channels(df)

   # 执行回测
   result = strategy.backtest(
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d')
    )

   fig = go.Figure()

   # 添加K线
   fig.add_trace(go.Candlestick(
       x=df.index,
       open=df['Open'],
       high=df['High'],
       low=df['Low'],
       close=df['Close'],
       name='K线'
   ))

   # 添加唐奇安通道
   fig.add_trace(go.Scatter(
       x=df.index,
       y=df['upper_channel'],
       name='上轨',
       line=dict(color='blue', dash='dash')
   ))

   fig.add_trace(go.Scatter(
       x=df.index,
       y=df['lower_channel'],
       name='下轨',
       line=dict(color='blue', dash='dash'),
       fill='tonexty'
   ))

   # 添加交易点和悬停信息
   for record in result['trade_records']:
       # 解析通道宽度百分比
       width_str = record['channel_width']
       width_value = float(width_str.strip('%'))
       
       # 判断通道宽窄状态
       if width_value < record.get('baseline_width', 0) * 0.8:
           channel_state = "窄通道"
       elif width_value > record.get('baseline_width', 0) * 1.2:
           channel_state = "宽通道"
       else:
           channel_state = "正常通道"

       hover_text = (
           f"时间: {record['time']}<br>"
           f"操作: {record['action']}<br>"
           f"价格: ${record['price']:.2f}<br>"
           f"数量: {record['size']}股<br>"
           f"原因: {record['reason']}<br>"
           f"通道状态: {channel_state}<br>"
           f"当前通道宽度: {width_str}<br>"
           f"基准通道宽度: {record.get('baseline_width', 0):.2f}%"
       )
       
       if 'pnl' in record:
           hover_text += f"<br>盈亏: ${record['pnl']:.2f}"

       if record['action'] == 'BUY':
           fig.add_trace(go.Scatter(
               x=[record['time']],
               y=[record['price']],
               mode='markers',
               marker=dict(
                   symbol='triangle-up',
                   size=15,
                   color='red'
               ),
               name='买入',
               text=hover_text,
               hoverinfo='text'
           ))
       else:
           fig.add_trace(go.Scatter(
               x=[record['time']],
               y=[record['price']],
               mode='markers',
               marker=dict(
                   symbol='triangle-down',
                   size=15,
                   color='green'
               ),
               name='卖出',
               text=hover_text,
               hoverinfo='text'
           ))

   # 更新布局
   fig.update_layout(
       title=f'{strategy.symbol} 回测结果',
       xaxis_title='时间',
       yaxis_title='价格',
       height=800
   )

   # 显示图表
   st.plotly_chart(fig, use_container_width=True)

   # 显示回测统计
   st.subheader('回测统计')
   col1, col2, col3 = st.columns(3)
   col1.metric('总交易次数', result['total_trades'])
   col2.metric('盈利交易', result['winning_trades'])
   col3.metric('胜率', f"{result['win_rate']:.2f}%")
   
   st.metric('总盈亏', f"${result['total_pnl']:.2f}")

   # 在页面下方添加交易详情表格
   st.subheader('交易详情')
   trades_df = pd.DataFrame([
   {
       '时间': record['time'],
       '操作': record['action'], 
       '价格': f"${record['price']:.2f}",
       '上轨价格': f"${record['upper_price']:.2f}",
       '下轨价格': f"${record['lower_price']:.2f}",
       '与上轨距离': f"{record['upper_dist']:.3f}%",
       '与下轨距离': f"{record['lower_dist']:.3f}%",
       '数量': record['size'],
       '原因': record['reason'],
       '通道状态': "窄通道" if float(record['channel_width'].strip('%')) < record.get('baseline_width', 0) * 0.8 
                  else ("宽通道" if float(record['channel_width'].strip('%')) > record.get('baseline_width', 0) * 1.2 
                        else "正常通道"),
       '当前通道宽度': record['channel_width'],
       '基准通道宽度': f"{record.get('baseline_width', 0):.2f}%",
       '调整后阈值': f"{record['adjusted_threshold']:.3f}%",
       '可交易状态': record['can_trade'],
       '盈亏': f"${record.get('pnl', 0):.2f}" if 'pnl' in record else '-'
   }
   for record in result['trade_records']
])
   
   st.dataframe(trades_df)

if __name__ == '__main__':
   plot_backtest_results()