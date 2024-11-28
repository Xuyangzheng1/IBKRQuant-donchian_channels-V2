# import streamlit as st
# import plotly.graph_objects as go
# from datetime import datetime, timedelta
# import yfinance as yf
# from improved_donchian_strategy import ImprovedDonchianStrategy
# import pandas as pd

# def plot_backtest_results():
#    st.title('唐奇安通道策略回测可视化')

#    # 初始化策略
#    strategy = ImprovedDonchianStrategy(
#        symbol='spy',
#        capital=100000,
#        period=20,
#        base_tranches=9,
#        alert_threshold=0.002,
#        max_capital_per_trade=30000
#    )

#    # 获取回测时间范围
#    end_date = datetime.now()
#    start_date = end_date - timedelta(days=2)

#    # 获取数据
#    df = yf.download(strategy.symbol, start=start_date, end=end_date, interval='1m')
#    df = strategy.get_donchian_channels(df)

#    # 执行回测
#    result = strategy.backtest(
#         start_date=start_date.strftime('%Y-%m-%d'),
#         end_date=end_date.strftime('%Y-%m-%d')
#     )

#    fig = go.Figure()

#    # 添加K线
#    fig.add_trace(go.Candlestick(
#        x=df.index,
#        open=df['Open'],
#        high=df['High'],
#        low=df['Low'],
#        close=df['Close'],
#        name='K线'
#    ))

#    # 添加唐奇安通道
#    fig.add_trace(go.Scatter(
#        x=df.index,
#        y=df['upper_channel'],
#        name='上轨',
#        line=dict(color='blue', dash='dash')
#    ))

#    fig.add_trace(go.Scatter(
#        x=df.index,
#        y=df['lower_channel'],
#        name='下轨',
#        line=dict(color='blue', dash='dash'),
#        fill='tonexty'
#    ))

#    # 添加交易点和悬停信息
#    for record in result['trade_records']:
#        # 解析通道宽度百分比
#        width_str = record['channel_width']
#        width_value = float(width_str.strip('%'))
       
#        # 判断通道宽窄状态
#        if width_value < record.get('baseline_width', 0) * 0.8:
#            channel_state = "窄通道"
#        elif width_value > record.get('baseline_width', 0) * 1.2:
#            channel_state = "宽通道"
#        else:
#            channel_state = "正常通道"

#        hover_text = (
#            f"时间: {record['time']}<br>"
#            f"操作: {record['action']}<br>"
#            f"价格: ${record['price']:.2f}<br>"
#            f"数量: {record['size']}股<br>"
#            f"原因: {record['reason']}<br>"
#            f"通道状态: {channel_state}<br>"
#            f"当前通道宽度: {width_str}<br>"
#            f"基准通道宽度: {record.get('baseline_width', 0):.2f}%"
#        )
       
#        if 'pnl' in record:
#            hover_text += f"<br>盈亏: ${record['pnl']:.2f}"

#        if record['action'] == 'BUY':
#            fig.add_trace(go.Scatter(
#                x=[record['time']],
#                y=[record['price']],
#                mode='markers',
#                marker=dict(
#                    symbol='triangle-up',
#                    size=15,
#                    color='red'
#                ),
#                name='买入',
#                text=hover_text,
#                hoverinfo='text'
#            ))
#        else:
#            fig.add_trace(go.Scatter(
#                x=[record['time']],
#                y=[record['price']],
#                mode='markers',
#                marker=dict(
#                    symbol='triangle-down',
#                    size=15,
#                    color='green'
#                ),
#                name='卖出',
#                text=hover_text,
#                hoverinfo='text'
#            ))

#    # 更新布局
#    fig.update_layout(
#        title=f'{strategy.symbol} 回测结果',
#        xaxis_title='时间',
#        yaxis_title='价格',
#        height=800
#    )

#    # 显示图表
#    st.plotly_chart(fig, use_container_width=True)

#    # 显示回测统计
#    st.subheader('回测统计')
#    col1, col2, col3 = st.columns(3)
#    col1.metric('总交易次数', result['total_trades'])
#    col2.metric('盈利交易', result['winning_trades'])
#    col3.metric('胜率', f"{result['win_rate']:.2f}%")
   
#    st.metric('总盈亏', f"${result['total_pnl']:.2f}")

#    # 在页面下方添加交易详情表格
#    st.subheader('交易详情')
#    trades_df = pd.DataFrame([
#    {
#        '时间': record['time'],
#        '操作': record['action'], 
#        '价格': f"${record['price']:.2f}",
#        '上轨价格': f"${record['upper_price']:.2f}",
#        '下轨价格': f"${record['lower_price']:.2f}",
#        '与上轨距离': f"{record['upper_dist']:.3f}%",
#        '与下轨距离': f"{record['lower_dist']:.3f}%",
#        '数量': record['size'],
#        '原因': record['reason'],
#        '通道状态': "窄通道" if float(record['channel_width'].strip('%')) < record.get('baseline_width', 0) * 0.8 
#                   else ("宽通道" if float(record['channel_width'].strip('%')) > record.get('baseline_width', 0) * 1.2 
#                         else "正常通道"),
#        '当前通道宽度': record['channel_width'],
#        '基准通道宽度': f"{record.get('baseline_width', 0):.2f}%",
#        '调整后阈值': f"{record['adjusted_threshold']:.3f}%",
#        '可交易状态': record['can_trade'],
#        '盈亏': f"${record.get('pnl', 0):.2f}" if 'pnl' in record else '-'
#    }
#    for record in result['trade_records']
# ])
   
#    st.dataframe(trades_df)

# if __name__ == '__main__':
#    plot_backtest_results()



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
        symbol='spy',
        capital=100000,
        period=20,
        base_tranches=3,
        alert_threshold=0.002,
        max_capital_per_trade=30000
    )

    # 获取回测时间范围
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)  # 扩展到30天以获得更稳定的通道

    # 获取数据
    df = yf.download(strategy.symbol, start=start_date, end=end_date, interval='1m')
    df = strategy.get_donchian_channels(df)

    # 执行回测并获取所有决策点
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

    # 添加所有决策点（包括未执行交易的点）
    for signal in result['all_signals']:
        hover_text = (
            f"时间: {signal['time']}<br>"
            f"价格: ${signal['price']:.2f}<br>"
            f"通道状态: {signal['channel_state']}<br>"
            f"当前通道宽度: {signal['channel_width']}<br>"
            f"基准通道宽度: {signal['baseline_width']:.2f}%<br>"
            f"与上轨距离: {signal['upper_dist']:.3f}%<br>"
            f"与下轨距离: {signal['lower_dist']:.3f}%<br>"
            f"触发类型: {signal['trigger_type']}<br>"
            f"决策结果: {signal['action']}<br>"  # 修改 'decision' 为 'action'
            f"原因: {signal['reason']}"
        )

        if signal['trigger_type'] == 'NEAR_UPPER':
            marker_color = 'rgba(255,0,0,0.3)' if signal['action'] == 'NO_ACTION' else 'red'
        else:  # NEAR_LOWER
            marker_color = 'rgba(0,255,0,0.3)' if signal['action'] == 'NO_ACTION' else 'green'

        fig.add_trace(go.Scatter(
            x=[signal['time']],
            y=[signal['price']],
            mode='markers',
            marker=dict(
                symbol='circle' if signal['action'] == 'NO_ACTION' else 
                    ('triangle-up' if signal['trigger_type'] == 'NEAR_LOWER' else 'triangle-down'),
                size=15,
                color=marker_color
            ),
            name=signal['action'],
            text=hover_text,
            hoverinfo='text'
        ))

    # 更新布局
    fig.update_layout(
        title=f'{strategy.symbol} 回测结果 (包含所有决策点)',
        xaxis_title='时间',
        yaxis_title='价格',
        height=800
    )

    # 显示图表
    st.plotly_chart(fig, use_container_width=True)

    # 显示统计信息
    st.subheader('回测统计')
    cols = st.columns(4)
    cols[0].metric('总决策点数', len(result['all_signals']))
    cols[1].metric('实际交易次数', result['total_trades'])
    cols[2].metric('未执行交易次数', 
                   len([s for s in result['all_signals'] if s['action'] == 'NO_ACTION']))
    cols[3].metric('胜率', f"{result['win_rate']:.2f}%")

    # 显示详细的决策点记录
    st.subheader('决策点详情')
    signals_df = pd.DataFrame([
    {
        '时间': signal['time'],
        '触发类型': signal['trigger_type'],
        '决策结果': signal['action'],  # 修改 'decision' 为 'action'
        '价格': f"${signal['price']:.2f}",
        '通道状态': signal['channel_state'],
        '当前通道宽度': signal['channel_width'],
        '基准通道宽度': f"{signal['baseline_width']:.2f}%",
        '与上轨距离': f"{signal['upper_dist']:.3f}%",
        '与下轨距离': f"{signal['lower_dist']:.3f}%",
        '原因': signal['reason']
    }
    for signal in result['all_signals']
])
    
    # 添加过滤器
    signal_types = ['全部'] + list(signals_df['触发类型'].unique())
    decision_types = ['全部'] + list(signals_df['决策结果'].unique())
    
    col1, col2 = st.columns(2)
    selected_signal = col1.selectbox('触发类型', signal_types)
    selected_decision = col2.selectbox('决策结果', decision_types)
    
    # 应用过滤
    # 修改过滤逻辑部分
    filtered_df = signals_df
    if selected_signal != '全部':
        filtered_df = filtered_df[filtered_df['触发类型'] == selected_signal]
    if selected_decision != '全部':
        filtered_df = filtered_df[filtered_df['决策结果'] == selected_decision]  # 确保这里用的列名与上面的命名一致
    
    st.dataframe(filtered_df)

if __name__ == '__main__':
    plot_backtest_results()