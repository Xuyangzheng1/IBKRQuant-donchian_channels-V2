# IBKRQuant-donchian_channels-V2


* `improved_donchian_strategy.py`: 策略主文件
* `mock_tws.py`: 模拟TWS客户端
* `test_strategy.py`: 测试文件
* 2024/11/27
* 待更新：1.盈利太tm低了,2,日志输入至csv,3如果差价大可以适当提升单次交易限额
* 2024/11/17晚更新：窄通道时：阈值 * 1.2，提高标准避免震荡
宽通道时：阈值 * 0.8，降低标准提前入场
* INFO:DonchianStrategy_TSLA:初始资金: $1000
INFO:DonchianStrategy_TSLA:最终资金: $1002.6782836914062
INFO:DonchianStrategy_TSLA:总盈亏: $2.68
INFO:DonchianStrategy_TSLA:总交易次数: 34
INFO:DonchianStrategy_TSLA:盈利交易: 25
INFO:DonchianStrategy_TSLA:胜率: 73.53%

2024/11/28
添加了和ibkr交互信息
2024-11-28 14:54:22,830 - DonchianStrategy_TSLA - INFO - 当前价格: $332.79
2024-11-28 14:54:22,830 - DonchianStrategy_TSLA - INFO - 上轨: $333.50
2024-11-28 14:54:22,830 - DonchianStrategy_TSLA - INFO - 下轨: $332.08
2024-11-28 14:54:22,830 - DonchianStrategy_TSLA - INFO - 持仓数量: 1049.0
2024-11-28 14:54:22,830 - DonchianStrategy_TSLA - INFO - 当日盈亏: $2248.21
2024-11-28 14:54:22,830 - DonchianStrategy_TSLA - INFO - 未实现盈亏: $2024.98
2024-11-28 14:54:22,830 - DonchianStrategy_TSLA - INFO - 已实现盈亏: $214.97

待更新：1.唐奇安连续下跌时建议多批少量买入。
2.终端输出格式化
3.增加macd指标辅助



# IBKRQuant-donchian_channels-V2

一个基于唐奇安通道的量化交易策略，结合 Interactive Brokers API实现自动化交易，并包含回测功能。

## 策略说明

### 唐奇安通道策略

唐奇安通道是一种趋势跟踪指标，由理查德·唐奇安发明。它由以下几条线组成：

- 上轨：过去n个周期的最高价
- 下轨：过去n个周期的最低价
- 中轨：上轨和下轨的平均值

### 交易逻辑

1. 入场条件

- 如果价格接近下轨（距离小于阈值），则考虑开仓做多
- 如果通道比基准通道窄，则降低触发阈值以便更容易入场
- 如果通道比基准通道宽，则提高触发阈值以避免假突破

2. 出场条件

- 如果价格接近上轨（距离小于阈值），则考虑平仓
- 如果到达收盘时间，强制平仓
- 动态调整出场阈值，通道越宽阈值越大

3. 仓位管理

- 如果交易量较小（小于10股），取消交易
- 每次交易不超过最大资金限制(max_capital_per_trade)
- 实时验证仓位，防止出现负仓位情况

### 策略参数

```python


strategy = ImprovedDonchianStrategy(
   symbol='AAPL',           # 交易标的
   capital=100000,          # 初始资金
   period=20,               # 通道周期，用于计算上下轨
   base_tranches=3,         # 分批数，将总交易量分成几次执行
   alert_threshold=0.004,   # 触发阈值，价格接近通道线的距离标准
   max_capital_per_trade=50000  # 单次最大交易金额
)
```


# 唐奇安通道策略参数解析

## 核心指标

### 1. 通道宽度

- 定义：`通道宽度 = (上轨 - 下轨) / 当前价格`
- 计算基准：使用过去三天的平均通道宽度作为基准
- 作用：用于判断当前市场波动程度，动态调整交易阈值

### 2. 交易触发阈值 (alert_threshold)

- 默认值：0.004 (0.4%)
- 定义：股价与通道线（上轨或下轨）的接近程度
- 计算方式：

```python
 与下轨距离 = |当前价格 - 下轨| / 下轨
 与上轨距离 = |当前价格 - 上轨| / 上轨
```
