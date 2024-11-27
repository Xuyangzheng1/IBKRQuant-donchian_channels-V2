# IBKRQuant-donchian_channels-V2


* `improved_donchian_strategy.py`: 策略主文件
* `mock_tws.py`: 模拟TWS客户端
* `test_strategy.py`: 测试文件
* 2024/11/27



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
