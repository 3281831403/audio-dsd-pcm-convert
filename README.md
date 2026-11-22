# audio-dsd-pcm-convert

## 题面

老 SACD 抓轨文件要转成工作站能用的 PCM，结果高频一片毛刺、音量还忽大忽小：DSD64 的 1-bit 流按 2.8224 MHz 采样，0 必须映射成 -1 而不是 0，否则整条流凭空多出半个 LSB 的直流；抽取时先隔 64 点再滤波，30 kHz 以上的噪声原样混叠回音频带；加窗 sinc 只套矩形窗，第一旁瓣只有 -13 dB，阻带压不下去；系数设计完不归一，直流增益不等于 1；输出长度按 len//factor 截断，最后一个不完整块被丢掉；移动平均不补 (n-1)/2 的群延迟，和 FIR 路径对不齐；dc_gain 拿最大系数当增益；passband_ripple 把 Hz 当归一化频率算，通带位置整体错位；dsd_batch 让第二条流复用第一条的滤波器状态，声道之间串音；validate_taps 不检查奇数长度，线性相位假设失效。dsd/decimate.py 是核心，tests/test_decimate.py 15个红。要修的是：先滤波后抽取、换 Blackman 窗并把系数归一化、输出长度取 ceil、移动平均补偿群延迟、dc_gain 改为系数求和、passband_ripple 按 2.8224 MHz 把 Hz 换算成归一化频率、0 映射为 -1、每条流使用独立状态、校验 taps 为奇数长度。

## 运行方式

在仓库根目录执行：

```powershell
python -m pytest tests -q
```

初始状态下恰好有 `15 failed`，且没有 collection error、import error 或 syntax error；15 条失败全部来自 `NotImplementedError`。实现 `dsd/decimate.py` 中 8 个函数后，同一命令应当变成 `15 passed`。测试离线运行，不使用随机数，也不 sleep。

## 字段表

| 字段 | 取值 |
| --- | --- |
| 所属领域 | 游戏、图形与多媒体 |
| 子方向 | 音视频处理 |
| 任务类型 | Bugfix |
| 语言/框架 | Python 3, pytest |
| 环境可复现等级 | 无外部依赖 |

## 约束

`tests/test_decimate.py` 不可修改；验收以该文件中的 15 条测试为准。实现只允许使用 Python 标准库（`math`）与 pytest，不得联网，不得引入第三方 DSP 库。
