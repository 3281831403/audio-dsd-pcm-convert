"""DSD64（2.8224 MHz 1-bit）到 PCM 的抽取、滤波与批量转换工具。

全局约定
--------
* 1-bit DSD 采样：1 -> +1.0，0 -> -1.0（双极性 NRZ）。
* DSD64 采样率：2_822_400 Hz，奈奎斯特频率 1_411_200 Hz。
* 数值比较容差：1e-9。
"""

from __future__ import annotations

import math

DSD64_RATE = 2_822_400.0
NYQUIST = DSD64_RATE / 2.0
TOL = 1e-9


def bitstream_to_pulses(bits):
    """把 1-bit DSD 流映射成 ±1 脉冲序列。

    参数
    ----
    bits : 可迭代对象
        元素只能是 0 或 1（bool 也可以）。

    返回
    ----
    list[float]
        与输入等长的脉冲序列：1 -> +1.0，0 -> -1.0。

    说明
    ----
    0 必须映射为 -1.0 而不是 0.0。若把 0 当成 0，全零流会凭空多出
    半个 LSB 的直流偏移，后续抽取输出的基线就不在 0 上了。
    出现 0/1 以外的值抛 ValueError。
    """
    pulses = []
    for bit in bits:
        if bit == 1:
            pulses.append(1.0)
        elif bit == 0:
            pulses.append(-1.0)
        else:
            raise ValueError(f"bitstream values must be 0 or 1, got {bit!r}")
    return pulses


def moving_average(x, n):
    """零相位（群延迟已补偿）的移动平均。

    参数
    ----
    x : 序列
        输入样本。
    n : int
        窗口长度，必须是正奇数。

    返回
    ----
    list[float]
        长度与 x 相同。y[i] 等于以 x[i] 为中心、宽度为 n 的窗口均值；
        窗口越出输入边界的位置按 0.0 计（零填充）。

    说明
    ----
    因果实现 y[k] = mean(x[k:k+n]) 自带 (n-1)/2 个样本的群延迟，会和
    FIR 低通路径对不齐；这里必须把窗口中心对齐到输出下标 i。
    """
    if not isinstance(n, int) or n <= 0 or n % 2 == 0:
        raise ValueError(f"n must be a positive odd integer, got {n!r}")
    x = list(x)
    half = (n - 1) // 2
    out = []
    for i in range(len(x)):
        total = 0.0
        for j in range(i - half, i + half + 1):
            if 0 <= j < len(x):
                total += x[j]
        out.append(total / n)
    return out


def fir_lowpass(taps, cutoff):
    """用加窗 sinc 设计线性相位低通 FIR。

    参数
    ----
    taps : int
        滤波器长度，必须是正奇数（Type-I 线性相位）。
    cutoff : float
        归一化截止频率，1.0 对应 DSD64 的奈奎斯特频率（1.4112 MHz）；
        必须满足 0 < cutoff < 1。例如 0.013 约等于 18.3 kHz。

    返回
    ----
    list[float]
        长度为 taps 的对称系数。窗函数使用 Blackman 窗
        w[k] = 0.42 - 0.5*cos(2*pi*k/(N-1)) + 0.08*cos(4*pi*k/(N-1))，
        理想低通为 h[k] = 2*fc*sinc(2*fc*(k-M))，其中
        fc = cutoff/2（单位为 cycles/sample）、M = (N-1)/2。
        返回前按总和归一化，使 sum(taps) == 1（误差 < 1e-9）。

    说明
    ----
    矩形窗的第一旁瓣只有约 -13 dB，30 kHz 以上的高频噪声压不下去；
    Blackman 窗可以把该频段压到 -60 dB 以下。系数若不归一，直流增益
    会偏离 1，PCM 电平就会整体偏大或偏小。
    """
    if not isinstance(taps, int) or taps <= 0 or taps % 2 == 0:
        raise ValueError(f"taps must be a positive odd integer, got {taps!r}")
    if not 0.0 < cutoff < 1.0:
        raise ValueError(f"cutoff must satisfy 0 < cutoff < 1, got {cutoff!r}")

    n = taps
    m = (n - 1) / 2.0
    fc = cutoff / 2.0  # cycles/sample，1.0 对应采样率
    coeffs = []
    for k in range(n):
        d = k - m
        if d == 0.0:
            ideal = 2.0 * fc
        else:
            ideal = math.sin(2.0 * math.pi * fc * d) / (math.pi * d)
        window = (
            0.42
            - 0.5 * math.cos(2.0 * math.pi * k / (n - 1))
            + 0.08 * math.cos(4.0 * math.pi * k / (n - 1))
        )
        coeffs.append(ideal * window)

    total = sum(coeffs)
    return [c / total for c in coeffs]


def decimate(x, factor, taps):
    """先低通滤波、再按整数因子抽取。

    参数
    ----
    x : 序列
        输入样本（通常是 bitstream_to_pulses 的输出）。
    factor : int
        正整数抽取因子，例如 DSD64 -> 44.1 kHz 时取 64。
    taps : 序列
        线性相位 FIR 系数，通常来自 fir_lowpass。

    返回
    ----
    list[float]
        长度严格等于 ceil(len(x) / factor)。第 m 个输出对应输入时间
        m*factor。滤波采用中心对称的零相位卷积（不引入
        (len(taps)-1)/2 个样本的延迟）；卷积窗口越界时按最近的端点
        样本延拓（edge replication），因此常数输入会得到常数输出。

    说明
    ----
    必须先滤波再抽取。反过来先隔点采样会让 30 kHz 以上的 DSD 噪声
    直接混叠进音频带。输出长度也不能写成 len(x)//factor，否则最后
    一个不完整抽取块里的样本会被丢掉。
    """
    x = list(x)
    taps = list(taps)
    if not isinstance(factor, int) or factor <= 0:
        raise ValueError(f"factor must be a positive integer, got {factor!r}")
    if not x:
        return []

    half = (len(taps) - 1) // 2
    last = len(x) - 1
    out = []
    for m in range(math.ceil(len(x) / factor)):
        center = m * factor
        acc = 0.0
        for k, coeff in enumerate(taps):
            j = center + k - half
            if j < 0:
                j = 0
            elif j > last:
                j = last
            acc += coeff * x[j]
        out.append(acc)
    return out


def dc_gain(taps):
    """返回滤波器的直流增益，即所有系数之和。

    参数
    ----
    taps : 序列
        FIR 系数。

    返回
    ----
    float
        sum(taps)。归一化良好的低通滤波器应约等于 1.0。

    说明
    ----
    不能用 max(taps) 代替求和；最大系数只是通带中心附近的峰值，
    会把直流增益低估好几倍。
    """
    return float(sum(taps))


def passband_ripple(taps, freqs):
    """计算给定频点上的最大幅度起伏（dB，峰峰值）。

    参数
    ----
    taps : 序列
        FIR 系数。
    freqs : 序列
        频率，单位 Hz，可为任意浮点数；按 DSD64 的 2.8224 MHz
        采样率解释。

    返回
    ----
    float
        各频点相对直流增益的幅度响应（dB）的 max-min；freqs 为空
        时返回 0.0。

    说明
    ----
    必须先把 Hz 换算成归一化数字频率 2*pi*f/2_822_400 再计算
    |H(f)|。直接把 Hz 当作归一化频率会让通带与阻带的位置整体错位，
    1 kHz 会被当成几千个周期/样本。
    """
    taps = list(taps)
    freqs = list(freqs)
    if not freqs:
        return 0.0

    def _response_db(freq):
        omega = 2.0 * math.pi * freq / DSD64_RATE
        real = 0.0
        imag = 0.0
        for k, coeff in enumerate(taps):
            real += coeff * math.cos(omega * k)
            imag -= coeff * math.sin(omega * k)
        magnitude = math.hypot(real, imag)
        return 20.0 * math.log10(max(magnitude, 1e-300))

    dc = _response_db(0.0)
    levels = [_response_db(f) - dc for f in freqs]
    return max(levels) - min(levels)


def validate_taps(taps, factor):
    """校验 FIR 系数与抽取因子是否自洽。

    按固定顺序检查：
    1. taps 长度必须是正奇数，否则追加
       "taps length must be odd, got {n}"；
    2. |sum(taps) - 1| 必须 <= 1e-6，否则追加
       "taps must be normalized: sum(taps)={sum:.12g}"；
    3. factor 必须是正整数，否则追加
       "factor must be a positive integer, got {factor!r}"。

    参数
    ----
    taps : 序列
        FIR 系数。
    factor : int
        抽取因子。

    返回
    ----
    list[str]
        发现的所有问题；全部通过时返回空列表 []。

    说明
    ----
    偶数长度的 taps 无法满足 Type-I 线性相位假设，群延迟也不再是
    整数个样本；不检查这一点会让后续对齐整体错开半个样本。
    """
    errors = []
    taps = list(taps)
    if len(taps) == 0 or len(taps) % 2 == 0:
        errors.append(f"taps length must be odd, got {len(taps)}")
    if abs(sum(taps) - 1.0) > 1e-6:
        errors.append(f"taps must be normalized: sum(taps)={sum(taps):.12g}")
    if not isinstance(factor, int) or isinstance(factor, bool) or factor <= 0:
        errors.append(f"factor must be a positive integer, got {factor!r}")
    return errors


def dsd_batch(streams, config):
    """批量把多条独立的 1-bit DSD 流转换成 PCM。

    参数
    ----
    streams : dict[str, 序列]
        名字 -> 1-bit 流。
    config : dict
        必须包含 "taps"（FIR 系数）和 "factor"（抽取因子）。

    返回
    ----
    dict[str, list[float]]
        键与 streams 相同；每个值等于对该条流单独执行
        decimate(bitstream_to_pulses(bits), factor, taps) 的结果。

    说明
    ----
    每条流必须使用独立的滤波器状态/历史。不能把上一条流的尾部
    接着喂给下一条流，否则第二条流的开头会串进第一条流的声音。
    """
    taps = config["taps"]
    factor = config["factor"]
    return {
        name: decimate(bitstream_to_pulses(bits), factor, taps)
        for name, bits in streams.items()
    }
