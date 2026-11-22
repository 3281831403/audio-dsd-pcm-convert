"""DSD64 -> PCM 抽取链路的黑盒验收测试。

测试只依赖 dsd.decimate 的公开契约，不修改实现，也不使用随机数或 sleep。
"""

import math

import pytest

from dsd.decimate import (
    bitstream_to_pulses,
    moving_average,
    fir_lowpass,
    decimate,
    dc_gain,
    passband_ripple,
    validate_taps,
    dsd_batch,
)

FS = 2_822_400.0
FACTOR = 64
TAPS = 769
CUTOFF = 0.013
TOL = 1e-9


def _tone(freq, count, amplitude=1.0):
    return [
        amplitude * math.sin(2.0 * math.pi * freq * i / FS)
        for i in range(count)
    ]


def _response_db(taps, freq):
    """直接由系数计算 |H(f)| 的 dB 值（相对直流增益）。"""
    omega = 2.0 * math.pi * freq / FS
    real = 0.0
    imag = 0.0
    for k, coeff in enumerate(taps):
        real += coeff * math.cos(omega * k)
        imag -= coeff * math.sin(omega * k)
    magnitude = math.hypot(real, imag)
    return 20.0 * math.log10(max(magnitude, 1e-300))


# ---------------------------------------------------------------- 10 个缺陷


def test_decimate_filters_before_downsampling():
    """先抽取再滤波会把 44.3 kHz 混叠成 200 Hz，阻带抑制断言必须失败。"""
    taps = fir_lowpass(TAPS, CUTOFF)
    x = _tone(44_300.0, 8192)
    input_rms = math.sqrt(sum(v * v for v in x) / len(x))
    assert input_rms > 0.6  # 输入确实是满幅单音

    y = decimate(x, FACTOR, taps)
    assert len(y) == math.ceil(len(x) / FACTOR)

    edge = 8
    core = y[edge:-edge]
    output_rms = math.sqrt(sum(v * v for v in core) / len(core))
    # 44.3 kHz 在 44.1 kHz 输出率下折叠到 200 Hz；先抽取会让它原样留下。
    assert output_rms < 1e-3


def test_fir_lowpass_stopband_is_not_rectangular():
    """矩形窗的第一旁瓣约 -13 dB，30 kHz 处必须被压到 -60 dB 以下。"""
    taps = fir_lowpass(TAPS, CUTOFF)
    assert _response_db(taps, 30_000.0) < -60.0


def test_fir_lowpass_coefficients_are_normalized():
    """系数和不等于 1 会让直流增益偏离 1，破坏 PCM 电平。"""
    for length, cutoff in ((129, 0.05), (257, 0.02)):
        taps = fir_lowpass(length, cutoff)
        assert len(taps) == length
        assert abs(sum(taps) - 1.0) < TOL


def test_decimate_keeps_final_partial_block():
    """1000 个样本按 64 抽取应有 16 个输出，不能用 len//factor 丢掉尾巴。"""
    x = [1.0] * 1000
    y = decimate(x, FACTOR, [1.0])
    assert len(y) == 16  # ceil(1000 / 64)
    assert y[-1] == pytest.approx(1.0, abs=TOL)


def test_moving_average_compensates_group_delay():
    """脉冲响应峰值必须对齐输入脉冲，而不是滞后 (n-1)/2 个样本。"""
    x = [0.0] * 41
    x[20] = 1.0
    y = moving_average(x, 9)

    assert len(y) == len(x)
    assert y[15] == pytest.approx(0.0, abs=TOL)
    for i in range(16, 25):
        assert y[i] == pytest.approx(1.0 / 9.0, abs=TOL)
    assert y[25] == pytest.approx(0.0, abs=TOL)


def test_dc_gain_is_sum_not_max_coefficient():
    """直流增益是系数之和；取最大系数会把 1.0 估成 0.4。"""
    taps = [0.1, 0.2, 0.4, 0.2, 0.1]
    assert dc_gain(taps) == pytest.approx(1.0, abs=TOL)
    assert dc_gain([0.25, 0.5, 0.25]) == pytest.approx(1.0, abs=TOL)


def test_passband_ripple_uses_hz_not_normalized_frequency():
    """频点按 Hz 解释；若当成归一化频率，1 kHz 会被算到几千个周期/样本。"""
    taps = fir_lowpass(TAPS, CUTOFF)
    # 故意使用非整数 Hz，避免“把 Hz 当归一化频率”的实现恰好落在直流上。
    ripple = passband_ripple(taps, [1_000.5, 4_321.7, 9_876.5])
    assert 0.0 <= ripple < 0.1


def test_bitstream_maps_zero_to_minus_one():
    """0 必须映射成 -1.0；映射成 0.0 会让全零流带上半个 LSB 的直流。"""
    pulses = bitstream_to_pulses([0, 0, 0, 0])
    assert pulses == [-1.0, -1.0, -1.0, -1.0]
    assert sum(pulses) / len(pulses) == pytest.approx(-1.0, abs=TOL)
    assert bitstream_to_pulses([1, 0, 1, 0]) == [1.0, -1.0, 1.0, -1.0]


def test_dsd_batch_does_not_share_filter_state():
    """两条流必须各自从零状态开始，第二条流不能接着第一条的尾部跑。"""
    taps = fir_lowpass(TAPS, CUTOFF)
    left = [1, 0] * 512
    right = [0, 1] * 256
    got = dsd_batch({"left": left, "right": right}, {"taps": taps, "factor": FACTOR})

    assert set(got) == {"left", "right"}
    expected_left = decimate(bitstream_to_pulses(left), FACTOR, taps)
    expected_right = decimate(bitstream_to_pulses(right), FACTOR, taps)
    assert got["left"] == pytest.approx(expected_left, abs=1e-12)
    assert got["right"] == pytest.approx(expected_right, abs=1e-12)


def test_validate_taps_rejects_even_length():
    """偶数长度无法满足 Type-I 线性相位，validate_taps 必须报错。"""
    valid = fir_lowpass(65, 0.05)
    assert validate_taps(valid, FACTOR) == []

    errors = validate_taps([0.5, 0.5], FACTOR)
    assert errors
    assert any("odd" in message.lower() for message in errors)


# ---------------------------------------------------------------- 5 个补充验收


def test_constant_bitstream_stays_constant_through_chain():
    """常数 1 比特流经完整链路后，每个输出都应精确为 1.0。"""
    taps = fir_lowpass(TAPS, CUTOFF)
    pcm = decimate(bitstream_to_pulses([1] * 8192), FACTOR, taps)
    assert len(pcm) == 128
    for value in pcm:
        assert abs(value - 1.0) < TOL


def test_one_khz_sine_amplitude_error_below_0_1_db():
    """1 kHz 正弦通过链路后，幅度误差必须小于 0.1 dB。"""
    taps = fir_lowpass(TAPS, CUTOFF)
    x = _tone(1_000.0, 16384)
    y = decimate(x, FACTOR, taps)

    edge = 8
    measured = y[edge:-edge]
    reference = [
        math.sin(2.0 * math.pi * 1_000.0 * (m * FACTOR) / FS)
        for m in range(edge, len(y) - edge)
    ]
    numerator = sum(a * b for a, b in zip(measured, reference))
    denominator = sum(b * b for b in reference)
    fitted_amplitude = numerator / denominator
    error_db = 20.0 * math.log10(abs(fitted_amplitude))
    assert abs(error_db) < 0.1


def test_stopband_above_30khz_is_below_minus_60_db():
    """30 kHz 以上的所有验收频点都要有 60 dB 以上的衰减。"""
    taps = fir_lowpass(TAPS, CUTOFF)
    for freq in (30_000.0, 35_000.0, 45_000.0, 70_000.0, 110_000.0):
        assert _response_db(taps, freq) < -60.0


def test_decimate_output_length_is_ceil():
    """输出长度必须严格是 ceil(len(x)/factor)，含不完整尾块。"""
    for count in (0, 1, 63, 64, 65, 127, 128, 1000):
        x = [1.0] * count
        y = decimate(x, FACTOR, [1.0])
        assert len(y) == math.ceil(count / FACTOR)


def test_batch_matches_individual_conversions():
    """批量结果必须逐条等于单独转换，名字和顺序都不能影响数值。"""
    taps = fir_lowpass(TAPS, CUTOFF)
    streams = {
        "a": [1, 1, 0, 0] * 256,
        "b": [0, 1, 1, 0] * 128,
        "c": [1] * 100,
    }
    got = dsd_batch(streams, {"taps": taps, "factor": FACTOR})
    for name, bits in streams.items():
        expected = decimate(bitstream_to_pulses(bits), FACTOR, taps)
        assert got[name] == pytest.approx(expected, abs=1e-12)
