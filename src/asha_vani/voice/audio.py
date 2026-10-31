"""Telephone audio conditioning and 1024-byte frame buffering (pure Python stdlib).

Implements Architecture Section 5.1 & 7 with zero external C/numpy dependencies:
- 8 kHz, 16-bit signed little-endian mono PCM (`SAMPLE_RATE_HZ = 8000`)
- Exact 1024-byte (512-sample, 64 ms) frame chunking (`FrameBuffer`)
- G.711 mu-law (8-bit) encode/decode round-trip (replaces removed `audioop`)
- 300-3400 Hz telephone bandpass filter (cascaded 1st-order high-pass + low-pass)
- Calibrated SNR background noise mixing (e.g. 10 dB and 5 dB SNR)
"""

from __future__ import annotations

from array import array
import math
import random
from typing import Iterator

SAMPLE_RATE_HZ: int = 8000
BYTES_PER_SAMPLE: int = 2  # 16-bit signed PCM (int16 little-endian)
FRAME_SAMPLES: int = 512   # 64 ms at 8000 Hz
FRAME_BYTES: int = FRAME_SAMPLES * BYTES_PER_SAMPLE  # 1024 bytes


def _clamp16(val: float | int) -> int:
    """Clamp a numeric sample to signed 16-bit integer range [-32768, 32767]."""
    i = int(round(val))
    if i > 32767:
        return 32767
    if i < -32768:
        return -32768
    return i


def bytes_to_samples(pcm16_bytes: bytes | bytearray) -> array[int]:
    """Unpack little-endian 16-bit PCM bytes into a signed short array."""
    if len(pcm16_bytes) % 2 != 0:
        pcm16_bytes = pcm16_bytes[:-1]
    samples = array("h")
    samples.frombytes(pcm16_bytes)
    return samples


def samples_to_bytes(samples: array[int]) -> bytes:
    """Pack signed 16-bit short array back into little-endian PCM bytes."""
    return samples.tobytes()


# ---------------------------------------------------------------------------
# 1. Exact 1024-byte Frame Buffer (64 ms @ 8 kHz PCM16)
# ---------------------------------------------------------------------------
class FrameBuffer:
    """Accumulates arbitrary incoming PCM16 byte chunks and yields exact 1024-byte frames."""

    def __init__(self, frame_bytes: int = FRAME_BYTES) -> None:
        if frame_bytes <= 0 or frame_bytes % BYTES_PER_SAMPLE != 0:
            raise ValueError("frame_bytes must be a positive multiple of 2.")
        self.frame_bytes = frame_bytes
        self._buf = bytearray()

    @property
    def buffered_bytes(self) -> int:
        return len(self._buf)

    def push(self, data: bytes | bytearray) -> list[bytes]:
        """Append incoming bytes and return all complete 1024-byte frames."""
        if data:
            self._buf.extend(data)
        frames: list[bytes] = []
        while len(self._buf) >= self.frame_bytes:
            frames.append(bytes(self._buf[: self.frame_bytes]))
            del self._buf[: self.frame_bytes]
        return frames

    def flush(self, pad_with_silence: bool = True) -> bytes | None:
        """Flush remaining bytes at end of stream (optionally zero-padded to frame_bytes)."""
        if not self._buf:
            return None
        remainder = bytes(self._buf)
        self._buf.clear()
        if pad_with_silence and len(remainder) < self.frame_bytes:
            remainder += b"\x00" * (self.frame_bytes - len(remainder))
        return remainder

    def clear(self) -> None:
        """Discard any partial frame (used when muting mic on half-duplex switch)."""
        self._buf.clear()


def iter_frames(pcm16_bytes: bytes, pad_final: bool = True) -> Iterator[bytes]:
    """Slice a complete PCM16 buffer into exact 1024-byte frames."""
    fb = FrameBuffer()
    for frame in fb.push(pcm16_bytes):
        yield frame
    tail = fb.flush(pad_with_silence=pad_final)
    if tail is not None:
        yield tail


# ---------------------------------------------------------------------------
# 2. G.711 mu-law Encode / Decode Round-Trip (Pure Python ITU-T G.711)
# ---------------------------------------------------------------------------
_MULAW_CLIP = 32635
_MULAW_BIAS = 0x84  # 132 in 16-bit linear PCM scale


def _encode_mulaw_sample(sample16: int) -> int:
    """Encode one 16-bit signed PCM sample to an 8-bit ITU-T G.711 mu-law byte."""
    sign = 0x80 if sample16 < 0 else 0x00
    mag = -sample16 if sample16 < 0 else sample16
    if mag > _MULAW_CLIP:
        mag = _MULAW_CLIP
    mag += _MULAW_BIAS

    exponent = 7
    mask = 0x4000
    while exponent > 0 and not (mag & mask):
        exponent -= 1
        mask >>= 1

    mantissa = (mag >> (exponent + 3)) & 0x0F
    return (~(sign | (exponent << 4) | mantissa)) & 0xFF


def _decode_mulaw_byte(ulaw_byte: int) -> int:
    """Decode one 8-bit ITU-T G.711 mu-law byte back to a 16-bit signed PCM sample."""
    u = (~ulaw_byte) & 0xFF
    sign = u & 0x80
    exponent = (u >> 4) & 0x07
    mantissa = u & 0x0F
    mag = (((mantissa << 3) + _MULAW_BIAS) << exponent) - _MULAW_BIAS
    return _clamp16(-mag if sign else mag)


# Precompute 256-entry lookup table for instant decoding
_MULAW_DECODE_LUT: tuple[int, ...] = tuple(_decode_mulaw_byte(b) for b in range(256))


def pcm16_to_mulaw(pcm16_bytes: bytes) -> bytes:
    """Compress 16-bit PCM bytes (2 bytes/sample) to 8-bit G.711 mu-law (1 byte/sample)."""
    samples = bytes_to_samples(pcm16_bytes)
    return bytes(_encode_mulaw_sample(s) for s in samples)


def mulaw_to_pcm16(mulaw_bytes: bytes) -> bytes:
    """Decompress 8-bit G.711 mu-law bytes (1 byte/sample) to 16-bit PCM (2 bytes/sample)."""
    decoded = array("h", (_MULAW_DECODE_LUT[b] for b in mulaw_bytes))
    return samples_to_bytes(decoded)


def mulaw_roundtrip(pcm16_bytes: bytes) -> bytes:
    """Round-trip 16-bit PCM through 8-bit G.711 mu-law to simulate PSTN quantization."""
    return mulaw_to_pcm16(pcm16_to_mulaw(pcm16_bytes))


# ---------------------------------------------------------------------------
# 3. Telephone Bandpass Filter (300 Hz - 3400 Hz @ 8 kHz)
# ---------------------------------------------------------------------------
def bandpass_300_3400(pcm16_bytes: bytes, sample_rate: int = SAMPLE_RATE_HZ) -> bytes:
    """Apply a 300-3400 Hz telephone bandpass filter (cascaded RC high-pass + low-pass)."""
    samples = bytes_to_samples(pcm16_bytes)
    if not samples:
        return b""

    dt = 1.0 / sample_rate
    # High-pass @ 300 Hz
    rc_hp = 1.0 / (2.0 * math.pi * 300.0)
    alpha_hp = rc_hp / (rc_hp + dt)

    # Low-pass @ 3400 Hz
    rc_lp = 1.0 / (2.0 * math.pi * 3400.0)
    alpha_lp = dt / (rc_lp + dt)

    out = array("h")
    prev_x = float(samples[0])
    hp_y = 0.0
    lp_y = 0.0

    for x in samples:
        fx = float(x)
        hp_y = alpha_hp * (hp_y + fx - prev_x)
        prev_x = fx
        lp_y = lp_y + alpha_lp * (hp_y - lp_y)
        out.append(_clamp16(lp_y))

    return samples_to_bytes(out)


# ---------------------------------------------------------------------------
# 4. Resampler & Calibrated SNR Noise Mixer (10 dB / 5 dB SNR)
# ---------------------------------------------------------------------------
def resample_pcm16(pcm16_bytes: bytes, src_rate: int, dst_rate: int = SAMPLE_RATE_HZ) -> bytes:
    """Linearly resample mono 16-bit PCM from src_rate to dst_rate (default 8000 Hz)."""
    if src_rate == dst_rate:
        return pcm16_bytes
    if src_rate <= 0 or dst_rate <= 0:
        raise ValueError("Sample rates must be positive.")

    src = bytes_to_samples(pcm16_bytes)
    n_src = len(src)
    if n_src == 0:
        return b""

    n_dst = int(round(n_src * (dst_rate / src_rate)))
    if n_dst <= 0:
        return b""

    ratio = (n_src - 1) / max(1, n_dst - 1)
    dst = array("h")
    for i in range(n_dst):
        pos = i * ratio
        idx = int(pos)
        frac = pos - idx
        if idx + 1 < n_src:
            interp = src[idx] * (1.0 - frac) + src[idx + 1] * frac
        else:
            interp = float(src[idx])
        dst.append(_clamp16(interp))
    return samples_to_bytes(dst)


def rms_energy(pcm16_bytes: bytes) -> float:
    """Compute Root-Mean-Square (RMS) amplitude of a 16-bit PCM buffer."""
    samples = bytes_to_samples(pcm16_bytes)
    if not samples:
        return 0.0
    mean_sq = sum(float(s) * float(s) for s in samples) / len(samples)
    return math.sqrt(mean_sq)


def mix_snr_noise(
    pcm16_bytes: bytes,
    snr_db: float | None,
    *,
    seed: int = 42,
) -> bytes:
    """Mix deterministic white/village-band noise at a target Signal-to-Noise Ratio (dB).

    Formula: SNR_dB = 20 * log10(RMS_signal / RMS_noise)
    => RMS_noise = RMS_signal / (10 ** (snr_db / 20))
    """
    if snr_db is None:
        return pcm16_bytes

    samples = bytes_to_samples(pcm16_bytes)
    if not samples:
        return b""

    sig_rms = rms_energy(pcm16_bytes)
    if sig_rms == 0.0:
        return pcm16_bytes

    target_noise_rms = sig_rms / (10.0 ** (snr_db / 20.0))
    rng = random.Random(seed)

    # Unit Gaussian noise has RMS ~= 1.0, so scaling by target_noise_rms gives exact SNR
    mixed = array(
        "h",
        (_clamp16(s + rng.gauss(0.0, target_noise_rms)) for s in samples),
    )
    return samples_to_bytes(mixed)


def condition_telephone_audio(
    pcm16_bytes: bytes,
    *,
    src_rate: int = SAMPLE_RATE_HZ,
    snr_db: float | None = None,
    apply_bandpass: bool = True,
    apply_mulaw: bool = True,
) -> bytes:
    """Full ingest pipeline: resample to 8 kHz -> 300-3400 Hz bandpass -> SNR noise -> mu-law round-trip."""
    audio = resample_pcm16(pcm16_bytes, src_rate=src_rate, dst_rate=SAMPLE_RATE_HZ)
    if apply_bandpass:
        audio = bandpass_300_3400(audio, sample_rate=SAMPLE_RATE_HZ)
    if snr_db is not None:
        audio = mix_snr_noise(audio, snr_db=snr_db)
    if apply_mulaw:
        audio = mulaw_roundtrip(audio)
    return audio
