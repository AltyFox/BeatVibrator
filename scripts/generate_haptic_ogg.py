#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import shutil
import struct
import subprocess
import sys
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path


DEFAULT_SAMPLE_RATE = 48_000
DEFAULT_LOW_PASS_CUTOFF = 200.0
DEFAULT_CARRIER_HZ = 180.0
DEFAULT_RMS_WINDOW = 512
DEFAULT_RMS_HOP = 256
DEFAULT_ONSET_THRESHOLD = 0.3
DEFAULT_MIN_ONSET_INTERVAL_MS = 50.0

MIN_INTERVAL_MS = 80
MERGE_WINDOW_MS = 120
MIN_INTENSITY = 0.08
MIN_PULSE_INTENSITY = 0.15
BASE_DURATION_MS = 40
MAX_DURATION_MS = 100
MIN_DURATION_MS = 20
ONSET_BOOST = 0.35
AMPLITUDE_COMPRESSION_EXPONENT = 0.6
ENVELOPE_TIME_SECONDS = 0.005


@dataclass(frozen=True)
class VibrationPulse:
    time_ms: int
    intensity: float
    duration_ms: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate an Android-compatible OGG/Vorbis file with a dedicated "
            "haptic channel from an input audio file."
        )
    )
    parser.add_argument("input", type=Path, help="Input audio file supported by ffmpeg")
    parser.add_argument("output", type=Path, help="Output .ogg file path")
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=DEFAULT_SAMPLE_RATE,
        help=f"Intermediate sample rate in Hz (default: {DEFAULT_SAMPLE_RATE})",
    )
    parser.add_argument(
        "--low-pass-cutoff",
        type=float,
        default=DEFAULT_LOW_PASS_CUTOFF,
        help=f"Butterworth low-pass cutoff in Hz (default: {DEFAULT_LOW_PASS_CUTOFF})",
    )
    parser.add_argument(
        "--carrier-hz",
        type=float,
        default=DEFAULT_CARRIER_HZ,
        help="Legacy compatibility option; ignored because raw haptic envelopes are written directly.",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=6,
        help="Vorbis quality level passed to ffmpeg (default: 6)",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep temporary WAV files for inspection",
    )
    return parser.parse_args()


def ensure_binary(name: str) -> str:
    binary = shutil.which(name)
    if binary is None:
        raise SystemExit(f"Missing required binary: {name}")
    return binary


def run_command(*args: str) -> None:
    try:
        subprocess.run(args, check=True)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from exc


def decode_audio(ffmpeg: str, input_path: Path, sample_rate: int, stereo_wav: Path, mono_wav: Path) -> None:
    run_command(
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-ar",
        str(sample_rate),
        "-ac",
        "2",
        "-c:a",
        "pcm_s16le",
        str(stereo_wav),
    )
    run_command(
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        str(mono_wav),
    )


def read_wav_float_samples(path: Path) -> tuple[int, list[float]]:
    with wave.open(str(path), "rb") as wav_file:
        channel_count = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frame_count = wav_file.getnframes()

        if channel_count != 1:
            raise ValueError(f"Expected mono WAV for analysis, received {channel_count} channels")
        if sample_width != 2:
            raise ValueError(f"Expected 16-bit WAV for analysis, received {sample_width * 8}-bit")

        raw_frames = wav_file.readframes(frame_count)
        pcm = struct.unpack("<" + "h" * frame_count, raw_frames)
        samples = [sample / 32768.0 for sample in pcm]
        return sample_rate, samples


def write_wav_float_samples(path: Path, sample_rate: int, samples: list[float]) -> None:
    clipped = [max(-1.0, min(1.0, sample)) for sample in samples]
    pcm = [int(sample * 32767.0) for sample in clipped]
    packed = struct.pack("<" + "h" * len(pcm), *pcm)

    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(packed)


def apply_butterworth_low_pass(samples: list[float], sample_rate: int, cutoff_hz: float) -> list[float]:
    omega = 2.0 * math.pi * cutoff_hz / sample_rate
    sin_omega = math.sin(omega)
    cos_omega = math.cos(omega)
    q = math.sqrt(2.0) / 2.0
    alpha = sin_omega / (2.0 * q)
    a0_inv = 1.0 / (1.0 + alpha)

    a0 = ((1.0 - cos_omega) / 2.0) * a0_inv
    a1 = (1.0 - cos_omega) * a0_inv
    a2 = a0
    b1 = (-2.0 * cos_omega) * a0_inv
    b2 = (1.0 - alpha) * a0_inv

    x1 = x2 = y1 = y2 = 0.0
    output: list[float] = []

    for sample in samples:
        y0 = a0 * sample + a1 * x1 + a2 * x2 - b1 * y1 - b2 * y2
        output.append(y0)
        x2 = x1
        x1 = sample
        y2 = y1
        y1 = y0

    return output


def compute_rms_over_windows(samples: list[float], window_size: int, hop_size: int) -> list[float]:
    values: list[float] = []
    start = 0

    while start + window_size <= len(samples):
        window = samples[start : start + window_size]
        sum_squares = sum(sample * sample for sample in window)
        values.append(math.sqrt(sum_squares / window_size))
        start += hop_size

    max_value = max(values, default=0.0)
    if max_value > 0.0:
        return [value / max_value for value in values]
    return values


def detect_onsets(rms_values: list[float], threshold: float, min_interval_ms: float, sample_rate: int, hop_size: int) -> list[int]:
    if len(rms_values) < 3:
        return []

    deltas = [0.0]
    deltas.extend(max(0.0, rms_values[index] - rms_values[index - 1]) for index in range(1, len(rms_values)))

    max_delta = max(deltas, default=0.0)
    if max_delta <= 0.0:
        return []

    normalized = [delta / max_delta for delta in deltas]
    min_interval_frames = max(1, int(min_interval_ms * sample_rate / (1000.0 * hop_size)))
    onsets: list[int] = []
    last_onset = -min_interval_frames

    for index in range(1, len(normalized) - 1):
        if (
            normalized[index] >= threshold
            and normalized[index] > normalized[index - 1]
            and normalized[index] >= normalized[index + 1]
            and index - last_onset >= min_interval_frames
        ):
            onsets.append(index)
            last_onset = index

    return onsets


def build_pulses(rms_values: list[float], onsets: list[int], sample_rate: int, hop_size: int) -> list[VibrationPulse]:
    candidates: list[tuple[int, float]] = []

    for index, rms_value in enumerate(rms_values):
        if rms_value >= MIN_INTENSITY:
            timestamp_ms = int(index * hop_size * 1000 / sample_rate)
            candidates.append((timestamp_ms, rms_value))

    for onset_index in onsets:
        timestamp_ms = int(onset_index * hop_size * 1000 / sample_rate)
        boost = min(
            1.0,
            rms_values[onset_index] + ONSET_BOOST if onset_index < len(rms_values) else ONSET_BOOST,
        )
        candidates.append((timestamp_ms, boost))

    candidates.sort(key=lambda item: item[0])
    pulses: list[VibrationPulse] = []
    last_time = -MIN_INTERVAL_MS
    cursor = 0

    while cursor < len(candidates):
        base_time, base_amp = candidates[cursor]
        if base_amp < MIN_INTENSITY or base_time - last_time < MIN_INTERVAL_MS:
            cursor += 1
            continue

        merged_time = float(base_time)
        merged_amp = base_amp
        count = 1
        merge_cursor = cursor + 1

        while merge_cursor < len(candidates) and candidates[merge_cursor][0] - base_time < MERGE_WINDOW_MS:
            merged_time += candidates[merge_cursor][0]
            merged_amp += candidates[merge_cursor][1]
            count += 1
            merge_cursor += 1

        merged_time /= count
        merged_amp /= count

        compressed_amp = max(0.0, min(1.0, merged_amp)) ** AMPLITUDE_COMPRESSION_EXPONENT
        duration_ms = int(BASE_DURATION_MS + compressed_amp * (MAX_DURATION_MS - BASE_DURATION_MS))
        duration_ms = max(MIN_DURATION_MS, min(MAX_DURATION_MS, duration_ms))

        pulses.append(
            VibrationPulse(
                time_ms=int(round(merged_time)),
                intensity=max(MIN_PULSE_INTENSITY, min(1.0, compressed_amp)),
                duration_ms=duration_ms,
            )
        )

        last_time = base_time
        cursor = merge_cursor

    return sorted(pulses, key=lambda pulse: pulse.time_ms)


def synthesize_haptic_track(
    pulses: list[VibrationPulse],
    sample_rate: int,
    total_samples: int,
) -> list[float]:
    output = [0.0] * total_samples

    if not pulses:
        return output

    for pulse in pulses:
        start = max(0, int(pulse.time_ms * sample_rate / 1000))
        duration_samples = max(1, int(pulse.duration_ms * sample_rate / 1000))
        end = min(total_samples, start + duration_samples)
        attack = max(1, min(duration_samples // 4, int(sample_rate * ENVELOPE_TIME_SECONDS)))
        release = attack

        for sample_index in range(start, end):
            local_index = sample_index - start
            if local_index < attack:
                envelope = local_index / attack
            elif sample_index >= end - release:
                envelope = max(0.0, (end - sample_index) / release)
            else:
                envelope = 1.0

            output[sample_index] += pulse.intensity * envelope

    max_amplitude = max((abs(sample) for sample in output), default=0.0)
    if max_amplitude > 0.95:
        scale = 0.95 / max_amplitude
        output = [sample * scale for sample in output]

    return output


def create_android_haptic_ogg(
    ffmpeg: str,
    stereo_wav: Path,
    haptic_wav: Path,
    output_path: Path,
    quality: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run_command(
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(stereo_wav),
        "-i",
        str(haptic_wav),
        "-filter_complex",
        (
            # Use a native Vorbis 3-channel layout (FL, FR, FC) and carry haptics in
            # the third channel; Android discovers the haptic lane via ANDROID_HAPTIC=1.
            "[0:a][1:a]join=inputs=2:channel_layout=3.0:"
            "map=0.0-FL|0.1-FR|1.0-FC[aout]"
        ),
        "-map",
        "[aout]",
        "-c:a",
        "libvorbis",
        "-q:a",
        str(quality),
        "-metadata:s:a:0",
        "ANDROID_HAPTIC=1",
        str(output_path),
    )


def main() -> int:
    args = parse_args()
    ffmpeg = ensure_binary("ffmpeg")

    if not args.input.exists():
        raise SystemExit(f"Input file does not exist: {args.input}")

    if "--carrier-hz" in sys.argv:
        print(
            "Warning: --carrier-hz is ignored; the haptic track is now written as a raw envelope.",
            file=sys.stderr,
        )

    with tempfile.TemporaryDirectory(prefix="beatvibrator-haptics-") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        stereo_wav = temp_dir / "audio_stereo.wav"
        mono_wav = temp_dir / "analysis_mono.wav"
        haptic_wav = temp_dir / "haptic_channel.wav"

        decode_audio(ffmpeg, args.input, args.sample_rate, stereo_wav, mono_wav)
        sample_rate, mono_samples = read_wav_float_samples(mono_wav)

        filtered = apply_butterworth_low_pass(mono_samples, sample_rate, args.low_pass_cutoff)
        rms_values = compute_rms_over_windows(filtered, DEFAULT_RMS_WINDOW, DEFAULT_RMS_HOP)
        onsets = detect_onsets(
            rms_values=rms_values,
            threshold=DEFAULT_ONSET_THRESHOLD,
            min_interval_ms=DEFAULT_MIN_ONSET_INTERVAL_MS,
            sample_rate=sample_rate,
            hop_size=DEFAULT_RMS_HOP,
        )
        pulses = build_pulses(rms_values, onsets, sample_rate, DEFAULT_RMS_HOP)
        haptic_samples = synthesize_haptic_track(
            pulses=pulses,
            sample_rate=sample_rate,
            total_samples=len(mono_samples),
        )

        write_wav_float_samples(haptic_wav, sample_rate, haptic_samples)
        create_android_haptic_ogg(ffmpeg, stereo_wav, haptic_wav, args.output, args.quality)

        print(
            f"Generated {args.output} with {len(pulses)} pulses "
            f"at {sample_rate} Hz."
        )

        if args.keep_temp:
            kept_dir = args.output.parent / f"{args.output.stem}_debug"
            kept_dir.mkdir(parents=True, exist_ok=True)
            stereo_wav.replace(kept_dir / stereo_wav.name)
            mono_wav.replace(kept_dir / mono_wav.name)
            haptic_wav.replace(kept_dir / haptic_wav.name)
            print(f"Kept intermediate files in {kept_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
