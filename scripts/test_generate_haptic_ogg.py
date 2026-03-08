import sys
import unittest
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parent))

import generate_haptic_ogg
from generate_haptic_ogg import VibrationPulse, create_android_haptic_ogg, synthesize_haptic_track


class SynthesizeHapticTrackTests(unittest.TestCase):
    def test_synthesizes_positive_envelope(self) -> None:
        samples = synthesize_haptic_track(
            pulses=[VibrationPulse(time_ms=0, intensity=0.75, duration_ms=40)],
            sample_rate=48_000,
            total_samples=4_800,
        )

        self.assertGreater(max(samples), 0.0)
        self.assertGreater(sum(1 for sample in samples if sample > 0.0), 0)
        self.assertFalse(any(sample < 0.0 for sample in samples))
        self.assertTrue(all(sample == 0.0 for sample in samples[1920:]))

    def test_returns_silence_for_empty_pulses(self) -> None:
        samples = synthesize_haptic_track(
            pulses=[],
            sample_rate=48_000,
            total_samples=128,
        )

        self.assertEqual(samples, [0.0] * 128)


class CreateAndroidHapticOggTests(unittest.TestCase):
    def test_maps_haptics_into_third_channel(self) -> None:
        captured_args: tuple[str, ...] | None = None

        def capture_run_command(*args: str) -> None:
            nonlocal captured_args
            captured_args = args

        with patch.object(generate_haptic_ogg, "run_command", side_effect=capture_run_command):
            create_android_haptic_ogg(
                ffmpeg="ffmpeg",
                stereo_wav=Path("/tmp/audio.wav"),
                haptic_wav=Path("/tmp/haptic.wav"),
                output_path=Path("/tmp/output.ogg"),
                quality=6,
            )

        self.assertIsNotNone(captured_args)
        self.assertIn(
            "[0:a][1:a]join=inputs=2:channel_layout=3.0:map=0.0|0.1|1.0[aout]",
            captured_args,
        )
        self.assertIn("ANDROID_HAPTIC=1", captured_args)


if __name__ == "__main__":
    unittest.main()
