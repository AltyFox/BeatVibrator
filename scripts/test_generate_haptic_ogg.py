import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_haptic_ogg import VibrationPulse, synthesize_haptic_track


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


if __name__ == "__main__":
    unittest.main()
