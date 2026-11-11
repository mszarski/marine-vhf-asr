"""Tests for the energy-gate transmission cutter."""

import numpy as np

from marine_vhf_asr.segment import find_transmissions

SR = 16000


def _recording(bursts, total=30.0, noise=0.001, seed=0):
    rng = np.random.default_rng(seed)
    x = noise * rng.standard_normal(int(total * SR))
    for a, b in bursts:
        x[int(a * SR):int(b * SR)] += 0.2 * rng.standard_normal(int(b * SR) - int(a * SR))
    return x.astype(np.float32)


class TestFindTransmissions:
    def test_finds_bursts(self):
        found = find_transmissions(_recording([(2, 6), (10, 13)]), SR)
        assert len(found) == 2
        assert abs(found[0][0] - 2) < 0.05 and abs(found[0][1] - 6) < 0.05

    def test_merges_short_gaps(self):
        found = find_transmissions(_recording([(2, 4), (4.3, 6)]), SR)
        assert len(found) == 1 and found[0][1] - found[0][0] > 3.9

    def test_drops_too_short_and_too_long(self):
        assert find_transmissions(_recording([(2, 2.5), (5, 28)]), SR) == []

    def test_silence(self):
        assert find_transmissions(_recording([]), SR) == []
        assert find_transmissions(np.zeros(0, dtype=np.float32), SR) == []
