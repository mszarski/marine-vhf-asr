"""
Cut continuous radio recordings into transmissions with an energy gate.

Scanner feeds are squelched or sit on a steady noise floor, so a threshold a
fixed margin above the recording's own floor finds push-to-talk bursts without
any speech model (a model-based cut would only keep what that model can hear).
"""

from typing import List, Tuple

import numpy as np


def find_transmissions(x: np.ndarray, sr: int, min_len: float = 2.0, max_len: float = 20.0,
                       gap: float = 0.6, margin_db: float = 12.0, frame: float = 0.02) -> List[Tuple[float, float]]:
    """
    Return (start, end) times in seconds of bursts louder than the noise floor.

    Args:
        min_len, max_len: keep bursts within this duration range
        gap: bursts closer than this are one transmission
        margin_db: threshold above the 10th-percentile frame level (never below -50 dB)
    """
    n = int(sr * frame)
    frames = x[:len(x) // n * n].reshape(-1, n)
    if not len(frames):
        return []
    db = 10 * np.log10((frames ** 2).mean(axis=1) + 1e-10)
    active = np.append(db > max(np.percentile(db, 10) + margin_db, -50.0), False)
    bursts, start = [], None
    for i, a in enumerate(active):
        if a and start is None:
            start = i
        elif not a and start is not None:
            if bursts and start * frame - bursts[-1][1] < gap:
                bursts[-1][1] = i * frame
            else:
                bursts.append([start * frame, i * frame])
            start = None
    return [(a, b) for a, b in bursts if min_len <= b - a <= max_len]
