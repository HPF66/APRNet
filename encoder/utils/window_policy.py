"""Shared window enumeration policy for Peptide Teacher caches."""

from __future__ import annotations


def enumerate_windows(
    sequence: str, lengths: tuple[int, ...], stride: int
) -> list[tuple[int, int, str, int]]:
    """Create full-coverage windows, including a real-length short fallback."""
    if not sequence:
        raise ValueError("sequence must not be empty")
    if not lengths or any(length <= 0 for length in lengths):
        raise ValueError("invalid window lengths")
    if not 0 < stride <= min(lengths):
        raise ValueError("stride must be in [1, minimum window length]")

    windows: list[tuple[int, int, str, int]] = []
    for scale_index, window_length in enumerate(lengths):
        if window_length > len(sequence):
            continue
        starts = list(range(0, len(sequence) - window_length + 1, stride))
        final_start = len(sequence) - window_length
        if starts[-1] != final_start:
            starts.append(final_start)
        windows.extend(
            (start, window_length, sequence[start : start + window_length], scale_index)
            for start in starts
        )

    if not windows:
        # Keep the real sequence and length. Scale 0 is the closest available
        # scale slot, while the downstream length feature distinguishes <6 aa.
        windows.append((0, len(sequence), sequence, 0))
    return windows
