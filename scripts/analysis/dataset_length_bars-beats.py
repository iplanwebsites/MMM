#!/usr/bin/python3 python

"""Measuring the length of GigaMIDI files in bars and beats."""

if __name__ == "__main__":
    from pathlib import Path
    from random import seed, shuffle

    import numpy as np
    from matplotlib import pyplot as plt
    from miditok.constants import SCORE_LOADING_EXCEPTION
    from miditok.utils import get_bars_ticks, get_beats_ticks
    from symusic import Score
    from tqdm import tqdm

    SEED = 777
    NUM_FILES = 5000
    NUM_HIST_BINS = 50
    X_AXIS_LIM_BARS = 500
    X_AXIS_LIM_BEATS = X_AXIS_LIM_BARS * 4
    seed(SEED)

    # Gather files
    files_paths = list(Path("..", "..", "data", "GigaMIDI").resolve().glob("**/*.mid"))
    shuffle(files_paths)

    # Measuring lengths in bars/beats
    dist_lengths_bars, dist_lengths_beats = [], []
    idx = 0
    with tqdm(total=NUM_FILES, desc="Measuring files lengths") as pbar:
        while len(dist_lengths_bars) < NUM_FILES and idx < len(files_paths):
            try:
                score = Score(files_paths[idx])
                dist_lengths_bars.append(len(get_bars_ticks(score)))
                dist_lengths_beats.append(len(get_beats_ticks(score)))
            except SCORE_LOADING_EXCEPTION:
                pass
            finally:
                idx += 1
                pbar.update(1)

    # Filter high durations (will bias real densities!)
    dist_lengths_bars = np.array(dist_lengths_bars)
    dist_lengths_bars = dist_lengths_bars[
        np.where(dist_lengths_bars <= X_AXIS_LIM_BARS)[0]
    ]
    dist_lengths_beats = np.array(dist_lengths_beats)
    dist_lengths_beats = dist_lengths_beats[
        np.where(dist_lengths_beats <= X_AXIS_LIM_BEATS)[0]
    ]

    # Plotting the distributions
    fig, ax = plt.subplots()
    ax.hist(dist_lengths_bars, bins=NUM_HIST_BINS)
    ax.grid(axis="y", linestyle="--", linewidth=0.6)
    ax.set_ylabel("Count files")
    ax.set_xlabel("Length in bars")
    fig.savefig(Path("GigaMIDI_length_bars"), bbox_inches="tight", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.hist(dist_lengths_beats, bins=NUM_HIST_BINS)
    ax.grid(axis="y", linestyle="--", linewidth=0.6)
    ax.set_ylabel("Count files")
    ax.set_xlabel("Length in beats")
    fig.savefig(Path("GigaMIDI_length_beats"), bbox_inches="tight", dpi=300)
    plt.close(fig)
