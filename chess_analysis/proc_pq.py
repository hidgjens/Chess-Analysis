"""
:file: chess_analysis/proc_pq.py
:author: hidgjens
:created: 2023/07/01
:last modified: 2023/07/02

Read capture records produced by `piece_capture.py` and
generate summary plots showing aggregated results.

This file looks at the locations where captures occured 
and what pieces were involve.

It produces heatmaps indicating the percentage of captures
that happens on each square of the board.
"""
from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass
from typing import Final
from typing import Iterable
from typing import Literal
from typing import MutableMapping

import chess
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def plot_counts(
    counts: MutableMapping[str, int],
    *,
    piece: str = "",
    desc: str = "counts",
    norm: bool = True,
    flip_board: bool = False,
):
    """
    Make heatmap plot showing counts.

    :param counts:
        Mapping of square name to counts.
    :type counts:
        MutableMapping[str, int]
    :param piece:
        Name of piece being studied, defaults to ""
    :type piece:
        str, optional
    :param desc:
        Description for data being rendered, defaults to "counts"
    :type desc:
        str, optional
    :param norm:
        Normalise counts to percentages, defaults to True
    :type norm:
        bool, optional
    :param flip_board:
        Mirror the board, defaults to False
    :type flip_board:
        bool, optional
    """
    # Create a 64-element array for tracking the
    # counts for each square.
    # Search `counts` for an entry for each square
    # aand store in the counts array.
    counts_array = np.zeros(64)
    for index, square_name in enumerate(chess.SQUARE_NAMES):
        counts_array[index] = counts.get(square_name, 0)

    # Reshape the counts array from (64,) to (8,8)
    # to reflect the 2D chessboard structure.
    counts_array = counts_array.reshape((8, 8))

    # If requested, we will normalise the counts
    # array so that values represent percentage
    # of entries rather than total count.
    total_counts = round(np.sum(counts_array))
    if total_counts == 0:
        return
    if norm:
        counts_array = counts_array / total_counts

    # This if statement looks confusing.
    #
    # flip_board indicates whether we wanted to mirror
    # the board for visualisation, which we achieve
    # with np.flip(..., axis=0).
    #
    # But, we apply np.flip(..., axis=0) for
    # rendering with imshow to show the board the
    # right way up.
    #
    # So, if flip_board is true, we flip once for `flip_board`
    # and once for rendering (effectively not flipping at all).
    # If flip_board is false, we flip once only for rendering.
    # The end result is no flip if flip_board is true, and flip
    # if flip_board is false.
    if not flip_board:
        counts_array = np.flip(counts_array, axis=0)

    # Make figure and save to file.
    plt.imshow(counts_array)
    plt.colorbar()
    plt.title(f"{piece} ({desc}, counts={total_counts})")
    plt.tight_layout()
    plt.savefig(f"plots/{desc}_{piece}.png", bbox_inches="tight")
    plt.close()


@dataclass(frozen=True)
class PlotConfig:
    """Config class describing the data to be plotted."""

    piece_column: str
    """Which column to look in for piece name."""

    square_column: str
    """Which column to look in for square name."""

    desc: str
    """Description of this config."""

    reverse: bool
    """Whether to mirror the board when displaying."""


# PLOT_CONFIGS: Final = [
#     PlotConfig("piece", "square", "taken_at", reverse=False),
#     PlotConfig("piece", "from_square", "taken_from", reverse=True),
#     PlotConfig("capturing_piece", "square", "taking_at", reverse=False),
#     PlotConfig("capturing_piece", "from_square", "taking_from", reverse=False),
# ]
# TODO temp fix to get around bug in source data.
PLOT_CONFIGS: Final = [
    PlotConfig("piece", "square", "taken_at", reverse=True),
    PlotConfig("piece", "from_square", "taken_from", reverse=False),
    PlotConfig("capturing_piece", "square", "taking_at", reverse=False),
    PlotConfig("capturing_piece", "from_square", "taking_from", reverse=False),
]
"""List of plot configurations to run."""

PIECES: Final = [
    "pawn",
    "knight",
    "bishop",
    "rook",
    "queen",
    "king",
]
"""List of piece names."""

COLUMNS: Final = [
    "piece",
    "capturing_piece",
    "square",
    "from_square",
]
"""List of columns to read from the parquet files."""


def count_occurances_for_piece_with_config(
    df: pd.DataFrame,
    piece: str | Literal["all"],
    config: PlotConfig,
) -> Counter[str, int]:
    """Count occurances in squares using given config.

    :param df:
        Source data.
    :type df:
        pd.DataFrame
    :param piece:
        Name of piece, or 'all'.
    :type piece:
        str
    :param config:
        Config to plot.
    :type config:
        PlotConfig
    :return:
        Mapping of square name to occurance counts.
    :rtype:
        Counter[str, int]
    """
    # Setup counter for storing results.
    square_counts = Counter()

    # Filter rows by piece name, and count occurances of each
    # square name.
    # If "all" is given for piece, we do not need to filter
    # rows by piece name.
    if piece == "all":
        file_square_counts = df.loc[:, config.square_column].value_counts()
    else:
        piece_condition = df[config.piece_column] == piece
        file_square_counts = df.loc[
            piece_condition,
            config.square_column,
        ].value_counts()

    # Store counts in the counter and return the counter.
    for index in file_square_counts.index:
        square_counts[index] += file_square_counts[index]
    return square_counts


def count_occurances_for_piece(
    df: pd.DataFrame,
    piece: str,
    configs: Iterable[PlotConfig] = ...,
) -> dict[tuple[str, str], Counter]:
    """Count occurances in squares for each given config.
    If no configs are given, all configs are used.

    :param df:
        Source data.
    :type df:
        pd.DataFrame
    :param piece:
        Name of piece, or 'all'.
    :type piece:
        str
    :param configs:
        List of configs to use, defaults to ...
    :type configs:
        Iterable[PlotConfig], optional
    :return:
        Mapping of (piece, config) to results counter.
    :rtype:
        dict[tuple[str, str], Counter]
    """
    # If no value is given for configs, use the full list.
    if configs is Ellipsis:
        configs = PLOT_CONFIGS

    # Process each of the configs and store each configs
    # results in a dictionary, using (piece, config) as
    # the key.
    results = {}
    for config in configs:
        counts = count_occurances_for_piece_with_config(
            df=df,
            piece=piece,
            config=config,
        )
        results[(piece, config)] = counts
    return results


def main(norm: bool = True):
    # Create plot directory.
    os.makedirs("plots", exist_ok=True)

    # Use pyarrow to detect the parquet dataset.
    path = "data\output_v4.parquet"
    dataset = pq.ParquetDataset(path)
    print(f"Loaded {len(dataset.files)} files")

    # In addition to individual piece, we will process all pieces.
    all_pieces = PIECES + ["all"]

    # For each element in all pieces, we will generate plots using
    # config in PLOT_CONFIGS.
    #
    # These plots will show the number of captures that occur in
    # each square.
    #
    # We will store a mapping of (piece, config) to counters.
    all_counts = {(p, c): Counter() for p in all_pieces for c in PLOT_CONFIGS}

    # To save my precious RAM, we will batch the data file by file.
    # Each file will be processed to get the square counts for each
    # piece.
    # We will then aggregate the results from each file in `all_counts`.
    for pq_file in dataset.files:
        # Read parquet file to pd.DataFrame.
        print(f"Reading {pq_file}")
        df = pd.read_parquet(pq_file, columns=COLUMNS)

        # For each piece, process all configs.
        # Store the returned results in all_counts.
        for piece in all_pieces:
            print(f"Counting for {piece=}")
            counts = count_occurances_for_piece(
                df=df,
                piece=piece,
                configs=PLOT_CONFIGS,
            )

            # Returned `counts` is a dict mapping (piece, config)
            # to Counter instances.
            # We can use the keys from the `counts` dict to insert
            # into the `all_counts` dict using the Counter.update()
            # function.
            for key, counter in counts.items():
                all_counts[key].update(counter)

    # All counts have been aggregated, now we can
    # generate plots for each of the generated counters.
    for (piece, config), counts in all_counts.items():
        print(f"Plotting for {piece=} {config.desc=}")
        plot_counts(
            counts,
            piece=piece,
            desc=config.desc,
            norm=norm,
            flip_board=config.reverse,
        )


if __name__ == "__main__":
    main()
