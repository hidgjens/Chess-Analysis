"""
:file: chess_analysis/piece_capture.py
:author: hidgjens
:created: 2023/07/01
:last modified: 2023/07/02

Analyses games from pgn files and records all captures made.

usage: python piece_capture.py [-h] [-D [DATA_DIR]] [-G [GAMELIMIT]] [-F [FILELIMIT]]
Use `python piece_capture.py --help` for full usage options. 


The pgn files provided by Lichess are big files which contain
a month's worth of ranked chess games.
Due to their size, they need to be processed in batches.

This script will process all pgn files found in a directory.
Limits can be set for the maximum number of games processed per
file, and the maximum number of files processed.

Output data is stored in a parquet database to be processed by
proc_pq.py.


"""

from __future__ import annotations

import os
import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any
from typing import Generator
from typing import Iterable

import chess.pgn
import pyarrow as pa
import pyarrow.parquet as pq
from chess import Board
from chess import Move
from chess import piece_name
from chess import square_name
from chess.pgn import Game
from tqdm import tqdm
from util import Timer

# ======================
# Create capture records
# ======================


@dataclass
class CA_Capture:
    """
    A dataclass for storing data about each piece capture.

    The `Capture.get_data()` method can be used to construct
    DataFrame rows. Header names can be accessed from
    `Capture.get_headers()`, and a PyArrow schema can be accessed
    from `Capture.get_schema()`.
    """

    captured_at_square: str
    """Name of the square where piece was captured."""

    captured_from_square: str
    """Name of the square which the capturing piece came from."""

    captured_at_square_mirrored: str
    """Name of the square where piece was captured (mirrored board)."""

    captured_from_square_mirrored: str
    """Name of the square which the capturing piece came from (mirrored board)."""

    captured_piece: str
    """Name of captured piece."""

    capturing_piece: str
    """Name of capturing piece."""

    white_piece: bool
    """Whether the captured piece was white."""

    game_id: str = ""
    """ID of the game according to lichess."""

    elo: float = 0
    """Mean ELO of the two players."""

    opening_full: str = ""
    """Full name of the opening. Combination of family and variant."""

    opening_family: str = ""
    """What family of openings was used (e.g., Italian game)."""

    opening_variant: str = ""
    """What opening variant was used."""

    time_remaining_secs: float = 0
    """Time remaining for capturing player."""

    time_control: str = ""
    """What time control the game used."""

    date: str = ""
    """Date of the game. (YYYY.MM.DD)"""

    year: int = 0
    """Year of the date."""

    month: int = 0
    """Month of the date."""

    day: int = 0
    """Day of the date."""

    @staticmethod
    def get_schema() -> pa.Schema:
        """
        Get a PyArrow schema for the data in this class.

        :return:
            PyArrow Schema
        :rtype:
            pa.Schema
        """
        schema = {
            "captured_at_square": pa.string(),
            "captured_from_square": pa.string(),
            "captured_at_square_mirrored": pa.string(),
            "captured_from_square_mirrored": pa.string(),
            "captured_piece": pa.string(),
            "capturing_piece": pa.string(),
            "white_piece": pa.bool_(),
            "game_id": pa.string(),
            "elo": pa.float32(),
            "opening_full": pa.string(),
            "opening_family": pa.string(),
            "opening_variant": pa.string(),
            "time_remaining_secs": pa.float32(),
            "time_control": pa.string(),
            "date": pa.string(),
            "year": pa.uint16(),
            "month": pa.uint8(),
            "day": pa.uint8(),
        }
        return pa.schema(schema)

    @staticmethod
    def get_headers() -> list[str]:
        """
        Get a list of header names, these are in the same
        order as the data presented in `Capture.get_data()`.

        :return:
            list of header names
        :rtype:
            list[str]
        """
        return [
            "captured_at_square",
            "captured_from_square",
            "captured_at_square_mirrored",
            "captured_from_square_mirrored",
            "captured_piece",
            "capturing_piece",
            "white_piece",
            "game_id",
            "elo",
            "opening_full",
            "opening_family",
            "opening_variant",
            "time_remaining_secs",
            "time_control",
            "date",
            "year",
            "month",
            "day",
        ]

    def get_data(self) -> list[Any]:
        """
        Get all the data in this instance as a list.
        See `Capture.get_headers()` for names of each
        element.

        :return:
            This instances data as a list.
        :rtype:
            list[Any]
        """
        return [
            self.captured_at_square,
            self.captured_from_square,
            self.captured_at_square_mirrored,
            self.captured_from_square_mirrored,
            self.captured_piece,
            self.capturing_piece,
            self.white_piece,
            self.game_id,
            self.elo,
            self.opening_full,
            self.opening_family,
            self.opening_variant,
            self.time_remaining_secs,
            self.time_control,
            self.date,
            self.year,
            self.month,
            self.day,
        ]


def get_captured_piece(board: Board, move: Move) -> CA_Capture | None:
    """
    Check if a piece is capture with given move.
    If no capture is made, None is return.
    If there was a capture, a `Capture` instance is returned
    for storing in a table.

    :param board:
        Current board state (pre-move).
    :type board:
        Board
    :param move:
        The move to be made.
    :type move:
        Move
    :return:
        A `Capture` instance if a piece was captures, None otherwise.
    :rtype:
        Capture | None
    """
    if board.is_capture(move):
        # A piece was captured, see whose turn it is.
        # The colour of the captured piece is the opposite player.
        is_white_turn = board.turn
        is_white_piece = not is_white_turn

        # If it is a black piece that was captured, mirror the
        # board so that everything is measured from the player's
        # perspective.
        #
        # Mirrored is from the capturers POV.
        from_square_int = move.from_square
        if not is_white_piece:
            from_square_int = chess.square_mirror(from_square_int)
        from_square = square_name(from_square_int)
        from_square_mirrored = square_name(chess.square_mirror(from_square_int))

        # Get the name of the piece making the capture.
        capturing_piece_int = board.piece_at(move.from_square).piece_type
        capturing_piece = piece_name(capturing_piece_int)

        # We will need to treat en passant differently since the
        # destination square is not where the captured piece was
        # taken from.
        if board.is_en_passant(move):
            # Only pawns can be captured with en passant.
            captured_piece = piece_name(chess.PAWN)

            # The capturing pawn ends up one square above
            # the captured pawn.
            # To get the location of the captured piece, we need
            # to go back one rank (by subtracting 8 from the square
            # number).
            # As before, we were mirror the board if it is
            # a white piece being taken.
            at_square_int = move.to_square
            if not is_white_piece:
                at_square_int = chess.square_mirror(at_square_int)
            at_square_int -= 8
            at_square = square_name(at_square_int)
        else:
            # Find the name of the piece at the square where the
            # capture was made.
            at_square_int = move.to_square
            captured_piece = piece_name(board.piece_at(at_square_int).piece_type)

            # As before, mirror the board if it is a white piece
            # being taken.
            if not is_white_piece:
                at_square_int = chess.square_mirror(at_square_int)
            at_square = square_name(at_square_int)

        # Create mirror for captured square (for capturer's POV).
        at_square_mirror = square_name(chess.square_mirror(at_square_int))

        # Sometimes captured_piece is "null", so we skip these.
        # I also skip "king" although I don't expect this to show up.
        if captured_piece == "null" or captured_piece == "king":
            return None

        # Return capture record.
        return CA_Capture(
            captured_at_square=at_square,
            captured_at_square_mirrored=at_square_mirror,
            captured_from_square=from_square,
            captured_from_square_mirrored=from_square_mirrored,
            captured_piece=captured_piece,
            capturing_piece=capturing_piece,
            white_piece=is_white_piece,
        )
    return None


_NUMBER_OPENING = r"(.+) #(\d+)"
_VARIANT_OPENING = r"(.+): (.+)"


def find_opening_family_variant(opening: str) -> tuple[str, str]:
    """
    Split an opening name into 'family' and 'variant'.

    For example, "Opening family: variant name" would
    return ('Opening family', 'variant name').

    :param opening:
        Name of opening.
    :type opening:
        str
    :return:
        Two string (family, variant).
    :rtype:
        tuple[str, str]
    """
    # Check for pattern "family: variant"
    match_ = re.match(_VARIANT_OPENING, opening)
    if match_ is not None:
        family = match_.group(1)

        variant = match_.group(2)
        return family, variant

    # Check for pattern "family #variant"
    match_ = re.match(_NUMBER_OPENING, opening)
    if match_ is not None:
        family = match_.group(1)
        variant = match_.group(2)
        return family, variant

    # If no matches are found, return the opening
    # name as the family, and variant as blank.
    return opening, ""


# ======================
# Process Game Functions
# ======================

_DATE_REGEX = r"(\d+)\.(\d+)\.(\d+)"


def parse_date(date: str) -> tuple[int, int, int]:
    """
    Parse date string in format "YYYY-MM-DD" into
    a tuple of three ints (year, month, day).

    :param date:
        Date string (formatted "YYYY-MM-DD").
    :type date:
        str
    :return:
        (year, month, day) as integers.
    :rtype:
        tuple[int, int, int]
    """
    # Use regex to parse date string.
    match_ = re.match(_DATE_REGEX, date)
    if match_ is None:
        # No match is found, return all zeros.
        return 0, 0, 0
    else:
        # Cast groups to into and return.
        year = int(match_.group(1))
        month = int(match_.group(2))
        day = int(match_.group(3))
        return year, month, day


_GAME_ID_REGEX = r"https:\/\/lichess\.org\/(\w+)"


def get_game_id(url: str) -> str:
    """
    Use regex to parse URL string and retrieve game ID.

    :param url:
        Game URL.
    :type url:
        str
    :return:
        Detected game ID.
    :rtype:
        str
    """
    re_match = re.match(_GAME_ID_REGEX, url)
    if re_match is None:
        return ""
    else:
        return re_match.group(1)


def proc_game(game: Game) -> list[list[Any]]:
    """
    Process a chess game to find all the captures that occur.

    :param game:
        Chess game to process.
    :type game:
        Game
    :return:
        List of capture data, each element stores a row
        of capture data (see `CA_Capture`).
    :rtype:
        list[list[Any]]
    """
    # Get all the column values which are constant within a game.
    # Take the average ELO for the players.
    elo_white = float(game.headers.get("WhiteElo", 0))
    elo_black = float(game.headers.get("BlackElo", 0))
    elo_mid = 0.5 * (elo_white + elo_black)

    # Get the date of the game, and parse (year, month, day)
    # to integers.
    date = game.headers["UTCDate"]
    year, month, day = parse_date(date)

    # The time control used for this game.
    time_control = game.headers["TimeControl"]

    # The opening used (split into family and variant if possible).
    opening = game.headers["Opening"]
    family, variant = find_opening_family_variant(opening)

    # Get the match ID by parsing the game URL.
    url = game.headers["Site"]
    game_id = get_game_id(url)

    # Iterate over mainline moves and check for captures.
    board = game.board()
    results = []
    for node in game.mainline():
        # Get remaining time for player making the move.
        remaining_time = node.clock()

        # Check if a piece was captured.
        captured_piece = get_captured_piece(board, node.move)
        if captured_piece is not None:
            # Add the additional info that was calculated
            # earlier.
            captured_piece.game_id = game_id
            captured_piece.elo = elo_mid
            captured_piece.opening_full = opening
            captured_piece.opening_family = family
            captured_piece.opening_variant = variant
            captured_piece.time_remaining_secs = remaining_time
            captured_piece.time_control = time_control
            captured_piece.date = date
            captured_piece.year = year
            captured_piece.month = month
            captured_piece.day = day
            # Store the data from the capture record.
            results.append(captured_piece.get_data())

        # Update the board with the move.
        board.push(node.move)
    return results


def load_game(
    filepath: str,
    location: int,
) -> Game | None:
    """
    Load a game from a given location in a pgn file
    and return chess.Game instance (or None if no
    game was found).

    :param filepath:
        Path to pgn file.
    :type filepath:
        str
    :param location:
        Location in file to start search for game.
    :type location:
        int
    :return:
        The loaded game (returns None if no game was found).
    :rtype:
        Game | None
    """
    with open(filepath, "r") as in_file:
        in_file.seek(location)
        game = chess.pgn.read_game(in_file)
        return game


# ==================
# PGN File functions
# ==================


def worker(loc_path: tuple[int, str]) -> list[Any]:
    """
    Thread worker which finds all the captures
    in a particular game.

    :param loc_path:
        A tuple indicating game location and pgn file path (location, path).
    :type loc_path:
        tuple[int, str]
    :return:
        A list of capture data.
    :rtype:
        list[Any]
    """
    # Split loc_path into location and path, and load the game.
    location, filepath = loc_path
    game = load_game(filepath, location)

    # If no game was found, return no capture records,
    # otherwise process the game and return the capture records.
    if game is None:
        return []
    else:
        return proc_game(game)


def find_games_locations(
    filepath: str,
    limit: int | None = None,
) -> Generator[int, None, None]:
    """
    Search a pgn file and return all the starting locations
    for the games.

    :param filepath:
        Path to pgn file.
    :type filepath:
        str
    :param limit:
        Maximum number of games to return, defaults to None
    :type limit:
        int | None, optional
    :yield:
        Starting position of a game.
    :rtype:
        Generator[int, None, None]
    """
    # File always starts with a game.
    yield 0
    total = 1
    with open(filepath, "r") as in_file:
        line = in_file.readline()
        while line:
            if line.startswith("1."):
                position = in_file.tell()
                yield position
                if limit is not None:
                    total += 1
                    if total >= limit:
                        break
            line = in_file.readline()


def get_record_batches(
    results: Iterable[Iterable[CA_Capture]],
    batch_size: int = 128 * 1024,
) -> Generator[pa.RecordBatch, None, None]:
    """
    Batch the list of capture records into RecordBatches.
    Used for constructing a PyArrow table.

    :param results:
        Iterable to an iterable of capture instances
        (first iterable is for each game, second for
        each capture in each game).
    :type results:
        Iterable[Iterable[CA_Capture]]
    :param batch_size:
        Number of captures records to include in a batch,
        defaults to 128*1024.
    :type batch_size:
        int, optional
    :yield:
        A batch of capture records.
    :rtype:
        Generator[pa.RecordBatch, None, None]
    """
    # Create a 1D iterable for collecting all capture results
    # from all games.
    all_captures = tqdm(
        (capture for game in results for capture in game),
        desc="Collecting capture results",
        unit=" captures",
    )

    # Get headers and schema for capture records.
    headers = CA_Capture.get_headers()
    schema = CA_Capture.get_schema()

    # Create a dict which maps column names to arrays of data.
    # This allows us to construct a RecordBatch using from_pydict.
    data = {c: [] for c in headers}

    # Iterate over all captures, counting how many records we
    # have collected.
    # Once batch size is hit, we yield the RecordBatch and
    # start a new batch.
    for n_captures, capture in enumerate(all_captures):
        for value, header in zip(capture, headers):
            data[header].append(value)
        if n_captures % batch_size == 0:
            yield pa.RecordBatch.from_pydict(data, schema=schema)
            data = {c: [] for c in headers}
    # Check if there are any elements still in the data dict
    # (this occurs when the last batch didn't reach the batch
    # limit).
    # If there is data, yield one more batch.
    if data[headers[0]]:
        yield pa.RecordBatch.from_pydict(data, schema=schema)


def combine_results(results: Iterable[Iterable[CA_Capture]]) -> pa.Table:
    """
    Combine capture records into a PyArrow table.

    :param results:
        Iterable to an iterable of capture instances
        (first iterable is for each game, second for
        each capture in each game).
    :type results:
        Iterable[Iterable[CA_Capture]]
    :return:
        A PyArrow table containing all the capture records.
    :rtype:
        pa.Table
    """
    schema = CA_Capture.get_schema()
    with Timer("Creating table"):
        table = pa.Table.from_batches(
            iter(get_record_batches(results)),
            schema=schema,
        )
    return table


_YEAR_MONTH_REGEX = r"lichess_db_standard_rated_(\d+)-(\d+).pgn"


def get_year_month(base_name: str) -> tuple[int, int]:
    """
    Extract the year and month from a lichess database
    filename.

    :param base_name:
        Name of lichess database file.
    :type base_name:
        str
    :return:
        A tuple of ints containing (year, month).
    :rtype:
        tuple[int, int]
    """
    # Use regex to extract year and month.
    # If no match is found, return zeros.
    match_ = re.match(_YEAR_MONTH_REGEX, base_name)
    if match_ is None:
        return 0, 0

    # Extract year and month and cast to int.
    year = match_.group(1)
    month = match_.group(2)
    return int(year), int(month)


def divide_chunks(
    full_list: Iterable[Any],
    chunk_size: int,
) -> Generator[list[Any], None, None]:
    """
    Split an iterable up into chunks of size `chunk_size`.

    :param full_list:
        Elements to be divided into chunks.
    :type full_list:
        Iterable[Any]
    :param chunk_size:
        Number of elements per chunk.
    :type chunk_size:
        int
    :yield:
        A chunk of elements.
    :rtype:
        Generator[list[Any], None, None]
    """
    # Create a list to store the current chunk.
    current_chunk = []

    # Iterate over full list of elements.
    # Store each element in the chunk until chunk_size
    # is reached.
    for item in full_list:
        current_chunk.append(item)

        # Check chunk size. Yield if limit is hit and reset
        # chunk.
        if len(current_chunk) == chunk_size:
            yield current_chunk
            current_chunk = []

    # If there is still data in the chunk, yield the chunk.
    if current_chunk:
        yield current_chunk


_DEFAULT_WORKERS = os.cpu_count() - 2


def proc_file(
    pgn_file: str,
    *,
    limit: int | None = None,
    workers: int = _DEFAULT_WORKERS,
    chunk_size: int = 200_000,
):
    """
    Process a pgn file of games.

    :param pgn_file:
        Path to pgn file.
    :type pgn_file:
        str
    :param limit:
        Maximum number of games to process, defaults to None.
    :type limit:
        int | None, optional
    :param workers:
        Maximum number of workers for thread pool, defaults
        to _DEFAULT_WORKERS.
    :type workers:
        int, optional
    :param chunk_size:
        Size of chunks to process games in, defaults to 200_000
    :type chunk_size:
        int, optional
    """
    # Get the name of the pgn file, and extract the year and month
    # of the database.
    base_name = os.path.basename(pgn_file)
    year, month = get_year_month(base_name)

    # Find where each game starts in the pgn file.
    # Pair up the game locations with the filepath for the
    # worker function.
    game_locations = find_games_locations(pgn_file, limit=limit)
    game_paths = ((location, pgn_file) for location in game_locations)

    # Split the games into chunks, so that the whole file
    # isn't processed at once.
    chunks = divide_chunks(game_paths, chunk_size)

    # Process the file chunk by chunk.
    for chunk in chunks:
        # Use a process pool to process the games in parallel.
        with ProcessPoolExecutor(max_workers=workers) as pool:
            # The worker will process each game in the pgn file.
            results = pool.map(
                worker,
                tqdm(
                    chunk,
                    desc=f"Proc {year}-{month}",
                    total=chunk_size,
                    unit=" games",
                ),
            )
            # Take the capture records and combine into a single table.
            table = combine_results(results)

        # Construct the path for the parquet data.
        dir_name = os.path.dirname(pgn_file)
        table_path = os.path.join(
            dir_name,
            "output_v5.parquet",
        )

        # Write the generated table to parquet.
        with Timer("Writing parquet"):
            pq.write_to_dataset(
                table,
                root_path=table_path,
                partition_cols=["year", "month"],
            )

    # To prevent this file from being reprocessed, we will
    # append ".done" to the filename so it won't be detected.
    os.rename(pgn_file, pgn_file + ".done")


# ====
# Main
# ====


def process_dir(
    data_dir: str,
    limit: int | None = 1_000_000,
    file_limit: int | None = None,
):
    """
    Process all pgn files in a directory.

    :param data_dir:
        Path to directory containing pgn files.
    :type data_dir:
        str
    :param limit:
        Maximum number of games per pgn file to process,
        defaults to 2_000_000.
    :type limit:
        int | None, optional
    :param file_limit:
        Maximum number of pgn files to process, defaults
        to None.
    :type file_limit:
        int | None, optional
    """
    # Search for pgn files in data_dir.
    pgn_files = [f for f in os.listdir(data_dir) if os.path.splitext(f)[1] == ".pgn"]
    print(f"Found {len(pgn_files)} .pgn file(s)")

    # If there is a file limit, apply it.
    if file_limit:
        print(f"Limiting to {file_limit}")
        pgn_files = pgn_files[:file_limit]

    # Process each file.
    for file in pgn_files:
        fullpath = os.path.join(data_dir, file)
        with Timer(file):
            proc_file(fullpath, limit=limit)


if __name__ == "__main__":
    # Create arg parser and parse arguments.
    import argparse
    import os
    import sys

    parser = argparse.ArgumentParser(
        prog=os.path.basename(__file__),
        description="Process pgn files to collect data on piece captures.",
    )
    parser.add_argument(
        "-D",
        "--data",
        default="./data",
        nargs="?",
        help="Directory containing pgn data.",
        dest="data_dir",
    )
    parser.add_argument(
        "-G",
        "--gamelimit",
        default="2_000_000",
        nargs="?",
        help="Maximum number of games to process per pgn file. Can be set to None.",
        dest="gamelimit",
    )
    parser.add_argument(
        "-F",
        "--filelimit",
        default="None",
        nargs="?",
        help="Maximum number of pgn files to process.",
        dest="filelimit",
    )
    parsed = parser.parse_args(sys.argv[1:])

    # Source directory for pgn data.
    data_dir = parsed.data_dir

    # Maximum number of games to process per pgn file.
    if parsed.game_limit == "None":
        limit = None
    else:
        limit = int(parsed.game_limit)

    # Maximum number of pgn files to process.
    if parsed.file_limit == "None":
        file_limit = None
    else:
        file_limit = int(parsed.file_limit)

    process_dir(data_dir=data_dir, limit=limit, file_limit=file_limit)
