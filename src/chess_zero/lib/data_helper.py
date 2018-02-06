import json
import os
from datetime import datetime
from glob import glob
import fnmatch
from logging import getLogger

import chess

from chess_zero.config import ResourceConfig

logger = getLogger(__name__)


def pretty_print(env, colors):
    new_pgn = open("test3.pgn", "at")
    game = chess.pgn.Game.from_board(env.board)
    game.headers["Result"] = env.result
    game.headers["White"], game.headers["Black"] = colors
    game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
    new_pgn.write(str(game) + "\n\n")
    new_pgn.close()


def find_pgn_files(directory, pattern='*.pgn'):
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))
    return files


def get_game_data_filenames(rc: ResourceConfig):
    pattern = os.path.join(rc.play_data_dir, rc.play_data_filename_tmpl % "*")
    files = list(sorted(glob(pattern)))
    return files


def get_next_generation_model_dirs(rc: ResourceConfig):
    dir_pattern = os.path.join(rc.next_generation_model_dir, rc.next_generation_model_dirname_tmpl % "*")
    dirs = list(sorted(glob(dir_pattern)))
    return dirs


def write_game_data_to_file(path, data):
    with open(path, "wt") as f:
        json.dump(data, f)


def read_game_data_from_file(path):
    with open(path, "rt") as f:
        return json.load(f)
