import os
from logging import getLogger
from random import random
from time import sleep
import chess
from chess_zero.agent.model_chess import ChessModel
from chess_zero.agent.player_chess import ChessPlayer
from chess_zero.config import Config
from chess_zero.env.chess_env import ChessEnv, Winner
from chess_zero.lib import tf_util
from chess_zero.lib.data_helper import get_next_generation_model_dirs
from chess_zero.lib.model_helper import save_as_best_model, load_best_model_weight

logger = getLogger(__name__)


class EvaluateWorker:
    def __init__(self, config: Config):
        """

        :param config:
        """
        self.config = config
        self.best_model = None

    def start(self):
        ng_model = None
        model_dir = None
        ng_model, model_dir = self.load_next_generation_model()
        save_as_best_model(ng_model)
        self.remove_model(model_dir)

    def load_next_generation_model(self):
        rc = self.config.resource
        while True:
            dirs = get_next_generation_model_dirs(self.config.resource)
            if dirs:
                break
            logger.info(f"There is no next generation model to evaluate")
            sleep(60)
        model_dir = dirs[-1] if self.config.eval.evaluate_latest_first else dirs[0]
        config_path = os.path.join(model_dir, rc.next_generation_model_config_filename)
        weight_path = os.path.join(model_dir, rc.next_generation_model_weight_filename)
        model = ChessModel(self.config)
        model.load(config_path, weight_path)
        return model, model_dir

    def remove_model(self, model_dir):
        rc = self.config.resource
        config_path = os.path.join(model_dir, rc.next_generation_model_config_filename)
        weight_path = os.path.join(model_dir, rc.next_generation_model_weight_filename)
        os.remove(config_path)
        os.remove(weight_path)
        os.rmdir(model_dir)
