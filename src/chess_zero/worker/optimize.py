import os
from datetime import datetime
from logging import getLogger
from time import sleep

import json
from sklearn.cross_validation import KFold

import keras.backend as k
import numpy as np
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, Callback

from chess_zero.agent.model_chess import ChessModel, objective_function_for_policy, \
    objective_function_for_value, log_loss
from chess_zero.config import Config
from chess_zero.lib import tf_util
from chess_zero.lib.data_helper import get_game_data_filenames, read_game_data_from_file, \
    get_next_generation_model_dirs
from chess_zero.lib.model_helper import load_best_model_weight
from chess_zero.env.chess_env import ChessEnv
import chess

logger = getLogger(__name__)


def start(config: Config):
    tf_util.set_session_config(per_process_gpu_memory_fraction=1)
    return OptimizeWorker(config).start()


class OptimizeWorker(Callback):
    def __init__(self, config: Config):
        self.config = config
        self.model = None  # type: ChessModel
        self.loaded_filenames = set()
        self.loaded_data = {}
        self.dataset = None
        self.optimizer = None
        self.file = config
        self.metadata = {
            "total_epochs": 0,
            "best_model" : 1,
            "best_epoch": 0,
            "epochs": []
        }

    def on_epoch_end(self, epoch, logs={}):
        # in case appending to logs (resuming training), get epoch number ourselves
        epoch = len(self.metadata["epochs"])
        self.metadata["epochs"].append(logs)

        if "val_loss" in logs:
            key = "val_loss"
        else:
            key = "loss"

        best_loss = self.metadata["epochs"][self.metadata["best_epoch"]][key]
        if logs.get(key) < best_loss:
            self.metadata["best_epoch"] = epoch
            self.metadata["best_model"] = epoch + 1

        self.metadata["total_epochs"] += 1

        with open(self.file, "w") as f:
            json.dump(self.metadata, f, indent=2)

    def start(self):
        self.model = self.load_model()
        self.training()

    def training(self):
        self.compile_model()
        last_load_data_step = last_save_step = total_steps = self.config.trainer.start_total_steps
        self.load_play_data()

        meta_dir = 'data/model'
        meta_file = os.path.join(meta_dir, 'metadata.json')
        self.meta_writer = OptimizeWorker(meta_file)

        while True:
            self.update_learning_rate(total_steps)
            steps = self.train_epoch(self.config.trainer.epoch_to_checkpoint)
            total_steps += steps
            if last_save_step + self.config.trainer.save_model_steps < total_steps:
                self.save_current_model()
                last_save_step = total_steps

            if last_load_data_step + self.config.trainer.load_data_steps < total_steps:
                self.load_play_data()
                last_load_data_step = total_steps

            #k.clear_session()

    def train_epoch(self, epochs):
        tc = self.config.trainer
        state_ary, policy_ary, z_ary = self.dataset
        #seed = 7
        #np.random.seed(seed)
        #x = len(state_ary)
        #kf = KFold(x, n_folds=2, shuffle=True, random_state=seed)
        #for train, test in kf:
        self.model.model.fit(state_ary, [policy_ary, z_ary],
                             batch_size=tc.batch_size,
                             epochs=epochs,
                             shuffle=True,
                             callbacks=[self.meta_writer],
                             validation_split=0.05)
        steps = (state_ary.shape[0] // tc.batch_size) * epochs
        return steps

    def compile_model(self):
        self.optimizer = SGD(momentum=0.9)
        losses = [log_loss, 'mean_squared_error']
        self.model.model.compile(optimizer=self.optimizer, loss=losses)

    def update_learning_rate(self, total_steps):
        # The deepmind paper says
        # ~400k: 1e-2
        # 400k~600k: 1e-3
        # 600k~: 1e-4

        if total_steps < 40000:
            lr = 1e-2
        elif total_steps < 60000:
            lr = 1e-3
        #elif total_steps < 90000:
        #    lr = 1e-4
        else:
            lr = 1e-4  # means (1e-4 / 4): the paper batch size=2048, ours is 512.
        k.set_value(self.optimizer.lr, lr)
        logger.debug(f"total step={total_steps}, set learning rate to {lr}")

    def save_current_model(self):
        rc = self.config.resource
        model_id = datetime.now().strftime("%Y%m%d-%H%M%S.%f")
        model_dir = os.path.join(rc.next_generation_model_dir, rc.next_generation_model_dirname_tmpl % model_id)
        os.makedirs(model_dir, exist_ok=True)
        config_path = os.path.join(model_dir, rc.next_generation_model_config_filename)
        weight_path = os.path.join(model_dir, rc.next_generation_model_weight_filename)
        self.model.save(config_path, weight_path)

    def collect_all_loaded_data(self):
        state_ary_list, policy_ary_list, z_ary_list = [], [], []
        for s_ary, p_ary, z_ary_ in self.loaded_data.values():
            state_ary_list.append(s_ary)
            policy_ary_list.append(p_ary)
            z_ary_list.append(z_ary_)

        state_ary = np.concatenate(state_ary_list)
        policy_ary = np.concatenate(policy_ary_list)
        z_ary = np.concatenate(z_ary_list)
        return state_ary, policy_ary, z_ary

    @property
    def dataset_size(self):
        if self.dataset is None:
            return 0
        return len(self.dataset[0])

    def load_model(self):
        from chess_zero.agent.model_chess import ChessModel
        model = ChessModel(self.config)
        rc = self.config.resource

        dirs = get_next_generation_model_dirs(rc)
        if not dirs:
            logger.debug(f"loading best model")
            if not load_best_model_weight(model):
                raise RuntimeError(f"Best model can not loaded!")
        else:
            latest_dir = dirs[-1]
            logger.debug(f"loading latest model")
            config_path = os.path.join(latest_dir, rc.next_generation_model_config_filename)
            weight_path = os.path.join(latest_dir, rc.next_generation_model_weight_filename)
            model.load(config_path, weight_path)
        return model

    def load_play_data(self):
        filenames = get_game_data_filenames(self.config.resource)
        updated = False
        for filename in filenames:
            if filename in self.loaded_filenames:
                continue
            self.load_data_from_file(filename)
            updated = True

        for filename in (self.loaded_filenames - set(filenames)):
            self.unload_data_of_file(filename)
            updated = True

        if updated:
            logger.debug("updating training dataset")
            self.dataset = self.collect_all_loaded_data()

    def load_data_from_file(self, filename):
        try:
            logger.debug(f"loading data from {filename}")
            data = read_game_data_from_file(filename)
            self.loaded_data[filename] = self.convert_to_training_data(data)
            self.loaded_filenames.add(filename)
        except Exception as e:
            logger.warning(str(e))

    def unload_data_of_file(self, filename):
        logger.debug(f"removing data about {filename} from training set")
        self.loaded_filenames.remove(filename)
        if filename in self.loaded_data:
            del self.loaded_data[filename]

    @staticmethod
    def convert_to_training_data(data):
        """

        :param data: format is SelfPlayWorker.buffer
        :return:
        """
        state_list = []
        policy_list = []
        z_list = []
        for state, policy, z in data:
            env = ChessEnv().update(state)

            black_ary, white_ary = env.black_and_white_plane()
            state = [black_ary, white_ary] if env.board.turn == chess.BLACK else [white_ary, black_ary]

            state_list.append(state)
            policy_list.append(policy)
            z_list.append(z)

        return np.array(state_list), np.array(policy_list), np.array(z_list)
