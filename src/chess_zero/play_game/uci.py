import sys
from logging import getLogger
import chess
from chess_zero.config import Config, PlayWithHumanConfig
from chess_zero.play_game.game_model import PlayWithHuman
from chess_zero.env.chess_env import ChessEnv

logger = getLogger(__name__)

def start(config: Config):

    PlayWithHumanConfig().update_play_config(config.play)
    config.play.thinking_loop = 1

    chess_model = None
    env = ChessEnv().reset()

    while True:
        line=input()
        words=line.rstrip().split(" ",1)
        if words[0] == "uci":
            print("id name ChessZero")
            print("id author ChessZero")
            print("uciok")
        elif words[0]=="isready":
            if(chess_model == None):
                chess_model = PlayWithHuman(config)
            print("readyok")
        elif words[0]=="ucinewgame":
            env.reset()
        elif words[0]=="position":
            words=words[1].split(" ",1)
            if words[0]=="startpos":
                env.reset()
            else:
                fen = words[0]
                for _ in range(5):
                    words=words[1].split(' ',1)
                    fen += " "+words[0]
                env.update(fen)
            if(len(words)>1):
                words=words[1].split(" ",1)
                if words[0]=="moves":
                    for w in words[1].split(" "):
                        env.step(w,False)
        elif words[0]=="go":
            action = chess_model.move_by_ai(env)
            print(f"bestmove {action}")
        elif words[0]=="stop":
            pass #lol
        elif words[0]=="quit":
            break

def info(depth,move, score):
    print(f"info score cp {int(score*100)} depth {depth} pv {move}")
    sys.stdout.flush()
