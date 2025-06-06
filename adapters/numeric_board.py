from typing import Dict, List
import pandas as pd
import torch
from torch import Tensor
# StateAdapter includes static methods for adapters
from elsciRL.encoders.poss_state_encoded import StateEncoder

import chess
from chess import Board, SQUARES_180

class Adapter:
    _cached_state_idx: Dict[str, int] = dict()
    @staticmethod
    def compact_lst(board: Board) -> List[str]:
        builder = ["."] * len(SQUARES_180)
        for i, square in enumerate(SQUARES_180):
            piece = board.piece_at(square)
            if piece:
                builder[i] = piece.symbol()
        return builder
   
    def __init__(self,setup_info:dict={}) -> None:
        # TODO: Update this based on the current problem, each requires preset knowledge of all possible states/actions/objects
        # - Possible States
        # - Possible Actions
        # - Prior Actions
        # - Possible Objects
    
        # Initialise encoder based on all possible env states
        self.observation_space = 12

        self.chess_piece_number = {
            '.':0,
            'K':1,'Q':2,'R':3,'B':4,'N':5,'P':6, 
            'k':7,'q':8,'r':9,'b':10,'n':11,'p':12
            }
        
    def adapter(self, state:any, legal_moves:list = None, episode_action_history:list = None, encode:bool = True, indexed: bool = False) -> Tensor:
        """  """
        board = chess.Board(state)
        board_flip = board.copy(stack=False)
        board_flip.apply_transform(chess.flip_vertical)
        state = self.compact_lst(board_flip)

        if encode:
            state_encoded = []
            for piece in state:
                state_encoded.append(self.chess_piece_number[piece])

            state_encoded = torch.tensor(state_encoded)

        else:
            state_encoded = state
        
        return state_encoded