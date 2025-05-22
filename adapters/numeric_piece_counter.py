from typing import List
import torch
from torch import Tensor

from gymnasium.spaces import Discrete

import chess
from chess import Board, SQUARES_180

# Link to relevant ENCODER
from elsciRL.encoders.observable_objects_encoded import ObjectEncoder

class Adapter: 
    @staticmethod
    def chess_object_lst() -> List[str]:
        chess_pieces = ['K','Q','R','B','N','P', 
                        'k','q','r','b','n','p']
        return chess_pieces
    
    @staticmethod
    def compact_lst(board: Board) -> List[str]:
        builder = ["."] * len(SQUARES_180)
        for i, square in enumerate(SQUARES_180):
            piece = board.piece_at(square)

            if piece:
                builder[i] = piece.symbol()

        return builder

    def __init__(self, setup_info:dict={}) -> None:
        # Initialise general encoder with local game objects
        self.local_objects = {obj: i for i, obj in enumerate(self.chess_object_lst())}
        self.encoder = ObjectEncoder(list(self.local_objects.keys()) + ["."])
        
        # Define observation space
        self.observation_space = Discrete(12)

    def adapter(self, state: str, legal_moves:list = None, episode_action_history:list = None, encode:bool=True, indexed: bool = False) -> Tensor:     
        """ Pieces on board are counted to define state.
        12 piece types define the observation space."""

        # Transform state
        board = chess.Board(state.fen())
        board_flip = board.copy(stack=False)
        board_flip.apply_transform(chess.flip_vertical)
        state = self.compact_lst(board_flip) # Returns board as list of strings for each board position -> len=64
        if encode:
            state_encoded = self.encoder.encode(state=state)
        else:
            state_encoded = state

        if (indexed):
            state_encoded = torch.tensor([self.local_objects.get(obj, len(self.local_objects)) for obj in state])
        
        return state_encoded
    
    def sample():
        board = chess.Board(fen='rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2')
        legal_moves = ['g1h3', 'g1f3', 'g1e2', 'f1a6', 'f1b5', 'f1c4', 'f1d3', 
                       'f1e2', 'e1e2', 'd1h5', 'd1g4', 'd1f3', 'd1e2', 'b1c3', 
                       'b1a3', 'e4e5', 'h2h3', 'g2g3', 'f2f3', 'd2d3', 'c2c3', 
                       'b2b3', 'a2a3', 'h2h4', 'g2g4', 'f2f4', 'd2d4', 'c2c4', 'b2b4', 'a2a4']
        episode_action_history = ['e2e4', 'c7c5']
        adapter = Adapter()
        state = adapter.adapter(board, legal_moves, episode_action_history, encode=False)
        state_encoded = adapter.adapter(board, legal_moves, episode_action_history, encode=True)
        return state, state_encoded