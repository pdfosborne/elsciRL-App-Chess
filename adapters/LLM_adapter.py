from typing import List
import torch
from torch import Tensor

import numpy as np
from gymnasium.spaces import Box

import chess
from chess import Board, SQUARES_180

# Link to relevant ENCODER
from elsciRL.adapters.LLM_state_generators.text_ollama import OllamaAdapter

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
        
        # Define observation space
        self.observation_space = Box(low=-1, high=1, shape=(1,384), dtype=np.float32)

        self.LLM_adapter = OllamaAdapter(
            model_name=setup_info.get('model_name', 'llama3.2'),
            base_prompt=setup_info.get('system_prompt', 'You are playing a game of Chess.'),
            action_history_length=setup_info.get('action_history_length', 5),
            encoder=setup_info.get('encoder', 'MiniLM_L6v2')
        )

    def adapter(self, state: str, legal_moves:list = [], episode_action_history:list = [], encode:bool=True, indexed: bool = False) -> Tensor:     
        """ Pieces on board are counted to define state.
        12 piece types define the observation space."""

        # Transform state
        board = chess.Board(state)
        board_flip = board.copy(stack=False)
        board_flip.apply_transform(chess.flip_vertical)
        state = self.compact_lst(board_flip) # Returns board as list of strings for each board position -> len=64
        
        # Use the elsciRL LLM adapter to transform and encode
        state_encoded = self.LLM_adapter.adapter(
            state=state, 
            legal_moves=legal_moves, 
            episode_action_history=episode_action_history, 
            encode=encode, 
            indexed=indexed
        )

        return state_encoded
    
    @staticmethod
    def sample():
        """Sample a random chess board state and return the encoded state."""
        import random
        adapter = Adapter()
        board_fen = chess.Board().fen()
        episode_action_history = []
        for i in range(10):
            print("\n ++++++++++++++++++++++++++++++++")
            print("Action Number:", i)
            legal_moves = str(list(chess.Board().legal_moves)).replace(" Move.from_uci('","").replace("[Move.from_uci('","").replace("')","").replace("]","").split(",")
            legal_moves = legal_moves if (legal_moves != "[]") else [""]
            print("\n ----------------------")            
            state = adapter.adapter(board_fen, legal_moves, episode_action_history, encode=False)
            print(f"\n State: {state}")
            state_encoded = adapter.adapter(board_fen, legal_moves, episode_action_history, encode=True)
            print(f"\n Encoded State: {state_encoded}")
            print("\n ----------------------")
            action = random.choice(legal_moves)
            chess.Board().push_san(chess.Board().san(chess.Move.from_uci(action)))
            episode_action_history.append(action)
            
            