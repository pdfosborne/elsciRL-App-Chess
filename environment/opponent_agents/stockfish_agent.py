import chess
from chess import Board
from chess.engine import Limit, SimpleEngine
from elsciRL.agents.agent_abstract import Agent


class StockfishAgent(Agent):
    def __init__(self, engine_path: str, skill_level: int = 10):
        self.engine: SimpleEngine = chess.engine.SimpleEngine.popen_uci(engine_path)
        self.engine.configure({"Skill Level": skill_level})

    def policy(self, board: Board, limit: Limit) -> str:
        return self.engine.play(board, limit).move

    #def policy_play(self, board: Board, limit: Limit) -> str:
    #    return  self.play(board, limit)
