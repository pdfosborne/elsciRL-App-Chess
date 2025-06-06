import chess
from chess import Board
from chess.engine import Limit, SimpleEngine
from elsciRL.agents.agent_abstract import Agent
import random

# Data importing and processing
import json
from tqdm import tqdm
import urllib

class RandomAgentFiltered(Agent):
    """ This is simply a random decision maker, does not learn. 
        Possible actions filtered on human player data to minimise randomness."""
    def __init__(self):
        # --- Data Import -------------------------------------------------------------------------------------------------
        # dict[req_board][move_uci] = {"prev_moves_uci": list,
        #                              "whiteWon": int, "blackWon": int, "draw": int,
        #                              "totalGames": int } 

        # Get player move stats data from public source
        data_source = 'https://raw.githubusercontent.com/pdfosborne/elsciRL-App-Chess/main/environment/opponent_agents/data/stats_map.json'
        self.player_data_dict = json.loads(urllib.request.urlopen(data_source).read())

        # with open('./environment/opponent_agents/data/stats_map.json', 'r') as json_file:
        #     self.player_data_dict = json.load(json_file)

    def policy(self, board_p: Board):
        # Go through each known move from player data and extract move_uci of known moves
        filtered_legal_moves = []
        for move in self.player_data_dict[str(board_p.fen())]:
            if self.player_data_dict[str(board_p.fen())][move]["totalGames"] > 10:
                filtered_legal_moves.append(move)
        
        # In cases where a known list of moves doesn't exist, pick from full list of legal moves
        if len(filtered_legal_moves) != 0:
            move_uci = str(random.choice(filtered_legal_moves))
        else:
            move_uci = str(random.choice(list(board_p.legal_moves))) 
            
        return move_uci