# Engine used to obtain move scores
import chess.engine
from chess import Board
import numpy as np

# Opponent agent imports
from elsciRL.agents.random_agent import RandomAgent

# Imports for rendering
import chess.svg
from io import BytesIO
from PIL import Image
import cairosvg
import matplotlib.pyplot as plt


class Engine:
    """Defines the environment function from the generator engine.
       Expects the following:
        - reset() to reset the env a start position(s)
        - step() to make an action and update the game state
        - legal_moves_generator() to generate the list of legal moves
    """
    def __init__(self, local_setup_info:dict={}) -> None:
        """Initialize Engine"""
        if "custom_termination" in local_setup_info:
            self.custom_termination = local_setup_info["custom_termination"]
        else:
            self.custom_termination = None
        # Ledger of the environment with meta information for the problem
        ledger_required = {
            'id': 'Unique Problem ID',
            'type': 'Numeric & Language',
            'description': 'Problem Description',
            'goal': 'Goal Description'
            }
        
        ledger_optional = {
            'reward': 'Reward Description',
            'punishment': 'Punishment Description (if any)',
            'state': 'State Description',
            'constraints': 'Constraints Description',
            'action': 'Action Description',
            'author': 'Author',
            'year': 'Year',
            'render_data':{}
        }
        ledger_gym_compatibility = {
            # Limited to discrete actions for now, set to arbitrary large number if uncertain
            'action_space_size':1000, 
        }
        self.ledger = ledger_required | ledger_optional | ledger_gym_compatibility
        # --- CHESS ENGINE SETUP ---
        self.board: Board = chess.Board()
        if local_setup_info["action_cap"]:
            self.action_cap = local_setup_info["action_cap"]
        else:
            self.action_cap = None
        
        if local_setup_info["reward_signal"]:
            self.reward_signal = local_setup_info["reward_signal"]
        else:
            self.reward_signal = None

        # --- CHESS OPPONENT AGENT SETUP ---
        # Opponent agent is unique to Chess as part of the Probabilistic environment
        # But for ease we utilize the agent functions within elcsiRL for the opponent
        OPPONENT_AGENT_TYPES = {
            "Random": RandomAgent
        }
        OPPONENT_AGENT_PARAMETERS = {
            "Random":{}
        }
        opponent_agent = local_setup_info['opponent_agent']
        opponent_agent_parameters = OPPONENT_AGENT_PARAMETERS[opponent_agent]
        self.training_opponent = OPPONENT_AGENT_TYPES[opponent_agent](**opponent_agent_parameters) 
        # ---

    def reward_signal(self, obs:any):
        game_result = obs.result()
        if self.reward_signal:
            # Custom reward signal from env config
            # Win
            if game_result == "1-0":
                reward = self.reward_signal[0]
            # Loss
            elif game_result == "0-1":
                reward = self.reward_signal[0]*-1
            # Draw 
            elif game_result == "1/2-1/2":
                reward = self.reward_signal[0]*-0.1
            # Custom reward for each action taken
            else:
                reward = self.reward_signal[1]
        else:
            # Default reward signal
            # Win
            if game_result == "1-0":
                reward = 1
            # Loss
            elif game_result == "0-1":
                reward = -1
            # Draw
            elif game_result == "1/2-1/2":
                reward = -0.1
            # Reward for each action taken
            else:
                reward = 0
        return reward


    def white_move(self, action:any):
        self.board.push_san(self.board.san(chess.Move.from_uci(action)))        
        obs = self.board.fen()
        terminated = self.board.is_game_over()
        if self.custom_termination:
            if not terminated:
                # Custom termination on first capture
                if self.custom_termination == "first_capture":
                    # - Check if the number of pieces on the board is less than 74
                    if np.sum([obs.piece_type_at(sq) for sq in chess.SQUARES if obs.piece_type_at(sq) is not None])<74:
                        terminated = True
        
        return obs, terminated
    

    def black_move(self):
        # Black move
        obs = self.board.fen()
        legal_moves = self.legal_move_generator(obs)
        if len(legal_moves) > 0:
            action = self.training_opponent.policy(obs, legal_moves)
            self.board.push_san(self.board.san(chess.Move.from_uci(action)))
            terminated = self.board.is_game_over()

        return terminated

    def reset(self, start_obs:any=None):
        """Fully reset the environment."""
        self.board.reset()
        obs = self.board.fen()
        return obs

    def step(self, state:any, action:any):
        """Enact an action."""
        # Each action completes a white move then a black move
        # White move
        obs, terminated = self.white_move(action)
        # Chess engine does not provide a reward signal by itself

        # Black move
        # - If the game is not over, the black agent will make a move
        if not terminated:
            terminated = self.black_move()
        
        # - Game may end on black move so need to apply this to white's last move
        reward =  self.reward_signal(obs)
        return obs, reward, terminated, {}

    def legal_move_generator(self, obs:any=None):
        """Define legal moves at each position"""
        legal_moves = str(list(self.board.legal_moves)).replace(" Move.from_uci('","").replace("[Move.from_uci('","").replace("')","").replace("]","").split(",")
        legal_moves = legal_moves if (legal_moves != "[]") else [""]
        return legal_moves

    def render(self, state:any=None):
        """Render the current chess board using matplotlib."""
        # Generate SVG image of the board
        if state:
            svg_board = chess.svg.board(state)
        else:
            svg_board = chess.svg.board(self.board)
        # Convert SVG to PNG using PIL
        img_bytes = BytesIO()
        img_bytes.write(svg_board.encode('utf-8'))
        img_bytes.seek(0)
        # Use cairosvg to convert SVG to PNG
        try:
            png_bytes = cairosvg.svg2png(bytestring=svg_board)
            img = Image.open(BytesIO(png_bytes))
        except ImportError:
            raise ImportError("cairosvg is required for rendering the chess board. Install with 'pip install cairosvg'.")
        # Display with matplotlib
        plt.figure(figsize=(6,6))
        plt.imshow(np.asarray(img))
        plt.axis('off')

        # Return the matplotlib figure object for further use
        fig = plt.gcf()
        return fig
    
    def close(self):
        """Close the environment."""
        self.board = None
        plt.close('all')
