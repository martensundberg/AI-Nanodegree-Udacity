
from sample_players import DataPlayer
from math import sqrt
from isolation import DebugState

W = 13
H = 9

class CustomPlayer(DataPlayer):
    """ Implement your own agent to play knight's Isolation

    The get_action() method is the only required method for this project.
    You can modify the interface for get_action by adding named parameters
    with default values, but the function MUST remain compatible with the
    default interface.

    **********************************************************************
    NOTES:
    - The test cases will NOT be run on a machine with GPU access, nor be
      suitable for using any other machine learning techniques.

    - You can pass state forward to your agent on the next turn by assigning
      any pickleable object to the self.context attribute.
    **********************************************************************
    """

    def get_action(self, state):
        """ Employ an adversarial search technique to choose an action
        available in the current state calls self.queue.put(ACTION) at least

        This method must call self.queue.put(ACTION) at least once, and may
        call it as many times as you want; the caller will be responsible
        for cutting off the function after the search time limit has expired.

        See RandomPlayer and GreedyPlayer in sample_players for more examples.

        **********************************************************************
        NOTE: 
        - The caller is responsible for cutting off search, so calling
          get_action() from your own code will create an infinite loop!
          Refer to (and use!) the Isolation.play() function to run games.
        **********************************************************************
        """
        # TODO: Replace the example implementation below with your own search
        #       method by combining techniques from lecture
        #
        # EXAMPLE: choose a random move without any search--this function MUST
        #          call self.queue.put(ACTION) at least once before time expires
        #          (the timer is automatically managed for you)
        import random
        import time
        from datetime import datetime

        self.nr_nodes = 0
        self.use_alpha_beta = True
        self.use_iterative_deepening = True
        
        if state.ply_count < 2:
            # print('New match @ {}'.format(datetime.fromtimestamp(time.time())))
            self.queue.put(random.choice(state.actions()))
        else:
            if self.use_iterative_deepening:
                max_depth = 10
                for depth in range(1,max_depth+1):
                    self.queue.put(self.apply_search(state, depth))                    
            else:
                self.queue.put(self.apply_search(state, depth=3))

    def apply_search(self, state, depth):
        alpha = float("-inf")
        beta = float("inf")
        best_score = float("-inf")
        best_move = None

        def min_value(state, depth, alpha, beta):
            if state.terminal_test(): 
              return state.utility(self.player_id)
            if depth <= 0: 
              return self.score(state)
            value = float("inf")
            for action in state.actions():
                self.nr_nodes += 1
                value = min(value, max_value(state.result(action), depth - 1, alpha, beta))

                if self.use_alpha_beta:
                    if value <= alpha:
                        return value
                    beta = min(beta,value)

            return value

        def max_value(state, depth, alpha, beta):
            if state.terminal_test():
              return state.utility(self.player_id)
            if depth <= 0: 
              return self.score(state)
            value = float("-inf")
            for action in state.actions():
                self.nr_nodes += 1
                value = max(value, min_value(state.result(action), depth - 1, alpha, beta))

                if self.use_alpha_beta:
                    if value > beta:
                        return value
                    alpha = max(alpha,value)
            return value

        for action in state.actions():
            own_x, own_y = self.get_position(state.locs[self.player_id])
            opp_x, opp_y = self.get_position(state.locs[1 - self.player_id])
            value = min_value(state.result(action), depth - 1, alpha, beta)
            if self.use_alpha_beta:
                alpha = max(alpha,value)
            self.nr_nodes += 1
            if value > best_score:
                best_score = value
                best_move = action

        return best_move

    def score(self, state):      
        own_loc = state.locs[self.player_id]
        opp_loc = state.locs[1 - self.player_id]

        own_board_score = self.get_board_score(own_loc)
        opp_board_score = self.get_board_score(opp_loc)

        own_liberties = state.liberties(own_loc)
        opp_liberties = state.liberties(opp_loc)
        
        own_score = [self.get_board_score(lib) for lib in own_liberties]
        opp_score = [self.get_board_score(lib) for lib in opp_liberties]

        return sum(own_score) - sum(opp_score)

    def get_position(self,position):
        return (position % W, position // W)

    def get_board_score(self,position):
        # Get the positional score in the board relative to the center position.
        
        gain = 1.4

        x, y = (position % W, position // W)

        # Distance from center
        dfc_x = abs(W/2-(x+0.5))
        dfc_y = abs(H/2-(y+0.5))

        if dfc_x == W//2:
          score = 1
        elif dfc_y == H//2:
          score = 1
        elif dfc_x == W//2 - 1:
          score = 1
        elif dfc_y == H//2 - 1:
          score = gain
        elif dfc_x == W//2 - 2:
          score = gain
        elif dfc_y == H//2 - 2:
          score = gain**2
        elif dfc_x == W//2 - 3:
          score = gain**2        
        elif dfc_x == 0 and dfc_y == 0:
          score = gain**3
        elif dfc_y <= H//2 - 3 and dfc_x <= W//2 - 4:
          score = gain**3
        else:
          score = 1

        return score




