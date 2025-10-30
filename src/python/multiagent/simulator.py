from typing import List
from .blackboard import Blackboard
from .consensus import majority_vote

class TeamSimulator:
    def __init__(self, env, agents: List):
        self.env = env
        self.agents = agents
        self.board = Blackboard()

    def play_round(self):
        s = self.env.reset()
        done = False
        while not done:
            self.board.clear()
            for ag in self.agents:
                self.board.post_proposal(ag.propose(s))
            decision = majority_vote(self.board.proposals)
            a = decision['action']
            res = self.env.step(a)
            s = res.next_state
            done = res.done
        return True
