from typing import List, Dict, Any

class Blackboard:
    def __init__(self):
        self.observations: List[Dict[str, Any]] = []
        self.proposals: List[Dict[str, Any]] = []

    def post_obs(self, obs: Dict[str, Any]): self.observations.append(obs)
    def post_proposal(self, prop: Dict[str, Any]): self.proposals.append(prop)
    def clear(self): self.observations.clear(); self.proposals.clear()
