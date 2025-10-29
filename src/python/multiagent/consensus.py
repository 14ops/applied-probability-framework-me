from typing import List, Dict, Any

def majority_vote(proposals: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not proposals: return {'action': None, 'confidence': 0.0, 'reason': 'no proposals'}
    # weight by confidence; pick highest total weight per action
    tally = {}
    for p in proposals:
        a = p.get('action'); w = float(p.get('confidence', 0.5))
        tally[a] = tally.get(a, 0.0) + w
    action = max(tally.items(), key=lambda kv: kv[1])[0]
    winners = [p for p in proposals if p.get('action') == action]
    reason = " | ".join(p.get('reason','') for p in winners[:3])
    conf = sum(p.get('confidence',0.5) for p in winners)/max(1,len(winners))
    return {'action': action, 'confidence': conf, 'reason': reason}
