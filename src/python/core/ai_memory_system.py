"""
AI Memory and Tracking System

This module implements a comprehensive memory system for AI agents to prevent
hallucinations by:
- Tracking all facts and decisions
- Maintaining persistent memory across sessions
- Validating information against ground truth
- Detecting and preventing hallucinations
- Providing audit trails
"""

import json
import hashlib
import time
from typing import Any, Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, asdict, field
from pathlib import Path
from collections import defaultdict, deque
import numpy as np


@dataclass
class MemoryEntry:
    """Single memory entry with validation."""
    timestamp: float
    category: str  # 'fact', 'decision', 'outcome', 'observation'
    content: Any
    confidence: float  # 0.0 to 1.0
    source: str  # Where this came from
    validated: bool = False
    validation_count: int = 0
    hash: str = field(default='')
    
    def __post_init__(self):
        """Generate hash for content integrity."""
        if not self.hash:
            content_str = json.dumps(self.content, sort_keys=True)
            self.hash = hashlib.sha256(content_str.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ValidationResult:
    """Result of validating information."""
    is_valid: bool
    confidence: float
    reason: str
    contradictions: List[str] = field(default_factory=list)
    supporting_evidence: List[str] = field(default_factory=list)


class MemoryBank:
    """
    Persistent memory bank for AI agents.
    
    Prevents hallucinations by:
    - Storing verified facts
    - Cross-referencing information
    - Detecting contradictions
    - Tracking confidence levels
    """
    
    def __init__(self, memory_file: Optional[str] = None):
        """
        Initialize memory bank.
        
        Args:
            memory_file: Path to persistent memory file
        """
        self.memory_file = memory_file
        
        # Memory storage
        self.memories: List[MemoryEntry] = []
        self.memory_index: Dict[str, List[int]] = defaultdict(list)  # category -> indices
        self.fact_database: Dict[str, MemoryEntry] = {}  # hash -> memory
        
        # Statistics
        self.total_stored = 0
        self.validated_count = 0
        self.contradictions_found = 0
        self.hallucinations_prevented = 0
        
        # Load existing memories
        if memory_file and Path(memory_file).exists():
            self.load(memory_file)
    
    def store(
        self, 
        category: str, 
        content: Any, 
        confidence: float = 1.0,
        source: str = "unknown"
    ) -> MemoryEntry:
        """
        Store new information in memory.
        
        Args:
            category: Type of information
            content: The actual information
            confidence: Confidence level (0.0 to 1.0)
            source: Source of information
            
        Returns:
            Created memory entry
        """
        entry = MemoryEntry(
            timestamp=time.time(),
            category=category,
            content=content,
            confidence=confidence,
            source=source
        )
        
        # Check for duplicates
        if entry.hash in self.fact_database:
            existing = self.fact_database[entry.hash]
            existing.validation_count += 1
            existing.confidence = min(1.0, existing.confidence + 0.1)
            return existing
        
        # Store new memory
        idx = len(self.memories)
        self.memories.append(entry)
        self.memory_index[category].append(idx)
        self.fact_database[entry.hash] = entry
        self.total_stored += 1
        
        return entry
    
    def recall(
        self, 
        category: Optional[str] = None,
        min_confidence: float = 0.0,
        limit: Optional[int] = None
    ) -> List[MemoryEntry]:
        """
        Recall memories by category and confidence.
        
        Args:
            category: Filter by category (None = all)
            min_confidence: Minimum confidence threshold
            limit: Maximum number to return
            
        Returns:
            List of matching memories
        """
        if category:
            indices = self.memory_index.get(category, [])
            candidates = [self.memories[i] for i in indices]
        else:
            candidates = self.memories
        
        # Filter by confidence
        results = [m for m in candidates if m.confidence >= min_confidence]
        
        # Sort by timestamp (most recent first)
        results.sort(key=lambda m: m.timestamp, reverse=True)
        
        if limit:
            results = results[:limit]
        
        return results
    
    def validate(self, content: Any, category: str) -> ValidationResult:
        """
        Validate information against memory.
        
        Args:
            content: Information to validate
            category: Category of information
            
        Returns:
            Validation result
        """
        # Create temporary entry for comparison
        temp_entry = MemoryEntry(
            timestamp=time.time(),
            category=category,
            content=content,
            confidence=0.0,
            source="validation"
        )
        
        # Check if exact match exists
        if temp_entry.hash in self.fact_database:
            existing = self.fact_database[temp_entry.hash]
            return ValidationResult(
                is_valid=True,
                confidence=existing.confidence,
                reason="Exact match in memory",
                supporting_evidence=[f"Validated {existing.validation_count} times"]
            )
        
        # Look for similar content in same category
        similar_memories = self.recall(category=category, min_confidence=0.5)
        
        if not similar_memories:
            return ValidationResult(
                is_valid=False,
                confidence=0.0,
                reason="No similar information in memory"
            )
        
        # Check for contradictions
        contradictions = []
        supporting = []
        
        for memory in similar_memories:
            similarity = self._calculate_similarity(content, memory.content)
            
            if similarity > 0.8:
                supporting.append(f"Similar to memory from {memory.source}")
            elif similarity < 0.3:
                contradictions.append(
                    f"Contradicts memory: {str(memory.content)[:50]}..."
                )
        
        if contradictions:
            self.contradictions_found += 1
            return ValidationResult(
                is_valid=False,
                confidence=0.0,
                reason="Contradicts existing memories",
                contradictions=contradictions
            )
        
        if supporting:
            return ValidationResult(
                is_valid=True,
                confidence=0.7,
                reason="Supported by similar memories",
                supporting_evidence=supporting
            )
        
        return ValidationResult(
            is_valid=False,
            confidence=0.3,
            reason="Uncertain - no strong evidence either way"
        )
    
    def _calculate_similarity(self, content1: Any, content2: Any) -> float:
        """Calculate similarity between two pieces of content."""
        # Convert to strings for comparison
        str1 = json.dumps(content1, sort_keys=True) if not isinstance(content1, str) else content1
        str2 = json.dumps(content2, sort_keys=True) if not isinstance(content2, str) else content2
        
        # Simple string similarity (can be improved with NLP)
        if str1 == str2:
            return 1.0
        
        # Jaccard similarity on words
        words1 = set(str1.lower().split())
        words2 = set(str2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def detect_hallucination(self, claimed_fact: Dict) -> Tuple[bool, str]:
        """
        Detect if a claimed fact is a hallucination.
        
        Args:
            claimed_fact: Fact being claimed
            
        Returns:
            (is_hallucination, reason)
        """
        validation = self.validate(claimed_fact, category="fact")
        
        if not validation.is_valid and validation.confidence < 0.3:
            self.hallucinations_prevented += 1
            return True, validation.reason
        
        if validation.contradictions:
            self.hallucinations_prevented += 1
            return True, f"Contradicts: {', '.join(validation.contradictions[:2])}"
        
        return False, "Fact validated"
    
    def get_ground_truth(self, query: str, category: str) -> Optional[MemoryEntry]:
        """
        Get the most confident ground truth for a query.
        
        Args:
            query: Query string
            category: Category to search
            
        Returns:
            Most confident memory or None
        """
        memories = self.recall(category=category, min_confidence=0.7)
        
        if not memories:
            return None
        
        # Return most validated memory
        memories.sort(key=lambda m: (m.validation_count, m.confidence), reverse=True)
        return memories[0]
    
    def save(self, filepath: Optional[str] = None) -> None:
        """Save memory bank to disk."""
        save_path = filepath or self.memory_file
        
        if not save_path:
            raise ValueError("No filepath provided")
        
        data = {
            'memories': [m.to_dict() for m in self.memories],
            'statistics': {
                'total_stored': self.total_stored,
                'validated_count': self.validated_count,
                'contradictions_found': self.contradictions_found,
                'hallucinations_prevented': self.hallucinations_prevented,
            }
        }
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, filepath: str) -> None:
        """Load memory bank from disk."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Reconstruct memories
        self.memories = []
        for mem_dict in data['memories']:
            entry = MemoryEntry(**mem_dict)
            self.memories.append(entry)
            
            # Rebuild indices
            idx = len(self.memories) - 1
            self.memory_index[entry.category].append(idx)
            self.fact_database[entry.hash] = entry
        
        # Load statistics
        stats = data.get('statistics', {})
        self.total_stored = stats.get('total_stored', 0)
        self.validated_count = stats.get('validated_count', 0)
        self.contradictions_found = stats.get('contradictions_found', 0)
        self.hallucinations_prevented = stats.get('hallucinations_prevented', 0)
    
    def get_statistics(self) -> Dict:
        """Get memory statistics."""
        return {
            'total_memories': len(self.memories),
            'by_category': {cat: len(indices) for cat, indices in self.memory_index.items()},
            'validated_count': self.validated_count,
            'contradictions_found': self.contradictions_found,
            'hallucinations_prevented': self.hallucinations_prevented,
            'average_confidence': np.mean([m.confidence for m in self.memories]) if self.memories else 0.0,
        }


class StateTracker:
    """
    Track AI state across time to detect inconsistencies.
    
    Maintains:
    - Game state history
    - Decision history
    - Performance metrics
    - Consistency checks
    """
    
    def __init__(self, max_history: int = 1000):
        """
        Initialize state tracker.
        
        Args:
            max_history: Maximum states to keep in memory
        """
        self.max_history = max_history
        
        # State tracking
        self.state_history: deque = deque(maxlen=max_history)
        self.decision_history: deque = deque(maxlen=max_history)
        self.outcome_history: deque = deque(maxlen=max_history)
        
        # Consistency tracking
        self.consistency_score = 1.0
        self.inconsistencies_detected = 0
        
        # Performance tracking
        self.wins = 0
        self.losses = 0
        self.total_reward = 0.0
    
    def record_state(self, state: Dict, timestamp: Optional[float] = None) -> None:
        """Record current state."""
        self.state_history.append({
            'timestamp': timestamp or time.time(),
            'state': state,
        })
    
    def record_decision(
        self, 
        state: Dict, 
        action: Any, 
        reasoning: str,
        timestamp: Optional[float] = None
    ) -> None:
        """Record decision made."""
        self.decision_history.append({
            'timestamp': timestamp or time.time(),
            'state': state,
            'action': action,
            'reasoning': reasoning,
        })
    
    def record_outcome(
        self, 
        reward: float, 
        success: bool,
        timestamp: Optional[float] = None
    ) -> None:
        """Record outcome of decision."""
        self.outcome_history.append({
            'timestamp': timestamp or time.time(),
            'reward': reward,
            'success': success,
        })
        
        # Update performance
        if success:
            self.wins += 1
        else:
            self.losses += 1
        
        self.total_reward += reward
    
    def check_consistency(self, current_state: Dict, claimed_fact: Dict) -> bool:
        """
        Check if claimed fact is consistent with history.
        
        Args:
            current_state: Current game state
            claimed_fact: Fact being claimed
            
        Returns:
            True if consistent
        """
        if not self.state_history:
            return True  # No history to compare
        
        # Check against recent states
        recent_states = list(self.state_history)[-10:]
        
        for hist in recent_states:
            # Check for contradictions
            if self._contradicts(hist['state'], claimed_fact):
                self.inconsistencies_detected += 1
                self.consistency_score *= 0.95
                return False
        
        return True
    
    def _contradicts(self, historical_state: Dict, claimed_fact: Dict) -> bool:
        """Check if claimed fact contradicts historical state."""
        # Simple contradiction detection
        for key, value in claimed_fact.items():
            if key in historical_state:
                if historical_state[key] != value:
                    # Values differ - potential contradiction
                    return True
        
        return False
    
    def get_confidence(self) -> float:
        """Get current confidence in AI's state tracking."""
        return self.consistency_score
    
    def get_statistics(self) -> Dict:
        """Get tracking statistics."""
        return {
            'states_tracked': len(self.state_history),
            'decisions_tracked': len(self.decision_history),
            'outcomes_tracked': len(self.outcome_history),
            'win_rate': self.wins / max(self.wins + self.losses, 1),
            'total_reward': self.total_reward,
            'consistency_score': self.consistency_score,
            'inconsistencies_detected': self.inconsistencies_detected,
        }


class HallucinationDetector:
    """
    Detect and prevent AI hallucinations.
    
    Methods:
    - Fact checking against memory
    - Consistency validation
    - Confidence scoring
    - Uncertainty quantification
    """
    
    def __init__(
        self, 
        memory_bank: MemoryBank,
        state_tracker: StateTracker,
        confidence_threshold: float = 0.7
    ):
        """
        Initialize hallucination detector.
        
        Args:
            memory_bank: Memory bank for fact checking
            state_tracker: State tracker for consistency
            confidence_threshold: Minimum confidence to accept facts
        """
        self.memory_bank = memory_bank
        self.state_tracker = state_tracker
        self.confidence_threshold = confidence_threshold
        
        # Detection statistics
        self.checks_performed = 0
        self.hallucinations_detected = 0
        self.uncertain_claims = 0
    
    def check_claim(
        self, 
        claim: Dict, 
        category: str = "fact"
    ) -> Tuple[bool, float, str]:
        """
        Check if a claim is valid or a hallucination.
        
        Args:
            claim: Claim to check
            category: Category of claim
            
        Returns:
            (is_valid, confidence, reason)
        """
        self.checks_performed += 1
        
        # Validate against memory
        validation = self.memory_bank.validate(claim, category)
        
        # Check consistency with state
        consistent = self.state_tracker.check_consistency({}, claim)
        
        # Combined score
        memory_score = validation.confidence
        consistency_score = 1.0 if consistent else 0.0
        combined_confidence = (memory_score * 0.7 + consistency_score * 0.3)
        
        # Determine if valid
        if combined_confidence < self.confidence_threshold:
            if validation.contradictions:
                self.hallucinations_detected += 1
                return False, combined_confidence, "Contradicts known facts"
            else:
                self.uncertain_claims += 1
                return False, combined_confidence, "Insufficient confidence"
        
        return True, combined_confidence, "Validated"
    
    def get_statistics(self) -> Dict:
        """Get detection statistics."""
        return {
            'checks_performed': self.checks_performed,
            'hallucinations_detected': self.hallucinations_detected,
            'uncertain_claims': self.uncertain_claims,
            'detection_rate': self.hallucinations_detected / max(self.checks_performed, 1),
        }


def create_memory_system(memory_file: str = "data/ai_memory.json") -> Dict:
    """
    Create complete memory system for AI.
    
    Args:
        memory_file: Path to persistent memory file
        
    Returns:
        Dictionary with all components
    """
    memory_bank = MemoryBank(memory_file)
    state_tracker = StateTracker()
    hallucination_detector = HallucinationDetector(memory_bank, state_tracker)
    
    return {
        'memory_bank': memory_bank,
        'state_tracker': state_tracker,
        'hallucination_detector': hallucination_detector,
    }
