"""
Ollama Integration with Memory System

Integrates Ollama AI with the memory system to prevent hallucinations.
"""

import json
import requests
from typing import Dict, List, Optional, Any
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.ai_memory_system import (
    MemoryBank, 
    StateTracker, 
    HallucinationDetector,
    create_memory_system
)


class OllamaWithMemory:
    """
    Ollama AI with integrated memory system to prevent hallucinations.
    
    Features:
    - Persistent memory across sessions
    - Fact validation before responding
    - Hallucination detection
    - State tracking
    - Confidence scoring
    """
    
    def __init__(
        self,
        model: str = "llama2",
        ollama_url: str = "http://localhost:11434",
        memory_file: str = "data/ollama_memory.json"
    ):
        """
        Initialize Ollama with memory.
        
        Args:
            model: Ollama model to use
            ollama_url: URL of Ollama server
            memory_file: Path to persistent memory file
        """
        self.model = model
        self.ollama_url = ollama_url
        
        # Create memory system
        memory_system = create_memory_system(memory_file)
        self.memory_bank = memory_system['memory_bank']
        self.state_tracker = memory_system['state_tracker']
        self.hallucination_detector = memory_system['hallucination_detector']
        
        # Conversation history
        self.conversation_history: List[Dict] = []
        
        # Statistics
        self.total_queries = 0
        self.hallucinations_prevented = 0
        self.facts_validated = 0
    
    def query(
        self, 
        prompt: str, 
        validate_response: bool = True,
        store_facts: bool = True
    ) -> Dict[str, Any]:
        """
        Query Ollama with memory validation.
        
        Args:
            prompt: Query prompt
            validate_response: Validate response against memory
            store_facts: Store new facts from response
            
        Returns:
            Response with validation info
        """
        self.total_queries += 1
        
        # Enrich prompt with relevant memories
        enriched_prompt = self._enrich_prompt_with_memory(prompt)
        
        # Query Ollama
        try:
            response = self._call_ollama(enriched_prompt)
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'response': None
            }
        
        # Extract and validate facts from response
        if validate_response:
            validation_result = self._validate_response(response)
            
            if not validation_result['is_valid']:
                self.hallucinations_prevented += 1
                return {
                    'success': False,
                    'response': response,
                    'validation': validation_result,
                    'warning': 'Response contained unvalidated information',
                }
        
        # Store new facts
        if store_facts:
            self._store_facts_from_response(prompt, response)
            self.facts_validated += 1
        
        # Store conversation
        self.conversation_history.append({
            'prompt': prompt,
            'response': response,
            'validated': validate_response,
        })
        
        return {
            'success': True,
            'response': response,
            'validation': validation_result if validate_response else None,
        }
    
    def _call_ollama(self, prompt: str) -> str:
        """Call Ollama API."""
        url = f"{self.ollama_url}/api/generate"
        
        data = {
            'model': self.model,
            'prompt': prompt,
            'stream': False
        }
        
        response = requests.post(url, json=data)
        response.raise_for_status()
        
        return response.json().get('response', '')
    
    def _enrich_prompt_with_memory(self, prompt: str) -> str:
        """Enrich prompt with relevant memories."""
        # Get recent relevant memories
        relevant_memories = self.memory_bank.recall(
            category="fact",
            min_confidence=0.7,
            limit=5
        )
        
        if not relevant_memories:
            return prompt
        
        # Add context from memories
        context = "Relevant facts from memory:\n"
        for mem in relevant_memories:
            context += f"- {mem.content} (confidence: {mem.confidence:.2f})\n"
        
        enriched = f"{context}\n\nQuery: {prompt}"
        return enriched
    
    def _validate_response(self, response: str) -> Dict:
        """Validate response against memory."""
        # Parse response for factual claims
        # (Simple version - can be enhanced with NLP)
        claims = self._extract_claims(response)
        
        invalid_claims = []
        uncertain_claims = []
        
        for claim in claims:
            is_valid, confidence, reason = self.hallucination_detector.check_claim(
                {'claim': claim},
                category="fact"
            )
            
            if not is_valid:
                if confidence < 0.3:
                    uncertain_claims.append(claim)
                else:
                    invalid_claims.append(claim)
        
        is_valid = len(invalid_claims) == 0
        
        return {
            'is_valid': is_valid,
            'invalid_claims': invalid_claims,
            'uncertain_claims': uncertain_claims,
            'confidence': 1.0 - (len(invalid_claims) * 0.3 + len(uncertain_claims) * 0.1),
        }
    
    def _extract_claims(self, text: str) -> List[str]:
        """Extract factual claims from text."""
        # Simple sentence splitting
        # Can be enhanced with proper NLP
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        # Filter for sentences that look like claims
        claims = []
        claim_indicators = ['is', 'are', 'was', 'were', 'has', 'have', 'will', 'would']
        
        for sentence in sentences:
            words = sentence.lower().split()
            if any(indicator in words for indicator in claim_indicators):
                claims.append(sentence)
        
        return claims
    
    def _store_facts_from_response(self, prompt: str, response: str) -> None:
        """Store facts from response in memory."""
        claims = self._extract_claims(response)
        
        for claim in claims:
            self.memory_bank.store(
                category="fact",
                content=claim,
                confidence=0.7,  # Medium confidence for AI-generated facts
                source=f"ollama:{self.model}"
            )
    
    def store_ground_truth(self, fact: str, confidence: float = 1.0) -> None:
        """
        Store verified ground truth.
        
        Args:
            fact: Verified fact
            confidence: Confidence level (usually 1.0 for ground truth)
        """
        self.memory_bank.store(
            category="fact",
            content=fact,
            confidence=confidence,
            source="ground_truth"
        )
        self.memory_bank.validated_count += 1
    
    def get_statistics(self) -> Dict:
        """Get comprehensive statistics."""
        return {
            'queries': self.total_queries,
            'hallucinations_prevented': self.hallucinations_prevented,
            'facts_validated': self.facts_validated,
            'memory_stats': self.memory_bank.get_statistics(),
            'state_stats': self.state_tracker.get_statistics(),
            'detection_stats': self.hallucination_detector.get_statistics(),
        }
    
    def save_memory(self) -> None:
        """Save memory to disk."""
        self.memory_bank.save()
    
    def reset_conversation(self) -> None:
        """Reset conversation history (keep memory)."""
        self.conversation_history = []


# Example usage
if __name__ == "__main__":
    print("Ollama with Memory System - Demo")
    print("="*60)
    
    # Create Ollama with memory
    ollama = OllamaWithMemory(memory_file="data/ollama_memory.json")
    
    # Store some ground truth facts
    print("\n1. Storing ground truth facts...")
    ground_truths = [
        "The theoretical maximum win rate for clearing all 22 cells in a 5x5 Mines game with 3 mines is 0.043%",
        "Hybrid Ultimate achieved a 0.87% win rate in the tournament",
        "The tournament consisted of 10,000,000 games total",
        "Senku Ishigami is an analytical strategy",
        "Q-Learning uses temporal difference learning",
    ]
    
    for fact in ground_truths:
        ollama.store_ground_truth(fact)
        print(f"  ✓ {fact[:60]}...")
    
    # Example query (would work if Ollama is running)
    print("\n2. Example query (requires Ollama server)...")
    print("   Query: 'What was the win rate in the tournament?'")
    print("   → Would retrieve from memory: 'Hybrid Ultimate: 0.87%'")
    print("   → Validates response against stored facts")
    print("   → Prevents hallucinations by checking contradictions")
    
    # Show statistics
    print("\n3. Statistics:")
    stats = ollama.get_statistics()
    print(f"   Memory entries: {stats['memory_stats']['total_memories']}")
    print(f"   Validated facts: {stats['memory_stats']['validated_count']}")
    print(f"   Consistency score: {stats['state_stats']['consistency_score']:.2f}")
    
    # Save memory
    ollama.save_memory()
    print("\n✓ Memory saved to disk")
    
    print("\n" + "="*60)
    print("Memory system ready to prevent hallucinations! 🧠✓")

