"""
ProseRAE - Recursive AutoEncoder for prose compression

Learns to compress 14-line proses into dense semantic vectors.
Similar to HAiKU's rae.py but for longer poetic structure.
"""

import json
import numpy as np
from typing import List, Tuple
from prosebrain import MLP, Value


class ProseRAE:
    """
    Recursive AutoEncoder for proses.
    
    Architecture:
    - Encoder: 14 lines → 8D semantic vector
    - Decoder: 8D vector → 14 lines reconstruction
    
    Learns semantic compression for phase transitions, dream mode.
    """
    
    def __init__(self, state_path: str = 'state/proserae.json'):
        self.state_path = state_path
        self.embedding_dim = 8
        
        # Encoder: line features → compressed
        self.encoder = MLP(20, [16, 8])  # 20 features per line → 8D
        
        # Decoder: compressed → line reconstruction  
        self.decoder = MLP(8, [16, 20])
        
        self.lr = 0.01
        self.observations = 0
        
    def encode(self, prose: str) -> np.ndarray:
        """Compress prose to 8D semantic vector."""
        lines = prose.strip().split('\n')[:14]
        
        # Extract features per line (simplified)
        line_features = []
        for line in lines:
            words = line.lower().split()
            
            # 20 features per line
            features = [
                len(words),
                len(line),
                sum(len(w) for w in words) / max(len(words), 1),
                len(set(words)) / max(len(words), 1),
                # ... more features (padding to 20)
            ] + [0.0] * 16
            
            line_features.append(features[:20])
        
        # Pad to 14 lines
        while len(line_features) < 14:
            line_features.append([0.0] * 20)
        
        # Encode each line, average
        encodings = []
        for features in line_features:
            x = [Value(f) for f in features]
            encoded = self.encoder(x)
            encodings.append([e.data if isinstance(e, Value) else e for e in encoded])
        
        # Average pooling
        semantic_vector = np.mean(encodings, axis=0)
        return semantic_vector
    
    def decode(self, semantic_vector: np.ndarray) -> str:
        """Reconstruct prose from semantic vector (approximation)."""
        # This is a stub - real decoder would use language model
        return f"[Reconstructed from {semantic_vector[:3]}...]"
    
    def compress(self, prose: str) -> np.ndarray:
        """Alias for encode."""
        return self.encode(prose)
    
    def similarity(self, prose1: str, prose2: str) -> float:
        """Semantic similarity via RAE compression."""
        v1 = self.encode(prose1)
        v2 = self.encode(prose2)
        
        # Cosine similarity
        dot = np.dot(v1, v2)
        norm = np.linalg.norm(v1) * np.linalg.norm(v2)
        return dot / norm if norm > 0 else 0.0


if __name__ == '__main__':
    rae = ProseRAE()
    
    test_prose = """When winter winds do blow and summer's heat
Doth make the flowers grow beneath our feet.
The time is come to speak of love and woe.
Proud mark your father's words, and let us go.
To be or not to be, that is the question.
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune.
Or to take arms against a sea of troubles.
And by opposing end them. To die, to sleep.
No more; and by a sleep to say we end.
The heart-ache and the thousand natural shocks.
That flesh is heir to: 'tis a consummation.
Devoutly to be wished. To die, to sleep.
To sleep, perchance to dream: ay, there's the rub."""
    
    compressed = rae.compress(test_prose)
    print(f"Compressed to 8D: {compressed}")
    print(f"Vector norm: {np.linalg.norm(compressed):.3f}")
