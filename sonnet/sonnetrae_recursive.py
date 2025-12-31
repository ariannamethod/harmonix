"""
SonnetRAE Recursive - Multi-level recursive semantic understanding

Hierarchical compression: lines → quatrains → couplet → full sonnet
Similar to HAiKU's rae_recursive.py but for sonnet structure.
"""

import json
import numpy as np
from typing import List, Tuple, Dict
from sonnetbrain import MLP, Value
from sonnetrae import SonnetRAE


class RecursiveNode:
    """Node in recursive sonnet tree."""

    def __init__(self, text: str, level: str, children: List['RecursiveNode'] = None):
        self.text = text
        self.level = level  # 'line', 'quatrain', 'couplet', 'sonnet'
        self.children = children or []
        self.embedding = None

    def __repr__(self):
        return f"RecursiveNode({self.level}, {len(self.children)} children)"


class SonnetRAERecursive:
    """
    Recursive AutoEncoder with structural awareness.

    Sonnet structure:
    - 3 quatrains (4 lines each) → compressed to 8D each
    - 1 couplet (2 lines) → compressed to 8D
    - Full sonnet → 32D (4 × 8D) → further compressed to 8D

    Learns hierarchical semantic patterns.
    """

    def __init__(self, state_path: str = 'state/sonnetrae_recursive.json'):
        self.state_path = state_path
        self.embedding_dim = 8

        # Base RAE for line-level encoding
        self.base_rae = SonnetRAE()

        # Quatrain encoder: 4 lines (4×8D=32D) → 8D
        self.quatrain_encoder = MLP(32, [24, 16, 8])

        # Couplet encoder: 2 lines (2×8D=16D) → 8D
        self.couplet_encoder = MLP(16, [12, 8])

        # Sonnet encoder: 3 quatrains + 1 couplet (4×8D=32D) → 8D
        self.sonnet_encoder = MLP(32, [24, 16, 8])

        self.lr = 0.005  # Lower LR for recursive
        self.observations = 0

    def parse_sonnet_structure(self, sonnet: str) -> RecursiveNode:
        """Parse sonnet into hierarchical structure."""
        lines = sonnet.strip().split('\n')[:14]

        # Pad to 14 lines
        while len(lines) < 14:
            lines.append("")

        # Build tree
        line_nodes = [RecursiveNode(line, 'line') for line in lines]

        # 3 quatrains
        quatrain1 = RecursiveNode('\n'.join(lines[0:4]), 'quatrain', line_nodes[0:4])
        quatrain2 = RecursiveNode('\n'.join(lines[4:8]), 'quatrain', line_nodes[4:8])
        quatrain3 = RecursiveNode('\n'.join(lines[8:12]), 'quatrain', line_nodes[8:12])

        # 1 couplet
        couplet = RecursiveNode('\n'.join(lines[12:14]), 'couplet', line_nodes[12:14])

        # Full sonnet
        root = RecursiveNode(sonnet, 'sonnet', [quatrain1, quatrain2, quatrain3, couplet])

        return root

    def encode_recursive(self, node: RecursiveNode) -> np.ndarray:
        """Recursively encode from bottom up."""

        if node.level == 'line':
            # Base case: encode single line
            features = self._extract_line_features(node.text)
            x = [Value(f) for f in features]
            encoded = self.base_rae.encoder(x)

            # Convert to numpy
            if isinstance(encoded, list):
                node.embedding = np.array([e.data if isinstance(e, Value) else e for e in encoded])
            else:
                node.embedding = np.array([encoded.data if isinstance(encoded, Value) else encoded])

            return node.embedding

        # Recursive case: encode children first
        child_embeddings = []
        for child in node.children:
            child_emb = self.encode_recursive(child)
            child_embeddings.append(child_emb)

        # Concatenate child embeddings
        combined = np.concatenate(child_embeddings)

        # Encode at this level
        if node.level == 'quatrain':
            # 4 lines × 8D = 32D → 8D
            x = [Value(f) for f in combined[:32]]
            encoded = self.quatrain_encoder(x)
        elif node.level == 'couplet':
            # 2 lines × 8D = 16D → 8D
            x = [Value(f) for f in combined[:16]]
            encoded = self.couplet_encoder(x)
        elif node.level == 'sonnet':
            # 4 parts × 8D = 32D → 8D
            x = [Value(f) for f in combined[:32]]
            encoded = self.sonnet_encoder(x)
        else:
            encoded = combined

        # Convert to numpy
        if isinstance(encoded, list):
            node.embedding = np.array([e.data if isinstance(e, Value) else e for e in encoded])
        else:
            node.embedding = np.array([encoded.data if isinstance(encoded, Value) else encoded])

        return node.embedding

    def _extract_line_features(self, line: str) -> List[float]:
        """Extract 20 features from a single line."""
        words = line.lower().split()

        features = [
            len(words),
            len(line),
            sum(len(w) for w in words) / max(len(words), 1),
            len(set(words)) / max(len(words), 1),
            line.count(','),
            line.count('.'),
            line.count('!'),
            line.count('?'),
        ]

        # Pad to 20
        features += [0.0] * (20 - len(features))
        return features[:20]

    def encode(self, sonnet: str) -> np.ndarray:
        """Encode sonnet with full recursive structure."""
        tree = self.parse_sonnet_structure(sonnet)
        return self.encode_recursive(tree)

    def get_quatrain_embeddings(self, sonnet: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get individual quatrain embeddings."""
        tree = self.parse_sonnet_structure(sonnet)
        self.encode_recursive(tree)

        return (
            tree.children[0].embedding,  # Q1
            tree.children[1].embedding,  # Q2
            tree.children[2].embedding,  # Q3
        )

    def get_couplet_embedding(self, sonnet: str) -> np.ndarray:
        """Get couplet embedding."""
        tree = self.parse_sonnet_structure(sonnet)
        self.encode_recursive(tree)

        return tree.children[3].embedding  # Couplet

    def structural_similarity(self, sonnet1: str, sonnet2: str) -> Dict[str, float]:
        """Compare structural similarity at multiple levels."""
        tree1 = self.parse_sonnet_structure(sonnet1)
        tree2 = self.parse_sonnet_structure(sonnet2)

        self.encode_recursive(tree1)
        self.encode_recursive(tree2)

        def cosine_sim(v1, v2):
            dot = np.dot(v1, v2)
            norm = np.linalg.norm(v1) * np.linalg.norm(v2)
            return dot / norm if norm > 0 else 0.0

        return {
            'quatrain1_sim': cosine_sim(tree1.children[0].embedding, tree2.children[0].embedding),
            'quatrain2_sim': cosine_sim(tree1.children[1].embedding, tree2.children[1].embedding),
            'quatrain3_sim': cosine_sim(tree1.children[2].embedding, tree2.children[2].embedding),
            'couplet_sim': cosine_sim(tree1.children[3].embedding, tree2.children[3].embedding),
            'full_sonnet_sim': cosine_sim(tree1.embedding, tree2.embedding),
        }


if __name__ == '__main__':
    rae = SonnetRAERecursive()

    test_sonnet = """When winter winds do blow and summer's heat
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

    print("Testing SonnetRAE Recursive...\n")

    # Full encoding
    compressed = rae.encode(test_sonnet)
    print(f"Full sonnet compressed to 8D: {compressed}")
    print(f"Vector norm: {np.linalg.norm(compressed):.3f}\n")

    # Quatrain embeddings
    q1, q2, q3 = rae.get_quatrain_embeddings(test_sonnet)
    print(f"Quatrain 1 embedding: {q1}")
    print(f"Quatrain 2 embedding: {q2}")
    print(f"Quatrain 3 embedding: {q3}\n")

    # Couplet embedding
    couplet = rae.get_couplet_embedding(test_sonnet)
    print(f"Couplet embedding: {couplet}\n")

    print("✓ Recursive sonnet encoding complete")
