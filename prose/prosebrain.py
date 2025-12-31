"""
ProseBrain - MLP-based prose quality scorer

Forked from Leo's mathbrain.py (via HAiKU) - uses micrograd-style autograd
for learning what makes a good prose.

Unlike formatter.validate() (rule-based), ProseBrain LEARNS from examples.
"""

import json
import random
import math
from typing import List, Tuple, Dict
from dataclasses import dataclass


# ============================================================================
# MICROGRAD-STYLE AUTOGRAD CORE
# ============================================================================

class Value:
    """
    Karpathy-style micrograd implementation.
    Scalar value with autograd support.
    """

    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad
        out._backward = _backward

        return out

    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward

        return out

    def backward(self):
        # Topological sort
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = 1.0
        for v in reversed(topo):
            v._backward()

    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other**-1
    def __rtruediv__(self, other): return other * self**-1
    def __repr__(self): return f"Value(data={self.data}, grad={self.grad})"


# ============================================================================
# MLP COMPONENTS
# ============================================================================

class Neuron:
    """Single neuron with weights and bias."""

    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0.0)

    def __call__(self, x):
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh()

    def parameters(self):
        return self.w + [self.b]


class Layer:
    """Layer of neurons."""

    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class MLP:
    """Multi-layer perceptron."""

    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


# ============================================================================
# SONNET BRAIN
# ============================================================================

@dataclass
class SonnetFeatures:
    """Features extracted from a prose for MLP scoring."""
    line_count: float           # Number of lines (target: 14)
    avg_syllables: float        # Avg syllables per line (target: ~10)
    syllable_variance: float    # Consistency of meter
    avg_word_length: float      # Lexical complexity
    unique_word_ratio: float    # Vocabulary diversity
    shakespeare_score: float    # Presence of Shakespearean words
    coherence_score: float      # Line-to-line semantic coherence
    novelty: float              # From harmonix (cloud comparison)

    def to_vector(self) -> List[float]:
        """Convert to input vector for MLP."""
        return [
            self.line_count / 14.0,              # Normalize to [0, 1]
            self.avg_syllables / 15.0,
            min(self.syllable_variance / 5.0, 1.0),
            self.avg_word_length / 10.0,
            self.unique_word_ratio,
            self.shakespeare_score,
            self.coherence_score,
            self.novelty
        ]


class ProseBrain:
    """
    MLP-based prose scorer that LEARNS quality.

    Architecture: 8 inputs → 16 hidden → 8 hidden → 1 output
    Trained on (prose_features, human_quality_score) pairs.
    """

    def __init__(self, state_path: str = 'prosebrain.json'):
        self.state_path = state_path
        self.mlp = MLP(8, [16, 8, 1])  # 8 features → 1 score

        self.observations = 0
        self.running_loss = 0.0
        self.last_loss = 0.0
        self.lr = 0.01  # Learning rate

        self._load_state()

    def extract_features(self, prose: str) -> SonnetFeatures:
        """Extract features from prose text."""
        lines = prose.strip().split('\n')
        words = prose.lower().split()

        # Line count
        line_count = float(len(lines))

        # Syllables (simplified - count vowel groups)
        def count_syllables(text):
            vowels = 'aeiouy'
            text = text.lower()
            count = 0
            prev_was_vowel = False
            for char in text:
                is_vowel = char in vowels
                if is_vowel and not prev_was_vowel:
                    count += 1
                prev_was_vowel = is_vowel
            return max(1, count)

        syllable_counts = [count_syllables(line) for line in lines]
        avg_syllables = sum(syllable_counts) / len(syllable_counts) if syllable_counts else 0.0
        syllable_variance = sum((s - avg_syllables)**2 for s in syllable_counts) / len(syllable_counts) if syllable_counts else 0.0

        # Word stats
        avg_word_length = sum(len(w) for w in words) / len(words) if words else 0.0
        unique_word_ratio = len(set(words)) / len(words) if words else 0.0

        # Shakespeare vocabulary (simple heuristic)
        shakespeare_words = {'thou', 'thy', 'thee', 'art', 'doth', 'hath', 'tis',
                            'ere', 'wherefore', 'hence', 'thence', 'whence'}
        shakespeare_score = sum(1 for w in words if w in shakespeare_words) / len(words) if words else 0.0

        # Coherence (word overlap between consecutive lines)
        coherence_scores = []
        for i in range(len(lines) - 1):
            words1 = set(lines[i].lower().split())
            words2 = set(lines[i+1].lower().split())
            overlap = len(words1 & words2) / max(len(words1 | words2), 1)
            coherence_scores.append(overlap)
        coherence_score = sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0.0

        # Novelty (default - should come from harmonix)
        novelty = 0.5

        return SonnetFeatures(
            line_count=line_count,
            avg_syllables=avg_syllables,
            syllable_variance=syllable_variance,
            avg_word_length=avg_word_length,
            unique_word_ratio=unique_word_ratio,
            shakespeare_score=shakespeare_score,
            coherence_score=coherence_score,
            novelty=novelty
        )

    def score(self, prose: str) -> float:
        """
        Score a prose using MLP.
        Returns value in [0, 1] range.
        """
        features = self.extract_features(prose)
        x = [Value(f) for f in features.to_vector()]

        # Forward pass
        y = self.mlp(x)

        # Sigmoid to [0, 1]
        score = 1.0 / (1.0 + math.exp(-y.data))

        return score

    def learn(self, prose: str, target_quality: float):
        """
        Update MLP weights based on feedback.

        Args:
            prose: Sonnet text
            target_quality: Human-provided quality score [0, 1]
        """
        features = self.extract_features(prose)
        x = [Value(f) for f in features.to_vector()]

        # Forward pass
        y = self.mlp(x)

        # Loss: MSE
        target = Value(target_quality)
        loss = (y - target) ** 2

        # Backward pass
        for p in self.mlp.parameters():
            p.grad = 0.0
        loss.backward()

        # Update weights (SGD)
        for p in self.mlp.parameters():
            p.data -= self.lr * p.grad

        # Track metrics
        self.observations += 1
        self.running_loss += loss.data
        self.last_loss = loss.data

        # Save state periodically
        if self.observations % 10 == 0:
            self._save_state()

    def _load_state(self):
        """Load MLP parameters from JSON."""
        try:
            with open(self.state_path, 'r') as f:
                state = json.load(f)

            self.observations = state.get('observations', 0)
            self.running_loss = state.get('running_loss', 0.0)
            self.last_loss = state.get('last_loss', 0.0)
            self.lr = state.get('lr', 0.01)

            params = state.get('parameters', [])
            if params:
                mlp_params = self.mlp.parameters()
                for i, p in enumerate(mlp_params):
                    if i < len(params):
                        p.data = params[i]

            print(f"✓ Loaded prosebrain state: {self.observations} observations")

        except FileNotFoundError:
            print("⚠️  No saved prosebrain state, starting fresh")

    def _save_state(self):
        """Save MLP parameters to JSON."""
        state = {
            'observations': self.observations,
            'running_loss': self.running_loss,
            'last_loss': self.last_loss,
            'lr': self.lr,
            'parameters': [p.data for p in self.mlp.parameters()]
        }

        with open(self.state_path, 'w') as f:
            json.dump(state, f, indent=2)


# ============================================================================
# TESTING
# ============================================================================

if __name__ == '__main__':
    brain = ProseBrain()

    # Test prose
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

    print("Testing ProseBrain MLP...\n")

    # Initial score
    score1 = brain.score(test_prose)
    print(f"Initial score: {score1:.3f}")

    # Train on this being a good prose
    for i in range(10):
        brain.learn(test_prose, target_quality=0.9)
        score = brain.score(test_prose)
        print(f"After training {i+1}: {score:.3f} (loss={brain.last_loss:.4f})")

    print(f"\n✓ ProseBrain trained on {brain.observations} observations")
