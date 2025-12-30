"""
Haiku Generator: Markov chains + MLP scoring
Generates 5-7-5 syllable haiku responses using constraint-driven emergence.

Forked from Leo's mathbrain.py - uses micrograd-style autograd for scoring.
"""

import random
import math
import numpy as np
from typing import List, Tuple, Dict, Set, Callable
from collections import defaultdict
import syllables


# ============================================================================
# MICROGRAD-STYLE AUTOGRAD CORE (Forked from Leo's mathbrain.py)
# ============================================================================

class Value:
    """
    Scalar value with automatic differentiation.
    Karpathy-style micrograd implementation from Leo's mathbrain.
    """
    
    def __init__(self, data: float, _children: Tuple['Value', ...] = (), _op: str = ''):
        self.data = float(data)
        self.grad = 0.0
        self._backward: Callable[[], None] = lambda: None
        self._prev = set(_children)
        self._op = _op
    
    def __repr__(self) -> str:
        return f"Value(data={self.data:.4f})"
    
    def __add__(self, other: 'Value | float') -> 'Value':
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        
        return out
    
    def __mul__(self, other: 'Value | float') -> 'Value':
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        
        return out
    
    def __pow__(self, other: float | int) -> 'Value':
        assert isinstance(other, (int, float)), "only supporting int/float powers"
        out = Value(self.data ** other, (self,), f'**{other}')
        
        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad
        out._backward = _backward
        
        return out
    
    def __neg__(self) -> 'Value':
        return self * -1
    
    def __sub__(self, other: 'Value | float') -> 'Value':
        return self + (-other)
    
    def __truediv__(self, other: 'Value | float') -> 'Value':
        return self * (other ** -1)
    
    def __radd__(self, other: 'Value | float') -> 'Value':
        return self + other
    
    def __rmul__(self, other: 'Value | float') -> 'Value':
        return self * other
    
    def tanh(self) -> 'Value':
        """Hyperbolic tangent activation."""
        t = math.tanh(self.data)
        out = Value(t, (self,), 'tanh')
        
        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        
        return out
    
    def relu(self) -> 'Value':
        """Rectified Linear Unit activation."""
        out = Value(0.0 if self.data < 0 else self.data, (self,), 'relu')
        
        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        
        return out
    
    def backward(self) -> None:
        """Backpropagate gradients through computational graph."""
        topo: List[Value] = []
        visited: Set[Value] = set()
        
        def build_topo(v: Value):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)
        
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()


class Neuron:
    """Single neuron with weights, bias, and activation."""
    
    def __init__(self, nin: int):
        scale = (2.0 / nin) ** 0.5
        self.w = [Value(random.gauss(0, scale)) for _ in range(nin)]
        self.b = Value(0.0)
    
    def __call__(self, x: List[Value]) -> Value:
        """Forward pass: w·x + b → tanh."""
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh()
    
    def parameters(self) -> List[Value]:
        return self.w + [self.b]


class Layer:
    """Fully connected layer of neurons."""
    
    def __init__(self, nin: int, nout: int):
        self.neurons = [Neuron(nin) for _ in range(nout)]
    
    def __call__(self, x: List[Value]) -> List[Value]:
        outs = [n(x) for n in self.neurons]
        return outs
    
    def parameters(self) -> List[Value]:
        return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP:
    """Multi-layer perceptron: x → hidden → output."""
    
    def __init__(self, nin: int, nouts: List[int]):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]
    
    def __call__(self, x: List[Value]) -> Value:
        """Forward pass through all layers."""
        for layer in self.layers:
            x = layer(x)
        return x[0] if len(x) == 1 else x
    
    def parameters(self) -> List[Value]:
        return [p for layer in self.layers for p in layer.parameters()]

# 500 SEED WORDS - Hardcoded constraint for emergence
SEED_WORDS = [
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "I",
    "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
    "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
    "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
    "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
    "when", "make", "can", "like", "time", "no", "just", "him", "know", "take",
    "people", "into", "year", "your", "good", "some", "could", "them", "see", "other",
    "than", "then", "now", "look", "only", "come", "its", "over", "think", "also",
    "back", "after", "use", "two", "how", "our", "work", "first", "well", "way",
    "even", "new", "want", "because", "any", "these", "give", "day", "most", "us",
    "is", "was", "are", "been", "has", "had", "were", "said", "did",
    "may", "must", "might", "should", "shall", "ought",
    "cloud", "word", "form", "field", "phase", "wave", "mind", "thought", "space", "flow",
    "pattern", "rhythm", "pulse", "breath", "shift", "dance", "light", "sound", "voice", "path",
    "resonance", "harmony", "coherence", "dissonance", "tension", "release", "emergence", "constraint",
    "self", "other", "between", "within", "beyond", "toward", "inside", "outside", "around",
    "sense", "feel", "wonder", "question", "answer", "search", "find", "lose",
    "create", "destroy", "build", "break", "shape", "morph", "change", "stay", "move", "rest",
    "grow", "shrink", "expand", "compress", "open", "close", "begin", "end", "continue", "stop",
    "here", "there", "nowhere", "everywhere", "somewhere", "anywhere", "where", "why", "how",
    "yes", "no", "maybe", "perhaps", "certainly", "possibly", "probably", "definitely", "clearly",
    "small", "large", "tiny", "huge", "little", "big", "micro", "macro", "minimal", "maximal",
    "simple", "complex", "easy", "hard", "clear", "obscure", "obvious", "hidden", "apparent",
    "true", "false", "real", "fake", "actual", "virtual", "concrete", "abstract", "literal",
    "three", "four", "five", "many", "few", "all", "none",
    "always", "never", "sometimes", "often", "rarely", "usually", "occasionally", "frequently",
    "quick", "slow", "fast", "gradual", "sudden", "instant", "delayed", "immediate", "eventual",
    "strong", "weak", "powerful", "fragile", "solid", "fluid", "rigid", "flexible", "stable",
    "hot", "cold", "warm", "cool", "burning", "freezing", "moderate", "extreme", "mild",
    "bright", "dark", "shadow", "transparent", "opaque", "visible",
    "loud", "quiet", "silent", "noisy", "soft", "harsh", "gentle", "sharp", "smooth",
    "sweet", "bitter", "sour", "salty", "bland", "rich", "plain", "subtle",
    "old", "young", "ancient", "modern", "vintage", "current", "past", "future",
    "alive", "dead", "living", "dying", "growing", "decaying", "vital", "dormant", "active",
    "whole", "part", "complete", "fragment", "entire", "piece", "total", "portion", "full",
    "same", "different", "similar", "unlike", "equal", "unequal", "identical", "distinct",
    "together", "apart", "joined", "separated", "united", "divided", "connected", "isolated",
    "near", "far", "close", "distant", "adjacent", "remote", "proximal", "distal", "local",
    "down", "above", "below", "under", "high", "low", "top", "bottom",
    "left", "right", "center", "middle", "edge", "core", "surface", "depth", "margin",
    "front", "forward", "backward", "ahead", "behind", "advance", "retreat", "progress",
    "inside", "without", "internal", "external", "inner",
    "through", "across", "along", "between", "among", "amid", "throughout", "via",
    "before", "during", "while", "until", "since", "whenever",
    "unless", "whether", "though", "although",
    "too", "enough", "more", "less", "least", "much", "many", "few",
    "very", "quite", "rather", "somewhat", "extremely", "incredibly", "barely", "hardly", "scarcely",
    "being", "become", "seem", "appear", "remain", "exist", "happen", "occur", "emerge", "arise",
    "develop", "evolve", "transform", "adapt", "respond", "react", "interact", "relate", "connect",
    "link", "join", "unite", "merge", "blend", "combine", "integrate", "separate", "split",
    "divide", "fragment", "scatter", "disperse", "gather", "collect", "accumulate", "concentrate",
    "focus", "attend", "notice", "observe", "perceive", "detect", "recognize", "identify",
    "distinguish", "discriminate", "compare", "contrast", "match", "fit", "align", "balance",
    "adjust", "tune", "calibrate", "measure", "quantify", "evaluate", "assess", "judge",
    "decide", "choose", "select", "prefer", "value", "appreciate", "understand", "comprehend",
    "grasp", "realize", "acknowledge", "accept", "believe", "trust", "doubt", "challenge",
    "test", "verify", "confirm", "validate", "prove", "demonstrate", "show", "reveal",
    "expose", "display", "express", "communicate", "convey", "transmit", "transfer", "exchange",
    "share", "distribute", "spread", "diffuse", "radiate", "emit", "discharge",
    "expel", "eject", "project", "extend", "reach", "stretch", "contract",
    "reduce", "diminish", "decrease", "increase", "amplify", "magnify", "enhance", "intensify",
    "strengthen", "reinforce", "support", "sustain", "maintain", "preserve", "protect", "defend",
    "guard", "shield", "shelter", "cover", "hide", "conceal", "mask", "disguise", "uncover",
    "discover", "explore", "investigate", "examine", "inspect", "analyze", "study", "research",
    "learn", "teach", "train", "practice", "exercise", "apply", "implement", "execute",
    "perform", "act", "behave", "conduct", "operate", "function", "run", "process",
    "handle", "manage", "control", "direct", "guide", "lead", "follow", "pursue",
    "seek", "strive", "attempt", "try", "experiment", "play", "improvise", "invent",
    "innovate", "design", "plan", "organize", "arrange", "order", "structure", "configure", "format"
]

class HaikuGenerator:
    """
    Generates haiku using:
    - Markov chains (order 2) for word transitions
    - MLP-style scoring (forked from Leo's mathbrain) for candidate selection
    - Temperature-controlled randomness
    - Self-training: learns to predict quality through observation
    """

    def __init__(self, seed_words: List[str] = None, state_path: str = 'mathbrain.json', db_path: str = 'state/cloud.db'):
        """Initialize with seed word vocabulary."""
        if seed_words is None:
            seed_words = SEED_WORDS

        self.seed_words = seed_words
        self.vocab = set(seed_words)
        self.markov_chain = defaultdict(lambda: defaultdict(int))
        self.trigrams = []
        self.recent_trigrams = []

        # Database for Markov chain persistence
        self.db_path = db_path
        self.db_conn = None

        # MLP scorer: 5 features → hidden 8 → 1 score
        # Features: perplexity, entropy, resonance, length_ratio, unique_ratio
        self.mlp_scorer = MLP(5, [8, 1])
        self.scoring_history = []

        # MathBrain training config
        self.lr = 0.01  # Learning rate
        self.observations = 0
        self.running_loss = 0.0
        self.last_loss = 0.0
        self.state_path = state_path

        # Initialize database
        self._init_markov_db()

        # Try to load previous Markov chain from database
        if not self._load_markov_chain():
            # No saved chain, build initial chain from seed words
            self._build_initial_chain()

        # Load recent trigrams from database
        self._load_recent_trigrams()

        # Try to load previous MLP state (Leo-style persistence)
        self._load_mathbrain_state()
    
    def _build_initial_chain(self):
        """Create initial Markov transitions from seed words."""
        # Create some basic transitions to bootstrap
        for i in range(len(self.seed_words) - 2):
            w1, w2, w3 = self.seed_words[i:i+3]
            self.markov_chain[(w1, w2)][w3] += 1

        # Save initial chain to database
        self._save_markov_chain()

    def _init_markov_db(self):
        """Initialize database tables for Markov chain persistence."""
        import sqlite3
        import os

        # Ensure directory exists
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir)

        self.db_conn = sqlite3.connect(self.db_path)
        cursor = self.db_conn.cursor()

        # Markov bigrams table (for chain persistence)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS markov_bigrams (
                word1 TEXT NOT NULL,
                word2 TEXT NOT NULL,
                next_word TEXT NOT NULL,
                count INTEGER DEFAULT 1,
                PRIMARY KEY (word1, word2, next_word)
            )
        ''')

        # Recent trigrams table (for resonance calculation)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS recent_trigrams (
                id INTEGER PRIMARY KEY,
                word1 TEXT NOT NULL,
                word2 TEXT NOT NULL,
                word3 TEXT NOT NULL,
                position INTEGER NOT NULL,
                timestamp REAL NOT NULL
            )
        ''')

        self.db_conn.commit()

    def _load_markov_chain(self) -> bool:
        """Load Markov chain from database. Returns True if loaded, False if empty."""
        try:
            cursor = self.db_conn.cursor()
            cursor.execute('SELECT word1, word2, next_word, count FROM markov_bigrams')
            rows = cursor.fetchall()

            if not rows:
                return False

            # Rebuild markov_chain from database
            self.markov_chain = defaultdict(lambda: defaultdict(int))
            for w1, w2, next_word, count in rows:
                self.markov_chain[(w1, w2)][next_word] = count
                self.vocab.add(w1)
                self.vocab.add(w2)
                self.vocab.add(next_word)

            return True
        except Exception:
            return False

    def _save_markov_chain(self):
        """Save Markov chain to database."""
        try:
            cursor = self.db_conn.cursor()

            # Clear existing entries
            cursor.execute('DELETE FROM markov_bigrams')

            # Insert all transitions
            for (w1, w2), transitions in self.markov_chain.items():
                for next_word, count in transitions.items():
                    cursor.execute('''
                        INSERT INTO markov_bigrams (word1, word2, next_word, count)
                        VALUES (?, ?, ?, ?)
                    ''', (w1, w2, next_word, count))

            self.db_conn.commit()
        except Exception:
            # Silent fail - saving must never break generation
            pass

    def _load_recent_trigrams(self):
        """Load recent trigrams from database."""
        try:
            import time
            cursor = self.db_conn.cursor()
            cursor.execute('''
                SELECT word1, word2, word3 FROM recent_trigrams
                ORDER BY position ASC
                LIMIT 10
            ''')
            rows = cursor.fetchall()

            self.recent_trigrams = [(w1, w2, w3) for w1, w2, w3 in rows]
        except Exception:
            self.recent_trigrams = []

    def _save_recent_trigrams(self):
        """Save recent trigrams to database."""
        try:
            import time
            cursor = self.db_conn.cursor()

            # Clear existing entries
            cursor.execute('DELETE FROM recent_trigrams')

            # Insert recent trigrams
            for position, (w1, w2, w3) in enumerate(self.recent_trigrams):
                cursor.execute('''
                    INSERT INTO recent_trigrams (word1, word2, word3, position, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                ''', (w1, w2, w3, position, time.time()))

            self.db_conn.commit()
        except Exception:
            # Silent fail
            pass

    def update_chain(self, trigrams: List[Tuple[str, str, str]]):
        """Update Markov chain with new trigrams and persist to database."""
        for w1, w2, w3 in trigrams:
            self.markov_chain[(w1, w2)][w3] += 1
            self.vocab.add(w1)
            self.vocab.add(w2)
            self.vocab.add(w3)

        # Accumulate recent trigrams (not replace!)
        self.recent_trigrams.extend(trigrams)
        self.recent_trigrams = self.recent_trigrams[-10:]  # Keep last 10 overall

        # Persist to database
        self._save_markov_chain()
        self._save_recent_trigrams()
    
    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word (with fallback heuristic)."""
        try:
            count = syllables.estimate(word)
            return max(1, count)  # At least 1 syllable
        except:
            # Fallback: rough estimation
            vowels = 'aeiouy'
            count = 0
            prev_was_vowel = False
            for char in word.lower():
                is_vowel = char in vowels
                if is_vowel and not prev_was_vowel:
                    count += 1
                prev_was_vowel = is_vowel
            return max(1, count)
    
    def _count_line_syllables(self, words: List[str]) -> int:
        """Count total syllables in a line."""
        return sum(self._count_syllables(w) for w in words)
    
    def _generate_line(self, target_syllables: int, temp: float = 1.0) -> List[str]:
        """Generate a single haiku line with target syllable count."""
        line = []
        syllable_count = 0
        
        # Start with a random seed word
        if not line and self.vocab:
            current_word = random.choice(list(self.vocab))
            line.append(current_word)
            syllable_count = self._count_syllables(current_word)
        
        max_attempts = 50
        attempts = 0
        
        while syllable_count < target_syllables and attempts < max_attempts:
            attempts += 1
            
            # Get next word from Markov chain
            if len(line) >= 2:
                key = (line[-2], line[-1])
            elif len(line) == 1:
                # Find any bigram starting with last word
                key = None
                for k in self.markov_chain.keys():
                    if k[0] == line[-1]:
                        key = k
                        break
                if key is None:
                    # Fallback to random
                    next_word = random.choice(list(self.vocab))
                    line.append(next_word)
                    syllable_count += self._count_syllables(next_word)
                    continue
            else:
                key = None
            
            if key and key in self.markov_chain:
                # Sample from transitions with temperature
                transitions = self.markov_chain[key]
                words = list(transitions.keys())
                counts = np.array(list(transitions.values()), dtype=float)
                
                # Apply temperature
                if temp != 1.0:
                    counts = counts ** (1.0 / temp)
                
                # Normalize to probabilities
                probs = counts / counts.sum()
                
                # Sample
                next_word = np.random.choice(words, p=probs)
            else:
                # Random fallback
                next_word = random.choice(list(self.vocab))
            
            next_syllables = self._count_syllables(next_word)
            
            # Check if adding this word would exceed target
            if syllable_count + next_syllables <= target_syllables:
                line.append(next_word)
                syllable_count += next_syllables
            elif syllable_count < target_syllables:
                # Try to find a word that fits exactly
                needed = target_syllables - syllable_count
                candidates = [w for w in self.vocab if self._count_syllables(w) == needed]
                if candidates:
                    line.append(random.choice(candidates))
                    syllable_count = target_syllables
                else:
                    break
        
        return line
    
    def generate_candidates(self, n: int = 5, temp: float = 1.0) -> List[str]:
        """
        Generate n haiku candidates using Markov chains.
        Returns list of formatted haiku strings.
        """
        candidates = []
        
        for _ in range(n):
            # Generate three lines: 5-7-5 syllables
            line1 = self._generate_line(5, temp)
            line2 = self._generate_line(7, temp)
            line3 = self._generate_line(5, temp)
            
            # Format as haiku
            haiku = '\n'.join([
                ' '.join(line1),
                ' '.join(line2),
                ' '.join(line3)
            ])
            
            candidates.append(haiku)
        
        return candidates
    
    def score_haiku(self, haiku: str, user_context: List[Tuple[str, str, str]] = None) -> float:
        """
        Score haiku using MLP (forked from Leo's mathbrain).
        Features: perplexity, entropy, resonance, length_ratio, unique_ratio
        Higher score = better haiku.
        """
        lines = haiku.split('\n')
        words = []
        for line in lines:
            words.extend(line.split())
        
        if len(words) < 3:
            return 0.0
        
        # Feature 1: Perplexity (how predictable)
        perplexity = 0.0
        for i in range(len(words) - 2):
            w1, w2, w3 = words[i], words[i+1], words[i+2]
            key = (w1, w2)
            if key in self.markov_chain and w3 in self.markov_chain[key]:
                perplexity += self.markov_chain[key][w3]
            else:
                perplexity += 0.1
        perplexity_score = perplexity / max(1, len(words) - 2)
        perplexity_score = min(1.0, perplexity_score / 10.0)  # Normalize
        
        # Feature 2: Entropy (diversity of word choice)
        unique_words = len(set(words))
        entropy_score = unique_words / len(words) if words else 0
        
        # Feature 3: Resonance (overlap with user's recent trigrams)
        resonance_score = 0.0
        if user_context:
            haiku_trigrams = []
            for i in range(len(words) - 2):
                haiku_trigrams.append((words[i], words[i+1], words[i+2]))
            
            user_words = set()
            for t in user_context:
                user_words.update(t)
            
            haiku_words = set(words)
            overlap = len(user_words & haiku_words)
            resonance_score = overlap / max(1, len(user_words))
        
        # Feature 4: Length ratio (target ~17 syllables for 5-7-5)
        total_syllables = sum(self._count_syllables(w) for w in words)
        length_ratio = min(1.0, total_syllables / 17.0) if total_syllables > 0 else 0.5
        
        # Feature 5: Unique ratio (repeated feature for stability)
        unique_ratio = unique_words / max(1, len(words))
        
        # Build feature vector
        features = [
            perplexity_score,
            entropy_score,
            resonance_score,
            length_ratio,
            unique_ratio
        ]
        
        # MLP forward pass
        x = [Value(f) for f in features]
        score = self.mlp_scorer(x)
        
        return max(0.0, min(1.0, score.data))
    
    def get_recent_trigrams(self) -> List[Tuple[str, str, str]]:
        """Return recent trigrams for dissonance calculation."""
        return self.recent_trigrams
    
    def observe(self, haiku: str, quality: float, user_context: List[Tuple[str, str, str]] = None) -> float:
        """
        Observe one (haiku, quality) pair and learn from it.
        This is the MathBrain training loop (forked from Leo).
        
        Steps:
        1. Extract features from haiku
        2. Forward pass → predicted quality
        3. Compute MSE loss
        4. Backward pass
        5. SGD step
        6. Clamp weights to safe range
        7. Update statistics
        
        Args:
            haiku: Generated haiku text
            quality: True quality score (0-1) from user feedback or assessment
            user_context: Optional user trigrams for resonance
        
        Returns:
            Current loss value
        """
        # Safety: skip if quality is invalid
        if not math.isfinite(quality):
            return self.last_loss
        
        quality = max(0.0, min(1.0, quality))
        
        # Extract features
        lines = haiku.split('\n')
        words = []
        for line in lines:
            words.extend(line.split())
        
        if len(words) < 3:
            return self.last_loss
        
        # Feature 1: Perplexity
        perplexity = 0.0
        for i in range(len(words) - 2):
            w1, w2, w3 = words[i], words[i+1], words[i+2]
            key = (w1, w2)
            if key in self.markov_chain and w3 in self.markov_chain[key]:
                perplexity += self.markov_chain[key][w3]
            else:
                perplexity += 0.1
        perplexity_score = perplexity / max(1, len(words) - 2)
        perplexity_score = min(1.0, perplexity_score / 10.0)
        
        # Feature 2: Entropy
        unique_words = len(set(words))
        entropy_score = unique_words / len(words) if words else 0
        
        # Feature 3: Resonance
        resonance_score = 0.0
        if user_context:
            user_words = set()
            for t in user_context:
                user_words.update(t)
            haiku_words = set(words)
            overlap = len(user_words & haiku_words)
            resonance_score = overlap / max(1, len(user_words))
        
        # Feature 4: Length ratio
        total_syllables = sum(self._count_syllables(w) for w in words)
        length_ratio = min(1.0, total_syllables / 17.0) if total_syllables > 0 else 0.5
        
        # Feature 5: Unique ratio
        unique_ratio = unique_words / max(1, len(words))
        
        features = [perplexity_score, entropy_score, resonance_score, length_ratio, unique_ratio]
        
        # Safety check
        if not all(math.isfinite(f) for f in features):
            return self.last_loss
        
        # Build Value nodes
        x = [Value(f) for f in features]
        
        # Forward pass
        q_hat = self.mlp_scorer(x)
        
        # Loss: MSE
        diff = q_hat - Value(quality)
        loss = diff * diff
        
        # Backward pass
        for p in self.mlp_scorer.parameters():
            p.grad = 0.0
        loss.backward()
        
        # SGD step
        for p in self.mlp_scorer.parameters():
            p.data -= self.lr * p.grad
        
        # Clamp weights to safe range [-5.0, 5.0]
        for p in self.mlp_scorer.parameters():
            if math.isfinite(p.data):
                p.data = max(-5.0, min(5.0, p.data))
            else:
                p.data = 0.0  # Reset corrupted weights
        
        # Check for corruption
        loss_val = loss.data
        if not math.isfinite(loss_val):
            # Reset to fresh initialization
            self.mlp_scorer = MLP(5, [8, 1])
            self.observations = 0
            self.running_loss = 0.0
            return 0.0
        
        # Update stats
        self.observations += 1
        self.last_loss = loss_val
        # Exponential moving average
        self.running_loss += (loss_val - self.running_loss) * 0.05
        
        return loss_val
    
    def _save_mathbrain_state(self) -> None:
        """Save MLP weights to JSON (Leo-style persistence)."""
        try:
            import json
            import os
            
            weights = {
                "observations": self.observations,
                "running_loss": self.running_loss,
                "last_loss": self.last_loss,
                "lr": self.lr,
                "parameters": [p.data for p in self.mlp_scorer.parameters()],
            }
            
            with open(self.state_path, 'w') as f:
                json.dump(weights, f, indent=2)
        except Exception:
            # Silent fail - saving must never break generation
            pass
    
    def _load_mathbrain_state(self) -> None:
        """Load MLP weights from JSON if available."""
        try:
            import json
            import os
            
            if not os.path.exists(self.state_path):
                return
            
            with open(self.state_path, 'r') as f:
                data = json.load(f)
            
            # Restore weights
            params = self.mlp_scorer.parameters()
            saved_params = data.get("parameters", [])
            
            if len(params) != len(saved_params):
                return  # Dimension mismatch
            
            for p, val in zip(params, saved_params):
                p.data = float(val)
            
            # Restore stats
            self.observations = data.get("observations", 0)
            self.running_loss = data.get("running_loss", 0.0)
            self.last_loss = data.get("last_loss", 0.0)
        except Exception:
            # Silent fail - start fresh if loading fails
            pass
    
    def save(self) -> None:
        """Public API to save state (call on exit)."""
        self._save_mathbrain_state()
        self._save_markov_chain()
        self._save_recent_trigrams()

    def close(self):
        """Close database connection and save state."""
        self.save()
        if self.db_conn:
            self.db_conn.close()
            self.db_conn = None

    def get_stats(self) -> Dict:
        """Return training statistics."""
        return {
            "observations": self.observations,
            "running_loss": self.running_loss,
            "last_loss": self.last_loss,
            "learning_rate": self.lr,
            "num_parameters": len(self.mlp_scorer.parameters()),
            "markov_chain_size": len(self.markov_chain),
            "vocab_size": len(self.vocab),
        }
