"""
Haiku Generator: Markov chains + MLP scoring
Generates 5-7-5 syllable haiku responses using constraint-driven emergence.
"""

import random
import numpy as np
from typing import List, Tuple, Dict
from collections import defaultdict
import syllables

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
    - MLP-style scoring for candidate selection
    - Temperature-controlled randomness
    """
    
    def __init__(self, seed_words: List[str]):
        """Initialize with seed word vocabulary."""
        self.seed_words = seed_words
        self.vocab = set(seed_words)
        self.markov_chain = defaultdict(lambda: defaultdict(int))
        self.trigrams = []
        self.recent_trigrams = []
        
        # Build initial Markov chain from seed words
        self._build_initial_chain()
    
    def _build_initial_chain(self):
        """Create initial Markov transitions from seed words."""
        # Create some basic transitions to bootstrap
        for i in range(len(self.seed_words) - 2):
            w1, w2, w3 = self.seed_words[i:i+3]
            self.markov_chain[(w1, w2)][w3] += 1
    
    def update_chain(self, trigrams: List[Tuple[str, str, str]]):
        """Update Markov chain with new trigrams."""
        for w1, w2, w3 in trigrams:
            self.markov_chain[(w1, w2)][w3] += 1
            self.vocab.add(w1)
            self.vocab.add(w2)
            self.vocab.add(w3)
        
        self.recent_trigrams = trigrams[-10:]  # Keep last 10 for resonance
    
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
        Score haiku based on perplexity, entropy, and resonance.
        Higher score = better haiku.
        """
        lines = haiku.split('\n')
        words = []
        for line in lines:
            words.extend(line.split())
        
        if len(words) < 3:
            return 0.0
        
        # Perplexity: how predictable (lower is better, so we invert)
        perplexity = 0.0
        for i in range(len(words) - 2):
            w1, w2, w3 = words[i], words[i+1], words[i+2]
            key = (w1, w2)
            if key in self.markov_chain and w3 in self.markov_chain[key]:
                # Higher count = lower perplexity = higher score
                perplexity += self.markov_chain[key][w3]
            else:
                perplexity += 0.1  # Penalty for unseen
        
        perplexity_score = perplexity / max(1, len(words) - 2)
        
        # Entropy: diversity of word choice (moderate is best)
        unique_words = len(set(words))
        entropy_score = unique_words / len(words) if words else 0
        
        # Resonance: overlap with user's recent trigrams
        resonance_score = 0.0
        if user_context:
            haiku_trigrams = []
            for i in range(len(words) - 2):
                haiku_trigrams.append((words[i], words[i+1], words[i+2]))
            
            # Count overlapping words
            user_words = set()
            for t in user_context:
                user_words.update(t)
            
            haiku_words = set(words)
            overlap = len(user_words & haiku_words)
            resonance_score = overlap / max(1, len(user_words))
        
        # Combine scores (weighted)
        total_score = (
            0.3 * perplexity_score +
            0.3 * entropy_score +
            0.4 * resonance_score
        )
        
        return total_score
    
    def get_recent_trigrams(self) -> List[Tuple[str, str, str]]:
        """Return recent trigrams for dissonance calculation."""
        return self.recent_trigrams
