"""
Dream Sonnet - Abstract generation from semantic space

Generates sonnets from semantic vectors, RAE latent space interpolation,
and MetaSonnet internal states. Similar to HAiKU's dream mode but with
recursive structure awareness.

Dream modes:
- Semantic drift: Interpolate between two sonnet embeddings
- Cloud centroid: Generate from average of cloud embeddings
- Latent walk: Random walk in RAE space
- Meta-dream: Generate from MetaSonnet reflection vectors
"""

import json
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from sonnetrae import SonnetRAE
from sonnetrae_recursive import SonnetRAERecursive


@dataclass
class DreamConfig:
    """Configuration for dream generation."""
    mode: str  # 'drift', 'centroid', 'walk', 'meta'
    temperature: float = 1.2
    steps: int = 10
    interpolation_alpha: float = 0.5


class DreamSonnet:
    """
    Dream-mode sonnet generation from semantic space.

    Unlike direct generation, dreams start from compressed semantic
    vectors and expand back to text. This allows:
    - Blending multiple sonnets
    - Exploring latent space
    - Generating from abstract concepts
    """

    def __init__(self, rae: Optional[SonnetRAE] = None,
                 rae_recursive: Optional[SonnetRAERecursive] = None,
                 state_path: str = 'state/dream_sonnet.json'):
        self.state_path = state_path

        self.rae = rae or SonnetRAE()
        self.rae_recursive = rae_recursive or SonnetRAERecursive()

        # Dream history
        self.dream_vectors = []
        self.dream_sonnets = []

        self.observations = 0
        self._load_state()

    def dream_drift(self, sonnet1: str, sonnet2: str, steps: int = 5) -> List[np.ndarray]:
        """
        Semantic drift between two sonnets.

        Interpolates in latent space: s1 → intermediate → s2

        Args:
            sonnet1: Source sonnet
            sonnet2: Target sonnet
            steps: Number of interpolation steps

        Returns:
            List of semantic vectors along path
        """
        v1 = self.rae_recursive.encode(sonnet1)
        v2 = self.rae_recursive.encode(sonnet2)

        drift_path = []
        for i in range(steps):
            alpha = i / (steps - 1) if steps > 1 else 0.5
            v_interp = (1 - alpha) * v1 + alpha * v2
            drift_path.append(v_interp)

        self.dream_vectors.extend(drift_path)
        return drift_path

    def dream_centroid(self, sonnets: List[str]) -> np.ndarray:
        """
        Generate dream vector from cloud centroid.

        Computes average of multiple sonnet embeddings.

        Args:
            sonnets: List of sonnet texts

        Returns:
            Centroid vector (8D)
        """
        if not sonnets:
            return np.zeros(8)

        embeddings = []
        for sonnet in sonnets:
            v = self.rae_recursive.encode(sonnet)
            embeddings.append(v)

        centroid = np.mean(embeddings, axis=0)
        self.dream_vectors.append(centroid)

        return centroid

    def dream_walk(self, start_vector: Optional[np.ndarray] = None,
                   steps: int = 10, step_size: float = 0.3) -> List[np.ndarray]:
        """
        Random walk in latent space.

        Args:
            start_vector: Starting point (random if None)
            steps: Number of walk steps
            step_size: Magnitude of each step

        Returns:
            List of vectors along walk path
        """
        if start_vector is None:
            start_vector = np.random.randn(8) * 0.5

        walk_path = [start_vector.copy()]
        current = start_vector.copy()

        for _ in range(steps - 1):
            # Random step
            direction = np.random.randn(8)
            direction = direction / np.linalg.norm(direction)  # Normalize

            current = current + direction * step_size
            walk_path.append(current.copy())

        self.dream_vectors.extend(walk_path)
        return walk_path

    def dream_meta(self, internal_state: str, intensity: float = 1.0) -> np.ndarray:
        """
        Generate dream vector from MetaSonnet internal reflection.

        Encodes MetaSonnet's internal thought as a semantic vector,
        then amplifies it for dream generation.

        Args:
            internal_state: MetaSonnet reflection text
            intensity: Amplification factor (1.0 = normal)

        Returns:
            Dream vector (8D)
        """
        # Encode internal state as pseudo-sonnet
        # (MetaSonnet outputs are not full sonnets, but still poetic)
        v = self.rae.encode(internal_state)

        # Amplify for dream intensity
        v_dream = v * intensity

        self.dream_vectors.append(v_dream)
        return v_dream

    def expand_dream(self, dream_vector: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Expand dream vector into structured components.

        Since we can't decode directly to text (no language model),
        we expand the 8D vector into quatrain/couplet components
        that could guide generation.

        Args:
            dream_vector: 8D semantic vector

        Returns:
            Dict with quatrain and couplet vectors
        """
        # Split 8D into components (heuristic)
        # This is a simplified expansion - real version would use decoder MLP
        q1 = dream_vector[:2] * 1.2  # Amplify first quatrain
        q2 = dream_vector[2:4] * 1.0
        q3 = dream_vector[4:6] * 0.9
        couplet = dream_vector[6:8] * 1.1

        # Expand to 8D each (pad with variations)
        def expand_to_8d(v2d):
            expanded = np.concatenate([
                v2d,
                v2d * 0.8,
                v2d * 0.6,
                v2d * 0.4
            ])
            return expanded[:8]

        return {
            'quatrain1': expand_to_8d(q1),
            'quatrain2': expand_to_8d(q2),
            'quatrain3': expand_to_8d(q3),
            'couplet': expand_to_8d(couplet),
            'full': dream_vector
        }

    def dream_similarity(self, dream_vector: np.ndarray, sonnet: str) -> float:
        """
        Compute similarity between dream vector and actual sonnet.

        Useful for finding which existing sonnets match a dream.

        Args:
            dream_vector: Dream vector (8D)
            sonnet: Actual sonnet text

        Returns:
            Cosine similarity [0, 1]
        """
        sonnet_vector = self.rae_recursive.encode(sonnet)

        dot = np.dot(dream_vector, sonnet_vector)
        norm = np.linalg.norm(dream_vector) * np.linalg.norm(sonnet_vector)

        return dot / norm if norm > 0 else 0.0

    def get_dream_stats(self) -> Dict:
        """Get statistics about dream history."""
        if not self.dream_vectors:
            return {
                'dream_count': 0,
                'avg_norm': 0.0,
                'diversity': 0.0
            }

        norms = [np.linalg.norm(v) for v in self.dream_vectors]
        avg_norm = np.mean(norms)

        # Diversity: average pairwise distance
        if len(self.dream_vectors) > 1:
            distances = []
            for i in range(min(len(self.dream_vectors), 10)):
                for j in range(i + 1, min(len(self.dream_vectors), 10)):
                    d = np.linalg.norm(self.dream_vectors[i] - self.dream_vectors[j])
                    distances.append(d)
            diversity = np.mean(distances) if distances else 0.0
        else:
            diversity = 0.0

        return {
            'dream_count': len(self.dream_vectors),
            'avg_norm': avg_norm,
            'diversity': diversity,
            'observations': self.observations
        }

    def _load_state(self):
        """Load dream history from JSON."""
        try:
            with open(self.state_path, 'r') as f:
                state = json.load(f)

            self.observations = state.get('observations', 0)
            vectors = state.get('dream_vectors', [])
            self.dream_vectors = [np.array(v) for v in vectors]

            print(f"✓ Loaded dream state: {len(self.dream_vectors)} dreams")

        except FileNotFoundError:
            print("⚠️  No saved dream state, starting fresh")

    def _save_state(self):
        """Save dream history to JSON."""
        state = {
            'observations': self.observations,
            'dream_vectors': [v.tolist() for v in self.dream_vectors[-50:]]  # Keep last 50
        }

        with open(self.state_path, 'w') as f:
            json.dump(state, f, indent=2)


if __name__ == '__main__':
    dreamer = DreamSonnet()

    print("Testing Dream Sonnet...\n")

    sonnet1 = """When winter winds do blow and summer's heat
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

    sonnet2 = """Shall I compare thee to a summer's day?
Thou art more lovely and more temperate.
Rough winds do shake the darling buds of May.
And summer's lease hath all too short a date.
Sometime too hot the eye of heaven shines.
And often is his gold complexion dimmed.
And every fair from fair sometime declines.
By chance or nature's changing course untrimmed.
But thy eternal summer shall not fade.
Nor lose possession of that fair thou owest.
Nor shall Death brag thou wanderest in his shade.
When in eternal lines to time thou growest.
So long as men can breathe or eyes can see.
So long lives this and this gives life to thee."""

    # Test 1: Semantic drift
    print("--- Test 1: Semantic Drift ---")
    drift_path = dreamer.dream_drift(sonnet1, sonnet2, steps=5)
    print(f"Generated {len(drift_path)} drift vectors")
    print(f"Start: {drift_path[0][:3]}...")
    print(f"End: {drift_path[-1][:3]}...\n")

    # Test 2: Centroid
    print("--- Test 2: Cloud Centroid ---")
    centroid = dreamer.dream_centroid([sonnet1, sonnet2])
    print(f"Centroid vector: {centroid}")
    print(f"Norm: {np.linalg.norm(centroid):.3f}\n")

    # Test 3: Random walk
    print("--- Test 3: Latent Walk ---")
    walk_path = dreamer.dream_walk(steps=5, step_size=0.3)
    print(f"Walk path: {len(walk_path)} steps")
    print(f"Start: {walk_path[0][:3]}...")
    print(f"End: {walk_path[-1][:3]}...\n")

    # Test 4: Expand dream
    print("--- Test 4: Expand Dream ---")
    components = dreamer.expand_dream(centroid)
    print(f"Quatrain 1: {components['quatrain1'][:3]}...")
    print(f"Quatrain 2: {components['quatrain2'][:3]}...")
    print(f"Quatrain 3: {components['quatrain3'][:3]}...")
    print(f"Couplet: {components['couplet'][:3]}...\n")

    # Test 5: Dream similarity
    print("--- Test 5: Dream Similarity ---")
    sim1 = dreamer.dream_similarity(centroid, sonnet1)
    sim2 = dreamer.dream_similarity(centroid, sonnet2)
    print(f"Centroid similarity to sonnet1: {sim1:.3f}")
    print(f"Centroid similarity to sonnet2: {sim2:.3f}\n")

    # Stats
    stats = dreamer.get_dream_stats()
    print(f"Dream stats: {stats}")

    print("\n✓ Dream sonnet system operational")
