"""
Phase Transitions - Dynamic temperature and style shifts

Detects and triggers phase transitions in generation based on:
- Cloud density (vocab saturation)
- Dissonance patterns
- MetaProse internal state
- Recursive semantic drift

Similar to HAiKU's phase transitions but with smoother gradients.
"""

import json
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class Phase(Enum):
    """Generation phases."""
    CRYSTALLINE = 0     # Low temp (0.4-0.6), precise, convergent
    LIQUID = 1          # Mid temp (0.7-0.9), balanced exploration
    VAPOR = 2           # High temp (1.0-1.3), creative, divergent
    PLASMA = 3          # Very high (1.4+), experimental, chaotic


@dataclass
class PhaseState:
    """Current phase state."""
    phase: Phase
    temperature_prose: float
    temperature_haiku: float
    stability: float        # How stable is this phase (0-1)
    transition_prob: float  # Probability of transitioning (0-1)
    duration: int           # Steps in this phase


class PhaseTransitions:
    """
    Manages dynamic phase transitions for generation.

    Monitors:
    - Cloud vocabulary saturation
    - Dissonance trends (increasing vs decreasing)
    - Novelty decay rate
    - MetaProse reflection frequency

    Transitions:
    - CRYSTALLINE → LIQUID: When dissonance rises (need exploration)
    - LIQUID → VAPOR: When novelty plateaus (need creativity)
    - VAPOR → CRYSTALLINE: When quality drops (need precision)
    - Any → PLASMA: Rare, triggered by extreme dissonance
    """

    def __init__(self, state_path: str = 'state/phase_transitions.json'):
        self.state_path = state_path
        self.current_state = PhaseState(
            phase=Phase.LIQUID,
            temperature_prose=0.8,
            temperature_haiku=0.8,
            stability=0.7,
            transition_prob=0.0,
            duration=0
        )

        # History for trend analysis
        self.dissonance_history = []
        self.novelty_history = []
        self.quality_history = []

        self.observations = 0
        self._load_state()

    def update(self, metrics: Dict) -> PhaseState:
        """
        Update phase based on metrics.

        Args:
            metrics: {
                'dissonance': float,
                'novelty': float,
                'quality': float,
                'vocab_size': int,
                'prose_count': int,
                'reflection_triggered': bool
            }

        Returns:
            Updated PhaseState
        """
        # Record metrics
        self.dissonance_history.append(metrics.get('dissonance', 0.5))
        self.novelty_history.append(metrics.get('novelty', 0.5))
        self.quality_history.append(metrics.get('quality', 0.5))

        # Keep last 20 observations
        if len(self.dissonance_history) > 20:
            self.dissonance_history = self.dissonance_history[-20:]
            self.novelty_history = self.novelty_history[-20:]
            self.quality_history = self.quality_history[-20:]

        self.current_state.duration += 1
        self.observations += 1

        # Compute trends
        trends = self._compute_trends()

        # Check for transition
        should_transition, new_phase = self._check_transition(metrics, trends)

        if should_transition:
            self._transition_to(new_phase, metrics)
        else:
            # Decay transition probability
            self.current_state.transition_prob *= 0.9

        # Adjust temperatures within phase
        self._adjust_temperatures(metrics)

        # Save periodically
        if self.observations % 5 == 0:
            self._save_state()

        return self.current_state

    def _compute_trends(self) -> Dict:
        """Compute trend signals from history."""
        if len(self.dissonance_history) < 3:
            return {
                'dissonance_trend': 0.0,
                'novelty_trend': 0.0,
                'quality_trend': 0.0
            }

        # Linear regression slope (simplified)
        def trend(values):
            n = len(values)
            x = np.arange(n)
            slope = np.polyfit(x, values, 1)[0]
            return slope

        return {
            'dissonance_trend': trend(self.dissonance_history),
            'novelty_trend': trend(self.novelty_history),
            'quality_trend': trend(self.quality_history)
        }

    def _check_transition(self, metrics: Dict, trends: Dict) -> Tuple[bool, Phase]:
        """Check if phase transition should occur."""
        current = self.current_state.phase
        dissonance = metrics.get('dissonance', 0.5)
        novelty = metrics.get('novelty', 0.5)
        quality = metrics.get('quality', 0.5)

        # Minimum duration in phase
        if self.current_state.duration < 3:
            return False, current

        # CRYSTALLINE → LIQUID: Rising dissonance
        if current == Phase.CRYSTALLINE:
            if trends['dissonance_trend'] > 0.05 or dissonance > 0.7:
                self.current_state.transition_prob += 0.3
                if self.current_state.transition_prob > 0.7:
                    return True, Phase.LIQUID

        # LIQUID → VAPOR: Novelty plateau, need creativity
        if current == Phase.LIQUID:
            if abs(trends['novelty_trend']) < 0.01 and novelty < 0.3:
                self.current_state.transition_prob += 0.2
                if self.current_state.transition_prob > 0.6:
                    return True, Phase.VAPOR

        # LIQUID → CRYSTALLINE: High quality, low dissonance
        if current == Phase.LIQUID:
            if quality > 0.8 and dissonance < 0.4:
                self.current_state.transition_prob += 0.25
                if self.current_state.transition_prob > 0.7:
                    return True, Phase.CRYSTALLINE

        # VAPOR → LIQUID: Quality recovering
        if current == Phase.VAPOR:
            if trends['quality_trend'] > 0.05:
                self.current_state.transition_prob += 0.3
                if self.current_state.transition_prob > 0.6:
                    return True, Phase.LIQUID

        # VAPOR → CRYSTALLINE: Quality drop, need precision
        if current == Phase.VAPOR:
            if quality < 0.5 and trends['quality_trend'] < -0.05:
                self.current_state.transition_prob += 0.4
                if self.current_state.transition_prob > 0.7:
                    return True, Phase.CRYSTALLINE

        # Any → PLASMA: Extreme dissonance (rare)
        if dissonance > 0.95 and self.current_state.duration > 5:
            if np.random.random() < 0.15:  # 15% chance
                return True, Phase.PLASMA

        # PLASMA → LIQUID: Always return after 2 steps (experimental)
        if current == Phase.PLASMA and self.current_state.duration >= 2:
            return True, Phase.LIQUID

        return False, current

    def _transition_to(self, new_phase: Phase, metrics: Dict):
        """Execute phase transition."""
        print(f"🌊 Phase transition: {self.current_state.phase.name} → {new_phase.name}")

        # Set base temperatures for phase
        temps = {
            Phase.CRYSTALLINE: (0.5, 0.6),
            Phase.LIQUID: (0.8, 0.8),
            Phase.VAPOR: (1.1, 1.0),
            Phase.PLASMA: (1.5, 1.2)
        }

        temp_s, temp_h = temps[new_phase]

        self.current_state = PhaseState(
            phase=new_phase,
            temperature_prose=temp_s,
            temperature_haiku=temp_h,
            stability=0.5,  # Start unstable, will stabilize
            transition_prob=0.0,
            duration=0
        )

    def _adjust_temperatures(self, metrics: Dict):
        """Fine-tune temperatures within current phase."""
        dissonance = metrics.get('dissonance', 0.5)
        novelty = metrics.get('novelty', 0.5)

        # Small adjustments (±0.1)
        if self.current_state.phase == Phase.LIQUID:
            # Dynamic adjustment in liquid phase
            if dissonance > 0.7:
                self.current_state.temperature_prose = min(1.0, self.current_state.temperature_prose + 0.05)
            elif dissonance < 0.3:
                self.current_state.temperature_prose = max(0.6, self.current_state.temperature_prose - 0.05)

        # Increase stability over time
        self.current_state.stability = min(1.0, self.current_state.stability + 0.05)

    def get_temperatures(self) -> Tuple[float, float]:
        """Get current temperatures for (prose, haiku)."""
        return (
            self.current_state.temperature_prose,
            self.current_state.temperature_haiku
        )

    def get_phase_info(self) -> Dict:
        """Get detailed phase information."""
        return {
            'phase': self.current_state.phase.name,
            'temperature_prose': self.current_state.temperature_prose,
            'temperature_haiku': self.current_state.temperature_haiku,
            'stability': self.current_state.stability,
            'transition_prob': self.current_state.transition_prob,
            'duration': self.current_state.duration,
            'observations': self.observations
        }

    def _load_state(self):
        """Load phase state from JSON."""
        try:
            with open(self.state_path, 'r') as f:
                state = json.load(f)

            phase_name = state.get('phase', 'LIQUID')
            self.current_state = PhaseState(
                phase=Phase[phase_name],
                temperature_prose=state.get('temperature_prose', 0.8),
                temperature_haiku=state.get('temperature_haiku', 0.8),
                stability=state.get('stability', 0.7),
                transition_prob=state.get('transition_prob', 0.0),
                duration=state.get('duration', 0)
            )

            self.observations = state.get('observations', 0)
            self.dissonance_history = state.get('dissonance_history', [])
            self.novelty_history = state.get('novelty_history', [])
            self.quality_history = state.get('quality_history', [])

            print(f"✓ Loaded phase state: {self.current_state.phase.name}")

        except FileNotFoundError:
            print("⚠️  No saved phase state, starting fresh")

    def _save_state(self):
        """Save phase state to JSON."""
        state = {
            'phase': self.current_state.phase.name,
            'temperature_prose': self.current_state.temperature_prose,
            'temperature_haiku': self.current_state.temperature_haiku,
            'stability': self.current_state.stability,
            'transition_prob': self.current_state.transition_prob,
            'duration': self.current_state.duration,
            'observations': self.observations,
            'dissonance_history': self.dissonance_history,
            'novelty_history': self.novelty_history,
            'quality_history': self.quality_history
        }

        with open(self.state_path, 'w') as f:
            json.dump(state, f, indent=2)


if __name__ == '__main__':
    pt = PhaseTransitions()

    print("Testing Phase Transitions...\n")

    # Simulate metrics over time
    scenarios = [
        {'dissonance': 0.4, 'novelty': 0.8, 'quality': 0.85, 'desc': 'High quality, low dissonance'},
        {'dissonance': 0.5, 'novelty': 0.7, 'quality': 0.80, 'desc': 'Stable'},
        {'dissonance': 0.7, 'novelty': 0.5, 'quality': 0.70, 'desc': 'Rising dissonance'},
        {'dissonance': 0.8, 'novelty': 0.4, 'quality': 0.65, 'desc': 'High dissonance'},
        {'dissonance': 0.6, 'novelty': 0.3, 'quality': 0.60, 'desc': 'Novelty plateau'},
    ]

    for i, metrics in enumerate(scenarios):
        print(f"\n--- Step {i+1}: {metrics['desc']} ---")
        state = pt.update(metrics)

        print(f"Phase: {state.phase.name}")
        print(f"Temperatures: prose={state.temperature_prose:.2f}, haiku={state.temperature_haiku:.2f}")
        print(f"Stability: {state.stability:.2f}, Transition prob: {state.transition_prob:.2f}")
        print(f"Duration: {state.duration} steps")

    print("\n✓ Phase transition system operational")
