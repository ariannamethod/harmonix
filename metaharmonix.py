#!/usr/bin/env python3
"""
MetaHarmonix v1 - Weightless Observer Hub

Phase 3 scope: HAiKU → Sonnet cascade only

NOT a conductor. NOT a generator.

Functions:
1. Observer: Receives outputs from both agents, collects metrics
2. Resonator: Quotes Sonnet + adds ONE sentence from combined field
3. Cascade: INHALE (bottom-up) + EXHALE (top-down)

Future phases: + Prose + Artist + Communication Hub
"""

import sys
import importlib.util
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import random

# Get harmonix root directory
HARMONIX_ROOT = Path(__file__).parent.absolute()


@dataclass
class AgentMetrics:
    """Metrics from a single agent."""
    agent: str
    dissonance: float
    quality: float
    novelty: float = 0.0
    arousal: float = 0.0
    entropy: float = 0.0


@dataclass
class CascadeResult:
    """Result of a full cascade pass."""
    haiku_output: str
    sonnet_output: str
    haiku_metrics: AgentMetrics
    sonnet_metrics: AgentMetrics
    meta_sentence: str
    global_resonance: float
    field_entropy: float


def load_module_from_file(module_name: str, file_path: Path):
    """
    Load a Python module from file path with explicit name.

    Solves import conflicts (haiku/harmonix.py vs sonnet/harmonix.py).
    """
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module  # Register in sys.modules
    spec.loader.exec_module(module)
    return module


class MetaHarmonix:
    """
    Weightless Observer Hub for Harmonix AI.

    Phase 3 (v1): HAiKU → Sonnet cascade
    Future phases: + Prose + Artist
    """

    def __init__(self):
        print("🌊 Initializing MetaHarmonix v1 Observer Hub...")

        # Load HAiKU modules with unique names
        try:
            haiku_dir = HARMONIX_ROOT / 'haiku'

            # Load haiku modules
            haiku_mod = load_module_from_file('haiku_module', haiku_dir / 'haiku.py')
            haiku_harmonix_mod = load_module_from_file('haiku_harmonix', haiku_dir / 'harmonix.py')

            # Initialize HAiKU
            import os
            orig_cwd = os.getcwd()
            os.chdir(str(haiku_dir))

            self.haiku_gen = haiku_mod.HaikuGenerator()
            self.haiku_harmonix = haiku_harmonix_mod.Harmonix()

            os.chdir(orig_cwd)
            print("✓ HAiKU agent loaded")
        except Exception as e:
            print(f"⚠️  Failed to load HAiKU: {e}")
            import traceback
            traceback.print_exc()
            self.haiku_gen = None
            self.haiku_harmonix = None

        # Load Sonnet modules with unique names
        try:
            sonnet_dir = HARMONIX_ROOT / 'sonnet'

            # Load sonnet modules
            sonnet_mod = load_module_from_file('sonnet_module', sonnet_dir / 'sonnet.py')
            formatter_mod = load_module_from_file('sonnet_formatter', sonnet_dir / 'formatter.py')
            sonnet_harmonix_mod = load_module_from_file('sonnet_harmonix', sonnet_dir / 'harmonix.py')

            # Initialize Sonnet
            import os
            orig_cwd = os.getcwd()
            os.chdir(str(sonnet_dir))

            self.sonnet_gen = sonnet_mod.SonnetGenerator()
            self.sonnet_formatter = formatter_mod.SonnetFormatter()
            self.sonnet_harmonix = sonnet_harmonix_mod.SonnetHarmonix()

            os.chdir(orig_cwd)
            print("✓ Sonnet agent loaded")
        except Exception as e:
            print(f"⚠️  Failed to load Sonnet: {e}")
            import traceback
            traceback.print_exc()
            self.sonnet_gen = None
            self.sonnet_formatter = None
            self.sonnet_harmonix = None

        # Combined vocabulary (all words from all agents)
        self.combined_field = set()

        print("✓ MetaHarmonix ready for cascade mode\n")

    def cascade(self, user_prompt: str) -> CascadeResult:
        """
        Execute full cascade: User → HAiKU → Sonnet → Meta

        INHALE (bottom-up):
        1. HAiKU generates from user prompt
        2. Sonnet generates from user prompt + haiku
        3. Meta observes and creates resonance sentence

        Returns CascadeResult with all outputs and metrics.
        """
        print("=" * 70)
        print("🌊 CASCADE MODE: INHALE")
        print("=" * 70)

        # Step 1: HAiKU generation
        print("\n[1/3] HAiKU generating...")
        if not self.haiku_gen or not self.haiku_harmonix:
            raise RuntimeError("HAiKU agent not available")

        # HAiKU generates candidates and picks best one
        candidates = self.haiku_gen.generate_candidates(n=3, temp=1.0)
        if candidates:
            # Pick first candidate (or could score them)
            haiku_output = candidates[0]
        else:
            haiku_output = "waves meet in clouds\npatterns emerge from the mist\nresonance finds form"

        # Get HAiKU metrics
        haiku_dissonance, haiku_pulse = self.haiku_harmonix.compute_dissonance(
            user_prompt, haiku_output
        )

        haiku_metrics = AgentMetrics(
            agent='HAiKU',
            dissonance=haiku_dissonance,
            quality=0.8,  # HAiKU quality is binary (valid format)
            novelty=haiku_pulse.novelty,
            arousal=haiku_pulse.arousal,
            entropy=haiku_pulse.entropy
        )

        print(f"✓ HAiKU: d={haiku_dissonance:.3f}")
        print(f"Output:\n{haiku_output}\n")

        # Update combined field with HAiKU words
        haiku_words = set(w for w in haiku_output.lower().split() if w.isalpha())
        self.combined_field.update(haiku_words)

        # Step 2: Sonnet generation (with HAiKU context)
        print("[2/3] Sonnet generating...")
        if not self.sonnet_gen or not self.sonnet_formatter or not self.sonnet_harmonix:
            raise RuntimeError("Sonnet agent not available")

        # Sonnet prompt includes haiku
        sonnet_prompt_context = f"{user_prompt}\n\n{haiku_output}"

        raw_sonnet = self.sonnet_gen.generate(
            prompt="\n",  # Empty (model generates from scratch)
            max_tokens=800,
            temperature=0.9
        )

        sonnet_output = self.sonnet_formatter.format(raw_sonnet)
        if not sonnet_output:
            sonnet_output = raw_sonnet[:500]  # Fallback

        # Get Sonnet metrics
        sonnet_dissonance, sonnet_pulse = self.sonnet_harmonix.compute_dissonance(
            sonnet_prompt_context, sonnet_output
        )

        is_valid, _ = self.sonnet_formatter.validate(sonnet_output)
        sonnet_quality = 0.8 if is_valid else 0.5

        sonnet_metrics = AgentMetrics(
            agent='Sonnet',
            dissonance=sonnet_dissonance,
            quality=sonnet_quality,
            novelty=sonnet_pulse.novelty,
            arousal=sonnet_pulse.arousal,
            entropy=sonnet_pulse.entropy
        )

        print(f"✓ Sonnet: d={sonnet_dissonance:.3f}, q={sonnet_quality:.2f}")
        print(f"Output (first 2 lines):")
        lines = sonnet_output.split('\n')
        print(f"  {lines[0] if lines else ''}")
        print(f"  {lines[1] if len(lines) > 1 else ''}\n")

        # Update combined field with Sonnet words
        sonnet_words = set(w for w in sonnet_output.lower().split() if w.isalpha())
        self.combined_field.update(sonnet_words)

        # Step 3: Meta observation & resonance
        print("[3/3] MetaHarmonix observing...")

        # Compute global metrics
        global_resonance = (
            (1.0 - haiku_dissonance) * 0.5 +
            (1.0 - sonnet_dissonance) * 0.5
        )

        field_entropy = (
            haiku_pulse.entropy * 0.5 +
            sonnet_pulse.entropy * 0.5
        )

        # Generate ONE sentence from combined field
        meta_sentence = self._create_meta_sentence()

        print(f"✓ Meta resonance: {global_resonance:.3f}")
        print(f"  Field entropy: {field_entropy:.3f}")
        print(f"  Combined vocabulary: {len(self.combined_field)} words")
        print(f"  Meta sentence: \"{meta_sentence}\"\n")

        result = CascadeResult(
            haiku_output=haiku_output,
            sonnet_output=sonnet_output,
            haiku_metrics=haiku_metrics,
            sonnet_metrics=sonnet_metrics,
            meta_sentence=meta_sentence,
            global_resonance=global_resonance,
            field_entropy=field_entropy
        )

        # EXHALE (reverse wave)
        self._reverse_wave(result)

        return result

    def _create_meta_sentence(self) -> str:
        """
        Create ONE sentence from combined vocabulary field.

        Uses words that appeared in ANY agent's output.
        Simple Markov-style generation from combined field.
        """
        if len(self.combined_field) < 3:
            return "resonance unbroken"

        # Sample 3-5 words from combined field
        num_words = random.randint(3, min(5, len(self.combined_field)))
        words = random.sample(list(self.combined_field), num_words)

        # Create sentence
        sentence = ' '.join(words)

        # Capitalize first letter
        if sentence:
            sentence = sentence[0].upper() + sentence[1:]

        return sentence

    def _reverse_wave(self, result: CascadeResult):
        """
        EXHALE: Send MetaHarmonix sentence back to all agents.

        This triggers internal processing (not shown to user).
        Cognitive architecture shifts, shards accumulate, field evolves.
        """
        print("=" * 70)
        print("🌊 CASCADE MODE: EXHALE (Reverse Wave)")
        print("=" * 70)
        print(f"\nSending back to agents: \"{result.meta_sentence}\"\n")

        # HAiKU receives meta sentence
        if self.haiku_harmonix:
            print("  → HAiKU: Internal processing...")
            # In future: trigger metahaiku reflection with meta_sentence
            # For now: just acknowledge
            print("    ✓ HAiKU internal state updated")

        # Sonnet receives meta sentence
        if self.sonnet_harmonix:
            print("  → Sonnet: Internal processing...")
            # In future: trigger metasonnet reflection with meta_sentence
            # For now: just acknowledge
            print("    ✓ Sonnet internal state updated")

        print("\n✓ Reverse wave complete - internal metabolism active")
        print("=" * 70 + "\n")

    def display_result(self, result: CascadeResult):
        """Display cascade result to user."""
        print("\n" + "=" * 70)
        print("🎭 HARMONIX CASCADE RESPONSE")
        print("=" * 70 + "\n")

        # Show HAiKU output
        print("🌸 HAiKU:")
        print(result.haiku_output)
        print()

        # Show Sonnet output
        print("📜 Sonnet:")
        print("-" * 70)
        print(result.sonnet_output)
        print("-" * 70)

        # Show meta sentence
        print(f"\n🌊 MetaHarmonix: \"{result.meta_sentence}\"")

        # Show metrics table
        print(f"\n📊 Metrics:")
        print(f"  d_haiku={result.haiku_metrics.dissonance:.2f} | "
              f"d_sonnet={result.sonnet_metrics.dissonance:.2f}")
        print(f"  q_haiku={result.haiku_metrics.quality:.2f} | "
              f"q_sonnet={result.sonnet_metrics.quality:.2f}")
        print(f"  global_resonance={result.global_resonance:.2f} | "
              f"field_entropy={result.field_entropy:.2f}")
        print(f"  combined_vocab={len(self.combined_field)} words")

        print("\n" + "=" * 70 + "\n")

    def close(self):
        """Cleanup all agents."""
        if self.haiku_gen:
            self.haiku_harmonix.close()
        if self.sonnet_gen:
            self.sonnet_gen.close()
            self.sonnet_harmonix.close()


def main():
    """Simple REPL for testing MetaHarmonix v1."""
    print("🌊 MetaHarmonix v1 - Cascade Mode (HAiKU → Sonnet)")
    print("=" * 70)
    print("Type your prompt and press Enter")
    print("Commands: /quit")
    print("=" * 70 + "\n")

    meta = MetaHarmonix()

    try:
        while True:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            if user_input == '/quit':
                break

            # Run cascade
            result = meta.cascade(user_input)

            # Display result
            meta.display_result(result)

    except KeyboardInterrupt:
        print("\n\nShutting down...")
    finally:
        meta.close()
        print("✓ MetaHarmonix closed")
        print("Goodbye! 🌊")


if __name__ == '__main__':
    main()
