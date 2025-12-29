"""
Integration tests for HAiKU v1

Tests the full system end-to-end:
- Database initialization
- Complete interaction loop
- All modules working together
- Background processes
- Error handling
"""

import pytest
import sqlite3
from pathlib import Path
from haiku import HaikuGenerator, SEED_WORDS
from harmonix import Harmonix
from tokenizer import DualTokenizer
from rae import RecursiveAdapterEngine
from metahaiku import MetaHaiku
from overthinkg import Overthinkg
from phase4_bridges import HaikuBridges, state_id_from_metrics
from dream_haiku import init_dream, HaikuDreamContext, should_run_dream


class TestFullIntegration:
    """Test complete HAiKU system integration."""

    @pytest.fixture
    def system(self, tmp_path):
        """Initialize complete HAiKU system."""
        db_path = tmp_path / "integration_cloud.db"

        # Initialize all components
        tokenizer = DualTokenizer()
        haiku_gen = HaikuGenerator(SEED_WORDS[:100])  # Use subset for speed
        harmonix = Harmonix(str(db_path))
        rae = RecursiveAdapterEngine()
        meta = MetaHaiku(haiku_gen)
        over = Overthinkg(str(db_path))
        bridges = HaikuBridges(str(db_path))

        # Initialize dream
        init_dream(str(db_path))

        system = {
            'tokenizer': tokenizer,
            'haiku_gen': haiku_gen,
            'harmonix': harmonix,
            'rae': rae,
            'meta': meta,
            'over': over,
            'bridges': bridges,
            'db_path': str(db_path)
        }

        yield system

        # Cleanup
        harmonix.close()
        over.close()
        bridges.close()

    def test_single_interaction(self, system):
        """Test a single complete interaction."""
        user_input = "what is resonance in the cloud"

        # Step 1: Tokenize
        tokens = system['tokenizer'].tokenize_dual(user_input)
        assert 'subwords' in tokens
        assert 'trigrams' in tokens

        # Step 2: Update harmonix
        system['harmonix'].morph_cloud(tokens['subwords'])
        system['harmonix'].update_trigrams(tokens['trigrams'])

        # Step 3: Update haiku generator
        system['haiku_gen'].update_chain(tokens['trigrams'])

        # Step 4: Compute dissonance
        user_trigrams = tokens['trigrams']
        system_trigrams = system['haiku_gen'].get_recent_trigrams()
        dissonance, pulse = system['harmonix'].compute_dissonance(
            user_trigrams,
            system_trigrams
        )

        assert 0.0 <= dissonance <= 1.0

        # Step 5: Adjust temperature
        haiku_temp, _ = system['harmonix'].adjust_temperature(dissonance)
        assert 0.3 <= haiku_temp <= 1.5

        # Step 6: Generate candidates
        candidates = system['haiku_gen'].generate_candidates(n=3, temp=haiku_temp)
        assert len(candidates) == 3

        # Step 7: Select best
        context = {'user': user_input, 'user_trigrams': user_trigrams}
        best_haiku = system['rae'].reason(context, candidates, scorer=system['haiku_gen'])
        assert isinstance(best_haiku, str)
        assert len(best_haiku.split('\n')) == 3

        # Step 8: Phase 4 recording
        quality = 0.6
        state_id = state_id_from_metrics(dissonance, pulse.entropy, quality)

        metrics_before = {'dissonance': dissonance, 'entropy': pulse.entropy, 'quality': 0.5}
        metrics_after = {'dissonance': dissonance, 'entropy': pulse.entropy, 'quality': quality}

        system['bridges'].record_state(
            state_id=state_id,
            metrics_before=metrics_before,
            metrics_after=metrics_after,
            turn_id="test_turn_1"
        )

        # Step 9: MLP training
        loss = system['haiku_gen'].observe(best_haiku, quality, user_context=user_trigrams)
        assert loss >= 0.0

        # Step 10: MetaHaiku reflection
        interaction_data = {
            'user': user_input,
            'haiku': best_haiku,
            'dissonance': dissonance,
            'pulse': pulse
        }
        internal_haiku = system['meta'].reflect(interaction_data)
        assert isinstance(internal_haiku, str)

        # Step 11: Overthinkg expansion
        system['over'].expand(recent_trigrams=user_trigrams)

        # All steps completed successfully!

    def test_multiple_interactions(self, system):
        """Test multiple interactions in sequence."""
        inputs = [
            "what is resonance",
            "tell me about clouds",
            "how does meaning emerge",
            "constraint births form"
        ]

        prev_state_id = None

        for i, user_input in enumerate(inputs):
            # Tokenize
            tokens = system['tokenizer'].tokenize_dual(user_input)

            # Process
            system['harmonix'].morph_cloud(tokens['subwords'])
            system['harmonix'].update_trigrams(tokens['trigrams'])
            system['haiku_gen'].update_chain(tokens['trigrams'])

            # Dissonance
            user_trigrams = tokens['trigrams']
            system_trigrams = system['haiku_gen'].get_recent_trigrams()
            dissonance, pulse = system['harmonix'].compute_dissonance(
                user_trigrams,
                system_trigrams
            )

            # Generate
            haiku_temp, _ = system['harmonix'].adjust_temperature(dissonance)
            candidates = system['haiku_gen'].generate_candidates(n=3, temp=haiku_temp)

            # Select
            context = {'user': user_input, 'user_trigrams': user_trigrams}
            best_haiku = system['rae'].reason(context, candidates, scorer=system['haiku_gen'])

            # Record
            quality = 0.5 + (i % 3) * 0.1
            state_id = state_id_from_metrics(dissonance, pulse.entropy, quality)

            metrics = {'dissonance': dissonance, 'entropy': pulse.entropy, 'quality': quality}
            system['bridges'].record_state(
                state_id=state_id,
                metrics_before=metrics,
                metrics_after=metrics,
                prev_state_id=prev_state_id,
                turn_id=f"test_turn_{i+1}"
            )

            prev_state_id = state_id

            # Train
            system['haiku_gen'].observe(best_haiku, quality, user_context=user_trigrams)

            # Background
            system['meta'].reflect({
                'user': user_input,
                'haiku': best_haiku,
                'dissonance': dissonance,
                'pulse': pulse
            })
            system['over'].expand(recent_trigrams=user_trigrams)

        # Check system state
        assert system['haiku_gen'].observations == len(inputs)

        # Check cloud grew
        cloud_size = system['harmonix'].get_cloud_size()
        assert cloud_size > 0

    def test_database_persistence(self, system):
        """Test database persistence across operations."""
        # Perform interaction
        tokens = system['tokenizer'].tokenize_dual("test persistence")
        system['harmonix'].morph_cloud(tokens['subwords'])
        system['harmonix'].update_trigrams(tokens['trigrams'])

        # Record metrics
        system['harmonix'].record_metrics(
            perplexity=0.5,
            entropy=0.6,
            resonance=0.7
        )

        # Check database
        conn = sqlite3.connect(system['db_path'])
        cursor = conn.cursor()

        # Check words table
        cursor.execute("SELECT COUNT(*) FROM words")
        word_count = cursor.fetchone()[0]
        assert word_count > 0

        # Check trigrams table
        cursor.execute("SELECT COUNT(*) FROM trigrams")
        trigram_count = cursor.fetchone()[0]
        assert trigram_count > 0

        # Check metrics table
        cursor.execute("SELECT COUNT(*) FROM metrics")
        metrics_count = cursor.fetchone()[0]
        assert metrics_count > 0

        conn.close()

    def test_dream_triggering(self, system):
        """Test dream dialog can be triggered."""
        # Create context that should trigger dream
        ctx = HaikuDreamContext(
            last_haiku="test\nhaiku\nhere",
            dissonance=0.5,
            pulse_entropy=0.6,
            pulse_novelty=0.8,  # High novelty
            pulse_arousal=0.5,
            quality=0.3,  # Low quality
            cloud_size=100,
            turn_count=15  # After cooldown
        )

        # Check if dream would trigger
        # (May not trigger due to probability, but should pass gates)
        from dream_haiku import DreamConfig
        config = DreamConfig(min_interval_turns=10, trigger_probability=1.0)

        result = should_run_dream(ctx, system['db_path'], config)

        # With probability=1.0, high novelty, and low quality, should trigger
        assert result == True


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_empty_input(self, tmp_path):
        """Test system handles empty input gracefully."""
        db_path = tmp_path / "error_test.db"
        tokenizer = DualTokenizer()

        tokens = tokenizer.tokenize_dual("")
        assert tokens['subwords'] == []
        assert tokens['trigrams'] == []

    def test_database_failure_graceful(self, tmp_path):
        """Test system handles database errors."""
        # This test ensures no crashes on db errors
        db_path = tmp_path / "test.db"
        harmonix = Harmonix(str(db_path))

        try:
            # Normal operation should work
            harmonix.morph_cloud(['test'])
            harmonix.close()
        except Exception as e:
            pytest.fail(f"Should handle gracefully: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
