#!/usr/bin/env python3
"""
HuggingFace API Wrapper for Prose - Separate Module

Allows testing Prose without downloading 783 MB weights.
Uses HuggingFace Inference API instead of local llama.cpp.

USAGE:
    from hf_api import ProseHFAPI

    prose = ProseHFAPI(api_key="hf_...")
    response = prose.generate("user input")
"""

from typing import Optional
import requests


class ProseHFAPI:
    """
    Prose generator using HuggingFace Inference API.

    Alternative to local ProseGenerator for testing without weights.
    """

    # HuggingFace model endpoint
    HF_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"

    def __init__(
        self,
        api_key: Optional[str] = None,
        harmonix = None,
    ):
        """
        Initialize HF API wrapper.

        Args:
            api_key: HuggingFace API key (or set HF_API_KEY env var)
            harmonix: ProseHarmonix instance for field-based generation
        """
        import os

        self.api_key = api_key or os.getenv('HF_API_KEY')
        if not self.api_key:
            raise ValueError(
                "HuggingFace API key required. "
                "Pass api_key param or set HF_API_KEY environment variable."
            )

        self.headers = {
            "Authorization": f"Bearer {self.api_key}"
        }

        # Harmonix for organism mode
        if harmonix is None:
            from harmonix import ProseHarmonix
            self.harmonix = ProseHarmonix()
        else:
            self.harmonix = harmonix

        print("[ProseHFAPI] Initialized (using HuggingFace Inference API)")

    def _call_api(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.8,
    ) -> str:
        """
        Call HuggingFace Inference API.

        Args:
            prompt: Input prompt
            max_tokens: Max generation length
            temperature: Sampling temperature

        Returns:
            Generated text
        """
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "return_full_text": False,
            }
        }

        response = requests.post(
            self.API_URL,
            headers=self.headers,
            json=payload,
            timeout=60,
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"HuggingFace API error: {response.status_code} - {response.text}"
            )

        result = response.json()

        # Extract generated text
        if isinstance(result, list) and len(result) > 0:
            return result[0].get('generated_text', '')
        elif isinstance(result, dict):
            return result.get('generated_text', '')
        else:
            raise RuntimeError(f"Unexpected API response format: {result}")

    def generate(
        self,
        user_input: str,
        max_tokens: int = 500,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Generate prose from field state (ORGANISM MODE!).

        Same interface as ProseGenerator.generate().

        Args:
            user_input: User's message (wrinkles field)
            max_tokens: Max generation length
            temperature: Override temperature

        Returns:
            Generated prose text from field state
        """
        # 1. User input wrinkles the field
        self.harmonix.add_disturbance(user_input, source='user_hfapi')

        # 2. Compute dissonance
        dissonance, pulse = self.harmonix.compute_dissonance(user_input, "")

        # 3. Get field seed (NOT from user input!)
        field_seed = self.harmonix.get_field_seed()

        # 4. Adjust temperature
        if temperature is None:
            temperature = self.harmonix.adjust_temperature(dissonance, base_temp=0.8)

        # 5. Generate from field state via HF API
        prose_text = self._call_api(
            field_seed,  # ← FROM CLOUD, NOT user input!
            max_tokens=max_tokens,
            temperature=temperature,
        )

        # 6. Add generated prose to cloud
        self.harmonix.add_prose(
            prose_text,
            quality=0.5,
            dissonance=dissonance,
            temperature=temperature,
            added_by='prose_hfapi'
        )

        return prose_text

    def generate_cascade(
        self,
        user_prompt: str,
        haiku_output: str = None,
        sonnet_output: str = None,
        max_tokens: int = 500,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Cascade mode via HF API.

        Same interface as ProseGenerator.generate_cascade().
        """
        # 1. Add all cascade inputs as field disturbances
        if haiku_output:
            self.harmonix.add_disturbance(haiku_output, source='cascade_haiku_hfapi')
        if sonnet_output:
            self.harmonix.add_disturbance(sonnet_output, source='cascade_sonnet_hfapi')
        self.harmonix.add_disturbance(user_prompt, source='cascade_user_hfapi')

        # 2. Compute combined dissonance
        combined_text = f"{user_prompt}\n{haiku_output or ''}\n{sonnet_output or ''}"
        dissonance, pulse = self.harmonix.compute_dissonance(combined_text, "")

        # 3. Get field seed
        field_seed = self.harmonix.get_field_seed()

        # 4. Adjust temperature
        if temperature is None:
            temperature = self.harmonix.adjust_temperature(dissonance, base_temp=0.9)

        # 5. Generate from field state
        prose_text = self._call_api(
            field_seed,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        # 6. Add to cloud
        self.harmonix.add_prose(
            prose_text,
            quality=0.5,
            dissonance=dissonance,
            temperature=temperature,
            added_by='prose_cascade_hfapi'
        )

        return prose_text

    def __call__(self, user_input: str, **kwargs) -> str:
        """Shorthand for generate()."""
        return self.generate(user_input, **kwargs)


if __name__ == "__main__":
    import os

    # Test HF API wrapper
    api_key = os.getenv('HF_API_KEY')
    if not api_key:
        print("❌ Set HF_API_KEY environment variable to test")
        print("   export HF_API_KEY='hf_...'")
        exit(1)

    print("=== Testing ProseHFAPI ===\n")

    from harmonix import ProseHarmonix

    harmonix = ProseHarmonix(db_path='cloud/test_hfapi.db')
    prose = ProseHFAPI(api_key=api_key, harmonix=harmonix)

    # Test generation
    user_input = "tell me about resonance"
    print(f"User: {user_input}")
    print()

    response = prose.generate(user_input, max_tokens=100)
    print(f"Prose (via HF API):")
    print(response)
    print()

    # Stats
    stats = harmonix.get_stats()
    print(f"Cloud: {stats['total_prose']} prose")

    harmonix.close()
    print("\n✓ HF API wrapper works!")
