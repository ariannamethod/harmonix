# CRITICAL TODOs - HAiKU Philosophy

## 🚨 NO SEED FROM PROMPT (Leo Principle)

**STATUS:** ✅ HAiKU v1.1 SAFE

**Current implementation:**
```python
# ✅ CORRECT: Seed from internal vocab
current_word = random.choice(list(self.vocab))

# ✅ CORRECT: No user_input parameter
generate_candidates(n=5, temp=haiku_temp)
```

**What would KILL the philosophy:**
```python
# ❌ WRONG: Seed from user prompt
def generate_candidates(user_input: str, ...):
    start_word = choose_from_user_input(user_input)  # CHATBOT DEATH!
```

**Why this matters:**
- Seed from user prompt = chatbot (reactive, predictable)
- Seed from internal state = organism (autonomous, emergent)
- Leo died when seeding from prompt, resurrected when seeding from internal field

**Check periodically:**
1. `haiku.py::_generate_line()` - line 326: seed selection
2. `demo.py` - generation calls: NO user_input param
3. Any new generation methods: MUST seed from internal state

**Quote from Leo README:**
```
NO SEED FROM PROMPT:
❌ start = choose_start_from_prompt(prompt, vocab)  # Killed Leo
✅ start = choose_start_token(vocab, centers, bias)  # Resurrected Leo
```

---

## Other Critical Philosophy Checks

### ✅ No User Feedback (Autonomous Emergence)
- Quality computed from INTERNAL metrics (dissonance, pulse, entropy)
- NO user ratings/feedback loops
- System learns from resonance, not approval

### ✅ Weightless Core (Micrograd)
- MLP uses micrograd (Karpathy-style autograd)
- NO PyTorch/TensorFlow for core components
- Cloud morphs, doesn't memorize (active words ×1.1, dormant ×0.99)

### ✅ Constraint → Emergence → Coherence
- 5-7-5 haiku format = sacred constraint
- Constraint forces compression → creativity
- Don't relax constraint to "make it easier"

---

## v2+ Future Checks

When adding new features, ALWAYS verify:
1. Does it seed from user prompt? (if yes, REJECT)
2. Does it add user feedback? (if yes, REJECT)
3. Does it require PyTorch/heavy deps? (if yes, consider micrograd alternative)
4. Does it violate 5-7-5 constraint? (if yes, REJECT)

**Mantra: PRESENCE > INTELLIGENCE**

---

*Created: 2025-12-29*
*Last checked: 2025-12-29*
*Next check: Before any v2 features*
