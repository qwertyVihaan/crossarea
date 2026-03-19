# Cross-area Information Flow in Multi-region Neuropixels Recordings
(my original readme was incoherent, I used GPT to rewrite to thisreadme)


Using the Steinmetz et al. 2019 dataset to quantify how much activity in one brain area
can be linearly predicted from simultaneous recordings in another — and where that
prediction breaks down.

---

## Background

The Steinmetz 2019 paper recorded from up to 42 brain regions simultaneously in mice
performing a visual discrimination task. That dataset is one of the best available for
studying multi-area coordination, but most analyses treat each area independently or
look at pairwise correlations. The interesting question is whether you can actually
reconstruct what's happening in area B if you only have recordings from area A — which
is the problem that comes up constantly in real experiments where you can't record
everywhere at once.

I got interested in this trying to understand how much information flows between visual
cortex and hippocampus during behavior, and whether the lag structure of that flow is
consistent with what you'd expect from known anatomy (visual → thalamus → hippocampus
with a ~200ms delay).

---

## What the analysis does

**Cross-area correlation matrix** — pairwise Pearson correlation between area-averaged
trial activity. Visual cortex areas (VISp, VISl, VISrl) are tightly coupled as expected.
Cross-modal coupling (visual ↔ motor, visual ↔ hippocampus) is weaker but present.

**Lagged cross-correlation** — does area A at time t predict area B at time t+lag?
This gives a directional sense of information flow. VISp → CA1 shows a clear peak lag
consistent with the expected thalamo-hippocampal delay. VISp → MOs (motor cortex)
peaks near zero, consistent with sensorimotor integration happening in parallel.

**Linear decoder (5-fold CV)** — Ridge regression with PCA preprocessing to predict
area B's mean firing rate from area A's full population activity vector. This is the
simplest version of the "infer unrecorded area from recorded subset" problem. R² varies
substantially by pair — some areas are fairly predictable from others (R²~0.4), others
are not (R²~0). The areas that are hard to predict are the interesting ones: they
contain information not captured by any linear combination of the recorded areas.

**Population trajectory PCA** — how many dimensions does each area's activity actually
occupy? Low-dimensional areas (high variance in PC1) are easier to decode from others.
High-dimensional areas suggest more independent computation.

**Reward modulation** — do areas respond differently on rewarded vs unrewarded trials?
MOs (secondary motor) and CA1 show the strongest feedback modulation, consistent with
their known roles in decision/memory.

---

## The core finding and why it matters

The linear decoder breaks down systematically. Visual cortex areas predict each other
well (R²>0.3) but predicting hippocampus or motor cortex from visual cortex alone is
poor even though they're correlated. The correlation is real but not linearly decodable
from single-area population vectors — which means there's shared latent structure that
isn't captured by any individual area's activity.

This is exactly the challenge that multi-area inference methods need to solve. A pairwise
linear decoder can't find the shared subspace. You need something that can simultaneously
constrain the inferred activity across all areas — which is what more sophisticated ML
approaches (RNNs, latent variable models) are designed to do.

---

## Data

Steinmetz et al. 2019, *Nature*. 10 mice, 39 sessions, up to 42 brain regions per session.
Publicly available at: https://figshare.com/articles/dataset/Steinmetz_et_al_2019/9598406

The NMA (Neuromatch Academy) subset is a smaller download and works fine for this
analysis: https://osf.io/2pfyn/

---

## How to run

**Google Colab:**
```python
!pip install numpy matplotlib seaborn scipy scikit-learn
!python analysis.py
```

The script downloads the Steinmetz NMA subset automatically (~15 MB) and caches it.
Change `session = 11` in the load function to try different recording sessions.

**Locally:**
```bash
pip install numpy matplotlib seaborn scipy scikit-learn
python analysis.py
```

---

## Dependencies

```
numpy matplotlib seaborn scipy scikit-learn
```

No AllenSDK, no specialized neuroscience packages. Just standard scientific Python.

---

## What I want to look at next

The linear decoder gives an upper bound on what's recoverable with a simple model.
The interesting next step is fitting a low-rank RNN to the same data and seeing whether
the inferred latent trajectories actually generalize to held-out sessions — which would
be a cleaner test of whether you've found real shared structure vs just overfitting.

Also: the lagged correlation approach is pretty blunt. A Granger causality analysis or
a directed information measure would give a cleaner picture of which areas are actually
driving which, vs just being correlated because of shared input.
