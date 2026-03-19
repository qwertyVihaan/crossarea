# Cross-area Information Flow in Multi-region Neuropixels Recordings
(my original readme was incoherent, I used GPT to rewrite to thisreadme)


# Cross-area Information Flow in Multi-region Neuropixels Recordings

Analyzing the Steinmetz et al. 2019 dataset to quantify how much activity in one brain
area can be linearly decoded from simultaneous recordings in another — and where that
breaks down.

---

## The question

If you record from area A and area B simultaneously, and they're correlated, how much
does knowing A actually tell you about B? Correlation and linear decodability turn out
to be almost completely decoupled in this data. That gap is the main finding.

---

## Dataset

Steinmetz et al. 2019, *Nature*. 39 sessions across 10 mice performing a visual
discrimination task. Mice were shown gratings of varying contrast on left and right
screens and turned a wheel to report which side had higher contrast. Neuropixels probes
recorded simultaneously from up to 42 brain regions per session.

Public data: https://figshare.com/articles/dataset/Steinmetz_et_al_2019/9598406

NMA subset (what this code uses, ~700 MB total across 3 files):
- https://osf.io/agvxh/download
- https://osf.io/uv3mw/download
- https://osf.io/ehmw2/download

Session 7 specifically: 1156 neurons across 13 areas — hippocampus (CA1, CA3, DG, SUB),
prefrontal cortex (ILA, PL), thalamus (LD, LP, PO), motor cortex (MOs), taenia tecta (TT),
and visual cortex (VISa, VISp). 250 trials, 10ms time bins.

---

## What the analysis does

**Cross-area pairwise correlation** — Pearson correlation between trial-averaged mean
firing rates across areas. The strongest coupling in session 7 is PL-SUB (r=0.773) and
MOs-PL (r=0.752). Thalamic nuclei (LD, LP, PO) are broadly correlated with prefrontal
and hippocampal areas.

**Lagged cross-correlation** — does area A at time t predict area B at t+lag? PL leads
MOs by ~50ms (prefrontal drives motor), and PL drives both PO and SUB with ~100ms lag.
That timing is consistent with known prefrontal→thalamic→hippocampal circuitry.

**Linear decoder** — Ridge regression with PCA preprocessing predicting area B's mean
firing rate from area A's full population activity vector, 5-fold cross-validation.
Best decoder in session 7 is CA3→LD at R²=0.299. Most area pairs are near zero or
negative despite having real pairwise correlations.

The correlation vs decoder gap is the result I care about. PL-SUB correlates at r=0.773
but the PL→SUB decoder is weak. The shared signal between those areas clearly exists but
doesn't sit in a subspace that pairwise linear regression can recover. This is the core
computational problem in multi-area inference — you need a method that finds shared
latent structure across all areas simultaneously, not pair by pair.

**Population trajectory PCA** — how low-dimensional is each area's activity during the
task? Top-3 PCs explain 18-27% of trial variance depending on area. CA3 is the most
structured (0.27), ILA the least (0.18). Low dimensionality means the area's activity
lives on a manifold — which makes it more tractable to model but also means single-area
recordings might miss the interesting dynamics happening off that manifold.

**Reward modulation** — 10 of 13 areas show significant firing rate differences between
rewarded and unrewarded trials (p<0.05). The three that don't — DG, LP, VISa — are
interesting by contrast. PO shows the largest absolute modulation. Visual cortex (VISp)
shows weak but significant modulation despite being primarily sensory, which is consistent
with top-down feedback from prefrontal on reward trials.

---

## Main finding

High inter-area correlation does not mean one area's activity is linearly decodable from
another. In this session, pairwise correlations range from 0.7-0.77 for the strongest
pairs, but cross-validated R² for linear decoders into those same areas is consistently
low. The shared variance between areas lives in a low-dimensional latent subspace that
isn't recoverable by regressing one area's population vector onto another's scalar mean.

This has a direct implication for the problem of inferring activity in unrecorded brain
areas from partial multi-session recordings — which is an active ML problem in systems
neuroscience. A pairwise linear decoder sets a lower bound on what's recoverable.
Anything that does substantially better has found structure the correlation analysis
missed.

---



## What I want to look at next

The linear decoder is a baseline — it sets a floor on what's recoverable with the
simplest possible model. The interesting next step is fitting a low-rank RNN to the
multi-area activity and testing whether the inferred latent trajectories generalize
across sessions. If they do, you've found something real about circuit structure rather
than just fitting noise.

Also the lag analysis here is pretty coarse (50ms resolution). A continuous cross-spectral
coherence analysis in the gamma band would give much better temporal resolution on the
PL→PO and PL→MOs timing, and would let you say something about the frequency-specific
nature of the coupling rather than just the overall lag.icture of which areas are actually
driving which, vs just being correlated because of shared input.
