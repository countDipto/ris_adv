# Physically constrained malicious-RIS attack: implementation contract (v0.1)

## 1. Scope and falsifiable hypothesis

**Claim to test:** a second, adversary-controlled RIS can choose *valid reflection
phases* that cause greater DL channel-estimation error than a random-phase RIS
with the same location, size and propagation links. Later, measure whether the
additional estimation error produces an **excess oracle-relative rate penalty**.

This is a test-time evasion attack, not poisoning, transmitter-side PGD, or
an independently radiating jammer. The RIS control circuitry still uses power;
"passive" means there is no separately generated RF attack signal.

## 2. Legitimate system and attacker

- Victim: DSRnet on 16 fixed sensing elements of a 36 x 36 legitimate RIS.
  Input is four-by-four complex pilot observations encoded as amplitude/phase.
- Malicious RIS: A programmable reflecting elements near the legitimate
  RIS/user path. Its placement, antenna geometry, frequency, reflection loss,
  number of phase bits and channel strengths are **fixed for each comparison**.
- Attacker controls its reflection vector `phi`, **not** the 16 sensed values
  directly, the legitimate RIS, the base station, the pilots or the receiver.
- For the primary study, `phi` remains unchanged from uplink CSI acquisition
  through data transmission. This removes a separate channel-aging attack.
- White-box: victim network and current physical links are known (upper bound).
- Transfer: optimize against DnCNN and evaluate without victim gradients.
- Universal: optimize *one* phase vector on training user locations and deploy
  it unchanged on held-out users. Population/channel knowledge during training
  does **not** make it an entirely knowledge-free attack.

## 3. Channel model and constraints

For the legitimate RIS-to-user link, in a reciprocal narrowband TDD model:

```
h(phi) = h0 + reshape(G_ar @ (phi * g_ua), 36, 36)
```

Shapes: `h0=(B,36,36)`, `G_ar=(1296,A)`, `g_ua=(B,A)` and
`phi=(A,)` for a universal attack, or `(B,A)` for per-user optimization.
`G_ar` is attacker-RIS to legitimate-RIS coupling, and `g_ua` is user to
attacker-RIS coupling. The two hops carry physical path loss and phase. There
is **no post-hoc epsilon clipping, normalization or arbitrary gain multiplier**.

For b-bit phases, every coefficient must satisfy

```
phi_a = exp(j * 2*pi*k_a/(2**b)), k_a in {0,...,2**b-1}
```

For continuous phases, `|phi_a|=1`. A fixed reflection-efficiency factor, if
required by hardware, belongs **inside the calibrated link model** and must
not be optimized unless an amplitude-controllable RIS is explicitly studied.

**Critical missing data:** the available RISnet tensors give `h0` and the
legitimate BS-to-RIS link, but not `G_ar` or `g_ua`. Do not manufacture these by
resampling or adding noise to `h0`. They must be supplied through ray tracing,
explicit geometry plus a propagation model, or measured/calibrated channels.
Ensure all channels use the same amplitude convention and carrier frequency.

**Geometry now established for a first-order study:** the RISnet paper reports
3.5 GHz and a 36x36 legitimate RIS spaced at 0.25 wavelength; its authors'
`params.py` contains `ris_loc=(278.42, 576.97, 2)` meters. The latter is a code
parameter, **not independently verified as the exact array phase center**. See
the [RISnet paper](https://livrepository.liverpool.ac.uk/3190436/1/RISnet__A_Domain_Knowledge_Driven_Neural_Network_Architecture_for_RIS_Optimization_with_Mutual_Coupling_and_Partial_CSI.pdf)
and [original params.py](https://github.com/bilepeng/risnet_mutual_partial/blob/main/params.py).

`link_model.py` now builds a nominal malicious-RIS contribution using exact
spherical distances, free-space two-hop voltage gains, and a simplified
front-hemisphere cosine pattern. The malicious panel is initially 8x8 elements
spaced at half a wavelength, offset 4 m toward the median user direction and
8 m sideways from the legitimate RIS. Both panels are assumed vertical;
their normals are inferred from street geometry. **These are explicitly new
simulation assumptions**, not positions/orientations recovered from DeepMIMO.
The panel is only reflective when incident/outgoing rays lie in front of it.
No ray-traced obstruction, phase-origin calibration, or inter-element mutual
coupling is included. The generator never rescales links to force an attack.

`evaluate_attack.py` accepts a trusted PyTorch `.pt` file with:

```python
{
  "G_ar": complex_tensor,               # (1296,A), fixed attacker placement
  "g_ua_training": complex_tensor,      # (20480,A), RISnet training order
  "g_ua_testing": complex_tensor,       # (1024,A), RISnet testing order
  "metadata": {
    "provenance": "exact channel generator, ray-tracing scene or measurement ID",
    "carrier_frequency_hz": 3.5e9,      # RISnet paper frequency
    "units_aligned_with_h0": True,      # must be verified, not assumed
    "geometry": "attacker location, orientation, element spacing, array layout",
    "calibration": "how both hops were matched to RISnet's h0 convention"
  }
}
```

The CLI supports a fixed `G_ar` for the initial experiment. The attack core
also supports a batched `G_ar` where channel geometry varies with user.

It additionally accepts `.npz` files written by `generate_links.py` with the
same field names and JSON metadata. Those files deliberately set
`units_aligned_with_h0=False` and `phase_reference_verified=False` until the
DeepMIMO phase convention, coherent reference, RIS orientation and amplitude
scaling can be validated. The `--exploratory_unverified_links` flag allows
**software debugging only** with such a file; results are clearly marked as
non-publication-ready.

## 4. Optimization

Let `S` extract the 16 fixed sensing elements; `E` encode partial CSI; and
`f_theta` be the frozen estimator. For a user u, the loss is

```
L_u(phi) = || f_theta(E(S*h_u(phi) + n)) - h_u(phi) ||^2 / ||h_u(phi)||^2
```

The denominator uses the **true perturbed** channel. Optimization maximizes
the mean `L_u` over the training-user batch and over multiple common-random
pilot-noise draws (EOT). Adam optimizes phase angles; forward phases are
quantized to b bits using a straight-through gradient. Multiple random
restarts are used, and the best **actually quantized** candidate is retained.
The per-user optimization is a white-box upper bound; the universal one is the
primary deployable hypothesis. Optimized and control phases are held constant
during each test sequence.

This is an NMSE surrogate objective, not proof of a communication-layer
impact. In particular, it might favor physical channel cancellation. Rate and
the oracle gap below are needed to separate effects.

## 5. Evaluation and scientific controls

Compare *with identical attacker geometry and link amplitudes*:

1. **No malicious RIS:** `h0`.
2. **Random-phase malicious RIS:** `h(phi_random)` with same A and b.
3. **Optimized malicious RIS:** `h(phi_attack)`.

For each: report NMSE, complex correlation, and the actual physical channel
change. Separate the trained-area test users from the geographically held-out
test users. Run no-noise, 30, 20 and 10 dB pilot SNR. Use at least three
independent geometry/phase seeds for a paper, not just three noise draws.

Once a legitimate RIS phase controller `Theta*(.)` and BS-to-RIS link are
integrated, report the same-channel estimation penalty:

```
R_oracle(phi) = R(Theta*(h(phi)); h(phi))
R_DL(phi)     = R(Theta*(f_theta(E(S*h(phi)+n))); h(phi))
L_ML(phi)     = R_oracle(phi) - R_DL(phi)
Excess_ML    = L_ML(phi_attack) - mean_random[L_ML(phi_random)]
```

An oracle is a **diagnostic upper bound**, not a same-pilot-budget LS baseline.
A sparse interpolation/compressed-sensing estimator is a fairer deployable
baseline using the 16 sensing elements. A full LS/MMSE estimator must be
reported with its extra observation/pilot requirements.

## 6. What exists and what does not

- `attack.py`: channel composition, 1-8-bit and continuous phase controls,
  white-box/transfer-capable gradients, universal and per-user optimization.
- `evaluate_attack.py`: train-only universal optimization, untouched test
  evaluation, random/no-attack comparisons and structured CSI metrics.
- `test_attack.py`: synthetic *software unit tests* for dimensions and phase
  constraints. Synthetic tests are not research evidence.
- `link_model.py` and `generate_links.py`: nominal two-hop physics and a
  geometry-derived link-file generator (not yet calibrated to published h0).
- `test_link_model.py`: executable numeric tests of geometry and visibility.
- `sweep_geometry.py`: **TRAIN-only** malicious-panel placement, size and
  orientation screening using a deliberately optimistic, per-sensor upper bound.
  It derives the 16 sensing indices from the actual `data.partial_from_full`
  function and reports finalists on a disjoint subset of victim TRAIN users.
- `test_sweep_geometry.py`: NumPy-only verification that the bound exceeds a
  physically constructed one-bit reflected perturbation in a toy scene.
- **Not yet provided:** calibrated/same-raytracer attacker links, exact RIS
  orientation, rate/WSR controller, mobility, adaptive defenses, hardware data.

The immediate next milestone is screening physically feasible, visible sites
without inspecting held-out testing data. Then calibrate malicious-link phase
and amplitude against the same RISnet ray-tracing convention before treating
the three-condition CSI table as research evidence. Only then should we
integrate the rate controller and evaluate the excess ML-specific rate loss.

## 7. Train-only geometry sweep (recommended next run)

The initial 16x16 attacker 4.47 m away produced a ~-55.6 dB *random-phase*
change relative to the unperturbed full channel. The independent per-sensor
triangle-inequality upper bound at the 16 sensing locations was extremely weak:
reported percentiles were -300 dB, -35.13 dB, and -28.29 dB; no user exceeded
-20 dB. The -300 dB floor denotes zero visible path in the selected model.
These are user-run exploratory diagnostics, not verified physical measurements.

Rather than spending more GPU time optimizing the phases at the same location,
screen new *hypothetical* panel locations using only `split["train"]` from the
frozen victim checkpoint:

```bash
# On Pine, inside ~/ris_smart_security_2/ris_adv:
python -m unittest test_sweep_geometry.py
python sweep_geometry.py \
    --data_dir ../data \
    --checkpoint runs/dsrnet_snr.pt \
    --out runs/attack/geometry_sweep.json
```

The default sweep compares 8x8/16x16 panels, three forward and eight lateral
offsets, and three panel-facing directions (144 configurations). For each,
the bound assumes a separate ideal phase alignment for **every individual
legitimate sensing element and user**. A real universal one-bit RIS cannot
generally achieve this, so the screening is intentionally optimistic. We rank
by coverage (minimum 90% preferred), then the median sensing-channel bound.
The five best candidates are checked on different users from the same victim
TRAIN split. No victim-validation IDs, test channels, test locations, or far
users influence the choice. `data.partial_from_full` supplies the *actual*
sensing-element coordinates; no hand-guessed sensor grid is used for ranking.

The script writes the ranking JSON and prints the exact command to run
`generate_links.py` with both RIS normals frozen. The generated links will
still be marked unverified. If none reaches adequate visibility or upper-bound
strength, stop and reconsider the geometry/physical model instead of launching
150-step DSRnet phase optimization. The panels are **not** confirmed real
facades/buildings, and the upper bound is conditional on the unverified
free-space link amplitudes and RISnet scale convention.

## 8. Commands (run from the existing `ris_adv` directory)

Copy `attack.py`, `evaluate_attack.py`, `test_attack.py`, and this document,
plus `link_model.py`, `generate_links.py`, `test_link_model.py`,
`sweep_geometry.py`, and `test_sweep_geometry.py` beside the existing
`data.py`, `models.py`, `metrics.py` and `runs/` directory. The old direct
generation/evaluation commands below are retained only for reproduction of
earlier smoke tests; the sweep in section 7 should come first.

```bash
python -m unittest test_link_model.py
python -m unittest test_attack.py
python generate_links.py --data_dir ../data \
    --out ../data/malicious_links_geom_v0.npz \
    --reference_channels ../data/channels_ris_rx_testing.pt
python evaluate_attack.py --data_dir ../data \
    --links ../data/malicious_links_geom_v0.npz \
    --exploratory_unverified_links \
    --victim runs/dsrnet_snr.pt --attacker runs/dncnn_snr.pt \
    --phase_bits 1 --steps 150 --restarts 3 --seed 0
```

The generator reads the RISnet user locations on Pine and writes explicit
links. The evaluator's exploratory flag permits this **unverified initial
geometry** so software experiments can proceed without pretending the two
channel models are calibrated. Each run writes
`runs/attack/universal_seed<seed>.json`, with keys
like `results["optimized@20dB"]["far"]["nmse_db"]`. Repeat with independent
seeds and attacker geometries before reporting paper results. **Do not use
the exploratory numbers in a paper until the coherent reference and amplitude
scale have been validated.** Only load `.pt` files from trusted sources
because PyTorch checkpoints may contain pickled data.
