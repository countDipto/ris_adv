# ris_adv — DL-based RIS channel estimation + adversarial-RIS attack

Adversarial security of DL channel estimation in RIS-aided networks. Stage 1 is the
victim estimators (re-implemented DSRnet plus DnCNN/SRCNN baselines) trained on the
public RISnet DeepMIMO data (Peng et al., GLOBECOM 2023): 20,480 train / 1,024 test,
36×36 RIS, 9 BS antennas, 16 sensing elements at indices [4, 13, 22, 31]. Stage 2 is the
**adversarial-RIS attack** and its matched controls (`attack_channel.py`, `attack.py`).

## Setup
```
pip install -r requirements.txt        # torch, numpy, scipy  (+ matplotlib for plots)
```
Data: the unzipped RISnet files in `C:\ris_smart_city_2\data` (already there).

## Train (on the V100, from `ris_adv/` with data in `../data`)
```
python train.py --data_dir ../data --model dsrnet --epochs 300 --pilot_snr_range 10 30 --name dsrnet_snr
python train.py --data_dir ../data --model dncnn  --epochs 300 --pilot_snr_range 10 30 --name dncnn_snr   # surrogate
python train.py --data_dir ../data --model dsrnet --epochs 300 --name dsrnet_clean                       # noiseless reference
```
Splits: train/val are a random split of the training file; `far` is the last 10% of the
street's x-range and is never trained on; `test` is the released test file. The checkpoint
is selected on val. Each run saves `runs/<name>.pt`, `_log.csv` and `_final.json`.

Reference from the DSRnet paper: NMSE −13.77 dB, ρ 0.934.

## Attack (stage 2)
A rogue, zero-power ("adversarial") reflecting surface adds a structured term to the 16
sensed samples; its phase configuration is optimised to push the victim off its learned,
site-specific prior. **The attacker controls only phases** (unit-modulus reflection); the
amplitude is fixed by a calibrated power budget, so this is a physically-realisable attack,
not an arbitrary ε-ball perturbation.

```
# white-box attack on DSRnet, swept over the power budget, trained-area vs far
python attack.py --data_dir ../data --victim runs/dsrnet_snr.pt \
    --alphas -20 -15 -10 -6 -3 0 --n_adv 8 --steps 200 --eval_snr 20 --out attack_runs/dsrnet_snr

# black-box transfer: optimise on the DnCNN surrogate, evaluate on DSRnet
python attack.py --data_dir ../data --victim runs/dsrnet_snr.pt --surrogate runs/dncnn_snr.pt \
    --transfer --alphas -10 -6 -3 0 --out attack_runs/dsrnet_transfer

python attack.py --smoke        # self-contained CPU check, no data/checkpoints needed
```
Outputs `<out>.json` (all methods × α × split) and `<out>.png` (NMSE vs budget, trained
and far panels). Key flags: `--per_user` (idealised per-user φ upper bound; default is one
universal static φ), `--n_adv` (surface side, M = n_adv²), `--adv_center x y z` and
`--far_field` (geometry), `--restarts`, `--eval_snr -1` (noiseless).

**Always reported together**, so the claim is about optimisation, not "a reflector appeared":
`no_attack` (δ=0) · `random_phase` (random φ at the **same** calibrated strength, mean and
worst-of-K) · `optimized` · and a learning-free **LS/plane-wave victim** under the same δ
(if the learned net is hurt more, the attack is exploiting the learned prior).

### Attack-module spec (for modifying the optimiser)
`attack_channel.AdversarialSurface(n_adv, spacing=0.5, center, incidence, near_field=True)`
— all geometry in wavelengths, ½λ spacing (carrier drops out):
- `calibrate(hp_clean, alpha_db)` → sets strength `c` so **E_φ|δ_s|² = α·P_clean** under
  random φ (α swept in dB). Returns `c`.
- `delta(phi)` : `phi (...,M) real` → `(...,4,4) complex`, differentiable in φ, aligned with
  `data.partial_from_full`. Model: `δ_s(φ) = c·Σ_k A[s,k]·exp(j(φ_k + κ_k + 2π·d[s,k]))`,
  κ = incidence steering, d = ‖p_k−q_s‖, A = 1/d (near-field) or 1.
- `random_phi(batch_shape)` : matched random-phase control at the same `c`.

`attack.optimize_phi(model, surface, hp_clean, h_clean, cfg, steps, lr, per_user, eval_snr,
restarts)` → `(best_phi, history)`. Objective: **maximise** per-sample NMSE of
`estimate(model, hp_clean + delta(φ) + fixed_noise)` vs `h_clean`; Adam on φ (unit-modulus
built in, so no projection). `cfg = {"scale", "input_repr"}` from the victim checkpoint.
Shapes: sensed CSI `(B,4,4)` complex, full channel `(B,36,36)` complex, φ `(M,)` universal
or `(B,M)` per-user. Pilot noise is frozen (fixed seed) so every method sees one sensing
realisation.

## Files
- `data.py` — loader, 4×4 partial-CSI extraction, differentiable `encode_partial`, pilot noise, splits.
- `models.py` — DSRnet (83,836 params), DnCNN, SRCNN.
- `metrics.py` — NMSE (dB), PSNR, complex ρ.
- `train.py` — Adam + cosine, val-selected checkpoint, far holdout, `--pilot_snr_range`, `_final.json`.
- `attack_channel.py` — adversarial-surface channel model δ(φ), power calibration, random-phase control.
- `attack.py` — phase optimiser (universal / per-user, white-box / transfer), controls, LS victim, eval harness, plots, `--smoke`.

## Results so far
- Victim `dsrnet_snr`: −25.7 dB NMSE trained-area, −7.1 dB far (≈19 dB generalisation gap = the
  site-specific prior the attack targets). Surrogate `dncnn_snr`: −8.7 dB far.
- Sensors 9 apart alias the spatial frequencies; a learning-free plane-wave fit scores +2.6 dB,
  so DSRnet's accuracy is a learned prior, not interpolation.
- Attack code validated end-to-end (`--smoke`, plus a trained synthetic victim): calibration exact,
  analytic gradient == finite-difference, and at α=−6 dB the optimised surface did ≈5 dB more NMSE
  damage than the same-power random surface. Real-victim numbers pending the pine run.

## Status / next
- Done: victims (stage 1); adversarial-RIS attack + controls + harness (stage 2, this commit).
- Next: run the α-sweep + transfer on the real victims (pine); the moving-target-sensing defense
  (mask-conditioned DSRnet + cross-mask consistency detector); WSR/RISnet downstream rate stage.
