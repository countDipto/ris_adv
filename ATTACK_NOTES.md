# ATTACK_NOTES.md: your workspace for the attack module

Syed writes the attack. Claude does not write the attack's formulation or optimization code. Everything else (defense, rate evaluation, baselines, plots, paper text) plugs into whatever you put here, so this file fixes the interfaces and the evaluation rules and leaves the attack itself blank.

---

## 1. What the codebase gives you (already working)

**Victim and surrogate checkpoints** (`runs/`):
- `dsrnet_snr.pt`: the victim (trained with pilot SNR ~ U[10, 30] dB).
- `dncnn_snr.pt`: the surrogate for gray-box and black-box settings.
- Each checkpoint is a dict with `state_dict`, `model`, `scale`, `input_repr`, `train_snr`, `split` (train/val/far index lists), `epoch`, `args`.

**Load a model:**
```python
ck = torch.load("runs/dsrnet_snr.pt", map_location=device, weights_only=False)
net = models.build(ck["model"]).to(device); net.load_state_dict(ck["state_dict"]); net.eval()
scale = ck["scale"]
```

**Tensor conventions** (`data.py`):

| Quantity | Shape | Notes |
|---|---|---|
| Full channel `h` | `(B, 36, 36)` complex64 | RIS to user, row-major element order |
| Sensed values `hp` | `(B, 4, 4)` complex64 | `partial_from_full(h)`, sensors at `SENSOR_IDX = [4, 13, 22, 31]` |
| Model input `x` | `(B, 2, 4, 4)` float | `encode_partial(hp, scale, "ap")` gives `[abs/scale, angle/pi]`; differentiable |
| Model output `y` | `(B, 2, 36, 36)` float | real and imaginary parts divided by `scale` |
| Estimate | `(B, 36, 36)` complex | `ri_to_complex(y, scale)` |

**Helpers:** `load_split`, `load_pt`, `RISChannelDataset`, `add_pilot_noise`, `split_indices`, `complex_to_ri`; metrics `nmse_db`, `psnr_db`, `complex_corr`.

**Data files** (`../data`): `channels_ris_rx_{training,testing}.pt` `(N, 1, 1296)`, `channel_tx_ris.pt` `(1296, 9)`, `locations_{training,testing}.pt` `(N, 3)`. Users sit on a 0.2 m grid at z = 2 m.

---

## 2. Threat model (fill in)

- Attacker's goal:
- Attacker's capabilities and constraints (hardware, size, placement, power):
- Attacker's knowledge (model, data, channel, the legitimate system's sensing schedule):
- What the attacker is allowed to change, and what stays fixed:
- Why this is physically realizable (one paragraph):

## 3. Formulation (fill in)

- Where the attacker's contribution enters the system model (which tensor does it change, and how?):
- Objective:
- Constraints:
- Knowledge variants you want to report (write each as its own subsection):

## 4. Algorithm (fill in)

- Procedure:
- Hyperparameters and their defaults:
- Compute cost and runtime per sample or per placement:

## 5. Implementation plan (fill in)

- File name(s): suggest `attack.py` in `ris_adv/`
- Function signatures you want the evaluation harness to call:
```python
# Fill in. The harness will import and call this. Suggested shape, change as you like:
# def run_attack(victim, scale, h, ...) -> h_perturbed  # (B, 36, 36) complex
```
- Open questions for Claude (anything you need from the channel model, e.g. geometry or path-loss calibration, that is not the attack itself):

---

## 6. Evaluation rules (fixed so results are comparable)

Every attack result goes into the paper only with all of the following.

1. **Controls in every table:**
   - no attack;
   - a random-phase control of the same size and placement (the claim is how much the attack adds beyond "a new reflector appeared");
   - an LS or interpolation estimator under the same condition, to separate model failure from signal change.
2. **Splits reported separately:** (a) test users in the trained area, (b) the far block (`split["far"]` and the test users with x beyond the far cut). Never only the pooled test number.
3. **Pilot SNR:** noiseless, 30, 20 and 10 dB. Fix the noise seed (the evaluation code uses `torch.manual_seed(123)`).
4. **Error is measured against the true perturbed channel**, not the original one. This is what separates ML failure from a plain channel change.
5. **Metrics:** NMSE (dB), ρ, and, once the WSR stage exists, achievable rate.
6. **Sweeps to include:** attacker size, placement, phase resolution, and (for the universal variant) how well it transfers to unseen users.
7. **Seeds:** at least 3, with the mean and the spread reported.

## 7. Results format (so the harness can read it)

Write one JSON per run to `runs/attack/<name>.json`:
```json
{"name": "...", "victim": "dsrnet_snr", "config": {},
 "results": {"<condition>": {"trained_area": {"nmse_db": 0, "rho": 0},
                              "far": {"nmse_db": 0, "rho": 0}}}}
```
where `<condition>` includes `no_attack`, `random_control`, and your attack variants, each at the pilot SNRs above.

## 8. Notes and scratch (yours)

-

---

*Workflow reminder: commit and push before pulling. One earlier update was overwritten when files written into the PC clone were pulled over.*
