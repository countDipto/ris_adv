# ris_adv — DL-based RIS channel estimation (victim models)

Stage 1 of the adversarial-RIS project: re-implemented DSRnet plus the DnCNN and SRCNN baselines, trained on the public RISnet DeepMIMO data (Peng et al., GLOBECOM 2023). This is the same data setup as DSRnet: 20,480 train and 1,024 test samples, a 36×36 RIS, 9 BS antennas, and 16 sensing elements at indices [4, 13, 22, 31].

## Setup
```
pip install -r requirements.txt        # torch, numpy, scipy
```
Data: the unzipped RISnet files in `C:\ris_smart_city_2\data` (already there).

## Train (on the V100, from `ris_adv/` with data in `../data`)
```
python train.py --data_dir ../data --model dsrnet --epochs 300 --pilot_snr_range 10 30 --name dsrnet_snr
python train.py --data_dir ../data --model dncnn  --epochs 300 --pilot_snr_range 10 30 --name dncnn_snr   # surrogate
python train.py --data_dir ../data --model dsrnet --epochs 300 --name dsrnet_clean                       # noiseless reference
```
Splits: train/val are a random split of the training file. far is the last 10% of the street's x-range and is never trained on. test is the released test file. The checkpoint is selected on val. Each run saves `runs/<name>.pt`, `runs/<name>_log.csv` and `runs/<name>_final.json` (test and far NMSE, noiseless and at 30/20/10 dB pilot SNR).

Options: `--input_repr ap|ri` (amplitude/phase input as in DSRnet Eq. 2, or real/imag); `--pilot_snr_db 20` (noisy sensed CSI); `--variant _5e-5|_iid` (the other RISnet channel sets).

Reference from the DSRnet paper: NMSE −13.77 dB, ρ 0.934. Our PSNR uses max|H|²/MSE, so it may not match the paper's PSNR exactly.

## Files
- `data.py`: loads the `.pt` files (stored as (N, 1, 1296), reshaped row-major to 36×36; the ordering was checked against phase smoothness), extracts the 4×4 sensed sub-grid, encodes features, and adds optional pilot noise. `encode_partial()` is differentiable, so later stages can perturb the sensed CSI and reuse the same input pipeline.
- `models.py`: DSRnet re-implementation (SR sub-network 4→12→36 plus 2 SAMs with dilated SFEB, LRBs and channel attention; 83,836 params against the paper's 80,916), DnCNN and SRCNN.
- `metrics.py`: NMSE (dB), PSNR and complex correlation ρ.
- `train.py`: Adam with lr 9e-4 and cosine schedule, MSE loss on normalised real/imag outputs.

## Results so far
First 300-epoch run (noiseless sensing, checkpoint chosen on test, earlier script): test NMSE −26.38 dB, ρ 0.9988. The same model under pilot noise it never saw in training: 30 dB → −24.2, 20 dB → −18.2, 10 dB → −9.7 dB.
Caveats:
- Every released test user sits 0.2 m from a training user on the same grid, so test measures interpolation within the street, not generalisation. The far split was added for that reason.
- The sensors are 9 elements apart, so the 4×4 samples alias the spatial frequencies. A learning-free plane-wave fit scores +2.6 dB NMSE (worse than outputting zero). The network gets its accuracy from a learned, site-specific prior.

## Status
- CPU smoke test (1,024 samples, train = test, 30 epochs): DSRnet reaches −11.1 dB NMSE, ρ 0.958. The pipeline works end to end; real numbers need the full training set on the GPU.
- Not yet included: the WSR/RISnet downstream stage and the malicious-RIS channel model and evaluation (next stage).
