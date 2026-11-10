import torch


def nmse_db(h_hat, h):
    """Per-dataset NMSE in dB: 10 log10 E[||h_hat-h||^2 / ||h||^2] (DSRnet Eq. 11)."""
    num = (h_hat - h).abs().pow(2).flatten(1).sum(1)
    den = h.abs().pow(2).flatten(1).sum(1)
    return 10 * torch.log10((num / den).mean())


def psnr_db(h_hat, h):
    mse = (h_hat - h).abs().pow(2).mean()
    return 10 * torch.log10(h.abs().max() ** 2 / mse)


def complex_corr(h_hat, h):
    a = (h - h.mean()).flatten()
    b = (h_hat - h_hat.mean()).flatten()
    return ((a * b.conj()).mean().abs() /
            torch.sqrt(a.abs().pow(2).mean() * b.abs().pow(2).mean()))
