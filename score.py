def score_sr(psnr, ssim, runtime):
    score = (2 ** (2 * (psnr - 26 + ssim))) / (1000 * runtime)
    return score
