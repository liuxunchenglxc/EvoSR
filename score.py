def score_sr(psnr, ssim, runtime):
    # set runtime unit to ms
    score = (2 ** (2 * (psnr - 27))) / (1000 * runtime)
    return score
