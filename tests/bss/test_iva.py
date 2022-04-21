import numpy as np
from scipy.io import loadmat


def _set_seed(seed=42):
    np.random.seed(seed)


def _convolve_rir(reverb=0.160, intervals=[8, 8, 8, 8, 8, 8, 8], degree=[0]):
    rir_path = (
        "./tests/.data/Impulse_response_Acoustic_Lab_Bar-Ilan_University_"
        "(Reverberation_{reverb:.3f}s)_{intervals}_1m_{degree:03d}.mat"
    )

    rir_path = rir_path.format(
        reverb=reverb, intervals="-".join(intervals), degree=degree
    )

    print(rir_path)

    rir_mat = loadmat(rir_path)
    rir = rir_mat["impulse_response"]

    print(rir.shape)


if __name__ == "__main__":
    _convolve_rir()
