"""Basic SI example: m/s PV, g-acc labels, mm displacement labels.

Run:
    python examples/basic_si.py
"""
import numpy as np
import matplotlib.pyplot as plt

from fcp_plotting import fcp


def main():
    # Example PV curve (replace with real data)
    f = np.logspace(0, 3, 400)  # 1 Hz .. 1 kHz
    pv = 0.5 / (2*np.pi*f) + 0.002 * (2*np.pi*f)  # toy shape

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_xlim(1, 1000)
    ax.set_ylim(1e-3, 20)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Pseudo Velocity (m/s)")
    fcp(ax, v_unit="m/s")
    # Grid via Matplotlib (optional)
    ax.grid(True, which="both", ls=":", lw=0.6, color="0.8")
    ax.loglog(f, pv, color="C0", label="PV (example)")
    ax.legend()
    fig.tight_layout()
    plt.savefig("examples/output_basic_si.png", dpi=160)
    print("Saved examples/output_basic_si.png")


if __name__ == "__main__":
    main()
