"""Basic Imperial example: in/s PV, g-acc labels, mil displacement labels.

Run:
    python examples/basic_imperial.py
"""
import numpy as np
import matplotlib.pyplot as plt

from fcp_plotting import fcp


def main():
    f = np.logspace(0, 3, 400)  # 1 Hz .. 1 kHz
    # Create a toy PV curve in in/s using inches base
    # Let's mimic a peak around 30 Hz
    pv = (1.5 / (1 + (f/30)**2)) + 0.05*np.sqrt(f)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_xlim(1, 1000)
    ax.set_ylim(1e-2, 100)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Pseudo Velocity (in/s)")
    fcp(ax, v_unit="in/s")
    ax.grid(True, which="both", ls=":", lw=0.6, color="0.8")
    ax.loglog(f, pv, color="C1", label="PV (example)")
    ax.legend()
    fig.tight_layout()
    plt.savefig("examples/output_basic_imperial.png", dpi=160)
    print("Saved examples/output_basic_imperial.png")


if __name__ == "__main__":
    main()
