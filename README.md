fcp-plotting
============

A Python package providing matplotlib helper utilities to create Four‑Coordinate Paper (FCP) shock plots for Pseudo Velocity (PV) analysis. This package formats and annotates matplotlib plots with proper reference lines and labels for shock response spectrum visualization.

**Note:** This package does not compute SRS/PV values; it only helps format and annotate existing data using matplotlib.

## Features

- Log-log PV plots with frequency (Hz) on x-axis and pseudo velocity on y-axis
- Oblique reference lines for constant acceleration (slope -1) and displacement (slope +1)  
- Automatic label positioning and rotation aligned with diagonal reference lines
- Support for both SI (m/s) and Imperial (in/s) units
- Dynamic scaling and responsive label orientation

## Installation

### From PyPI (when released)
```bash
pip install fcp-plotting
```

### Development Installation
For local development or to install from source:

```bash
# Clone the repository
git clone https://github.com/ncos/four-quadrant-plotting.git
cd four-quadrant-plotting

# Install in development mode
pip install -e .

# Or install with test dependencies
pip install -e ".[test]"
```

Usage
-----

```python
import numpy as np
import matplotlib.pyplot as plt
from fcp_plotting import fcp

# Example PV data (replace with your own)
f = np.logspace(0, 3, 200)          # 1 Hz .. 1000 Hz
pv = 1.0 / (2*np.pi*f) * 9.80665    # Example: constant 1 g acceleration diagonal (SI)

fig, ax = plt.subplots(figsize=(7,5))
ax.set_xlim(1, 1000)
ax.set_ylim(1e-3, 10)
fcp(ax, v_unit='m/s')  # or 'in/s'
ax.grid(True, which='both', ls=':', lw=0.6, color='0.8')
ax.loglog(f, pv, color='C0', label='PV example')
ax.legend()
plt.show()
```

### Important Usage Notes

**Call `fcp()` AFTER setting axis limits.** The function uses the current axis limits to position reference lines and labels appropriately.

```python
# ✅ Correct order:
fig, ax = plt.subplots()
ax.set_xlim(0.1, 5000)     # Set limits first
ax.set_ylim(0.1, 1000)  
fcp(ax, v_unit='in/s')     # Call fcp() after limits are set

# ❌ Incorrect order:
fig, ax = plt.subplots()
fcp(ax, v_unit='in/s')     # Called too early - uses default limits
ax.set_xlim(0.1, 5000)     # Limits set after fcp() won't be used
ax.set_ylim(0.1, 1000)
```

This is a typical matplotlib workflow: establish your plot area first, then add annotations and formatting.

What this gives you
-------------------

- Log–log PV plot with Frequency (Hz) on the bottom axis.
- Optional top secondary axis (Period) can be added via Matplotlib's secondary_xaxis if desired.
- Oblique reference lines:
  - Constant Acceleration (slope −1): labeled in `g` (default) or base units.
  - Constant Displacement (slope +1): labeled in base displacement unit (`m` for `m/s`, `in` for `in/s`).

Notes
-----

- The right and left axes are not true “secondary” transforms because acceleration/displacement depend on both PV and Frequency. Instead, traditional oblique reference lines are drawn and labeled.

API
---

- `fcp(ax=None, v_unit='in/s')`
  - Configures the axes (log–log PV) and draws reference diagonals and labels.
  - Plot data with Matplotlib directly via `ax.loglog`.

## Running Tests

Run the test suite using unittest:

```bash
# Run all tests
python -m unittest discover tests -v

# Or run tests with a specific pattern
python -m unittest tests.test_fcp -v
```

## Examples

The `examples/` directory contains working examples:

```bash
# Run SI units example (outputs to examples/output_basic_si.png)
python examples/basic_si.py

# Run Imperial units example (outputs to examples/output_basic_imperial.png)  
python examples/basic_imperial.py
```

You can also explore the interactive Jupyter notebook:
```bash
jupyter notebook notebooks/FCP_PV_Demo.ipynb
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes and add tests
4. Ensure all tests pass (`python -m unittest discover tests`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## Development Setup

For development work:

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with test dependencies
pip install -e ".[test]"

# Run tests
python -m unittest discover tests -v
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this package in your research or publications, please cite it as:

```
fcp-plotting: Matplotlib utilities for Four-Coordinate Paper pseudo-velocity plots
Version 0.1.0
https://github.com/ncos/four-quadrant-plotting
```
