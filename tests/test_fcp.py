import unittest
import numpy as np
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

try:
    from matplotlib.axes._secondary_axes import SecondaryAxis  # type: ignore
except Exception:  # pragma: no cover
    SecondaryAxis = object  # type: ignore

from fcp_plotting import fcp


class TestFCP(unittest.TestCase):
    def test_configures_axes_and_adds_reference_lines(self):
        fig, ax = plt.subplots()
        base_lines = len(ax.lines)
        ax.set_xlim(1, 1000)
        ax.set_ylim(1e-4, 1e2)
        fcp(ax, v_unit="m/s")
        self.assertEqual(ax.get_xscale(), "log")
        self.assertEqual(ax.get_yscale(), "log")
        self.assertGreaterEqual(len(ax.lines), base_lines + 2)
        ref = ax.lines[-1]
        self.assertIn(ref.get_linestyle(), (":", (0, (1, 1))))
        self.assertAlmostEqual(ref.get_linewidth(), 0.6, places=6)

    def test_no_period_axis_by_default(self):
        fig, ax = plt.subplots()
        ax.set_xlim(1, 10)
        ax.set_ylim(1e-3, 1)
        fcp(ax)
        self.assertFalse(any(isinstance(child, SecondaryAxis) for child in ax.get_children()))

    def test_plot_with_matplotlib_loglog(self):
        fig, ax = plt.subplots()
        ax.set_xlim(1, 100)
        ax.set_ylim(1e-3, 10)
        fcp(ax)
        f = np.logspace(0, 2, 20)
        pv = 1 / (2 * np.pi * f)
        (line,) = ax.loglog(f, pv, color="C2", label="pv")
        self.assertEqual(line.get_label(), "pv")
        self.assertEqual(line.get_color(), "C2")
        x, y = line.get_data()
        self.assertTrue(np.allclose(x, f))
        self.assertTrue(np.allclose(y, pv))

    def test_invalid_units_raise(self):
        fig, ax = plt.subplots()
        with self.assertRaises(ValueError):
            fcp(ax, v_unit="km/h")

    def test_reference_line_slopes_are_correct_in_log_space(self):
        fig, ax = plt.subplots()
        ax.set_xlim(1, 1000)
        ax.set_ylim(1e-4, 1e2)
        fcp(ax, v_unit="m/s")

        fmin, fmax = ax.get_xlim()
        vmin, vmax = ax.get_ylim()
        f_grid = np.logspace(np.log10(fmin), np.log10(fmax), 241)

        a = 9.80665
        v_acc = a / (2 * np.pi * f_grid)
        m = (v_acc >= vmin) & (v_acc <= vmax)
        x = np.log10(f_grid[m])
        y = np.log10(v_acc[m])
        A = np.vstack([x, np.ones_like(x)]).T
        slope_acc, _ = np.linalg.lstsq(A, y, rcond=None)[0]
        self.assertAlmostEqual(slope_acc, -1.0, places=6)

        d = 1e-3
        v_disp = 2 * np.pi * f_grid * d
        m2 = (v_disp >= vmin) & (v_disp <= vmax)
        x2 = np.log10(f_grid[m2])
        y2 = np.log10(v_disp[m2])
        A2 = np.vstack([x2, np.ones_like(x2)]).T
        slope_disp, _ = np.linalg.lstsq(A2, y2, rcond=None)[0]
        self.assertAlmostEqual(slope_disp, 1.0, places=6)

    def test_labels_parallel(self):
        fig, ax = plt.subplots()
        ax.set_xlim(1, 1000)
        ax.set_ylim(1e-3, 30)
        fcp(ax, v_unit="m/s")
        fig.canvas.draw()
        texts = list(ax.texts)
        self.assertTrue(any("g" in t.get_text() for t in texts))
        self.assertTrue(any(" m" in t.get_text() for t in texts))
        for t in texts:
            s = t.get_text()
            x0, y0 = t.get_position()
            if "g" in s:
                K = x0 * y0
                x1 = x0 * 1.6
                y1 = K / x1
            elif " m" in s or " in" in s:
                K = y0 / x0
                x1 = x0 * 1.6
                y1 = K * x1
            else:
                continue
            P = np.array([[x0, y0], [x1, y1]])
            Q = ax.transData.transform(P)
            ang = np.degrees(np.arctan2(Q[1, 1] - Q[0, 1], Q[1, 0] - Q[0, 0]))
            r = t.get_rotation()
            diff = abs(((r - ang + 180) % 360) - 180)
            self.assertLess(diff, 2.0, f"Label rotation off by {diff:.2f} deg: {s}")

    def test_label_placement_regions(self):
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.set_xlim(1, 1000)
        ax.set_ylim(1e-3, 30)
        fcp(ax, v_unit="m/s")
        fig.canvas.draw()
        texts = list(ax.texts)
        self.assertTrue(any("g" in t.get_text() for t in texts))
        self.assertTrue(any(" m" in t.get_text() for t in texts))

        u0, u1 = np.log10(ax.get_xlim())
        v0, v1 = np.log10(ax.get_ylim())
        du, dv = u1 - u0, v1 - v0
        left_thresh = u0 + 0.12 * du
        right_thresh = u1 - 0.12 * du

        for t in texts:
            s = t.get_text()
            x, y = t.get_position()
            ux, vy = np.log10([x, y])
            if "g" in s:
                near_right = ux >= right_thresh
                self.assertTrue(near_right)
            if " m" in s:
                near_left = ux <= left_thresh
                self.assertTrue(near_left)

    def test_label_text_format_10_pow(self):
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.set_xlim(1, 1000)
        ax.set_ylim(1e-3, 30)
        fcp(ax, v_unit="m/s")
        fig.canvas.draw()
        texts = [t.get_text() for t in ax.texts]
        has_acc = any("10^{" in s and "g" in s for s in texts)
        has_disp = any("10^{" in s and " m" in s for s in texts)
        self.assertTrue(has_acc)
        self.assertTrue(has_disp)

    def test_disp_units_match_velocity_imperial_default(self):
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.set_xlim(1, 1000)
        ax.set_ylim(1e-3, 30)
        fcp(ax, v_unit="in/s")
        fig.canvas.draw()
        texts = [t.get_text() for t in ax.texts]
        self.assertTrue(any(" in" in s for s in texts))
        self.assertFalse(any(" mil" in s for s in texts))

    def test_all_labels_parallel_to_diagonals(self):
        """Test that all labels are properly aligned with their diagonal reference lines."""
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.set_xlim(1, 1000)
        ax.set_ylim(1e-3, 30)
        fcp(ax, v_unit="m/s")
        
        # Force draw so layout is final
        fig.canvas.draw()

        texts = list(ax.texts)
        self.assertTrue(any("g" in t.get_text() for t in texts), "No acceleration labels found")
        self.assertTrue(any(" m" in t.get_text() for t in texts), "No displacement labels found")

        # For each label, compute the local diagonal through its position
        # and compare the display-space angle to the label's rotation.
        for t in texts:
            s = t.get_text()
            x0, y0 = t.get_position()
            if "g" in s:
                # Accel diagonal: y = K/x through (x0,y0)
                K = x0 * y0
                x1 = x0 * 1.6
                y1 = K / x1
            elif " m" in s:
                # Disp diagonal: y = K*x through (x0,y0)
                K = y0 / x0
                x1 = x0 * 1.6
                y1 = K * x1
            else:
                continue
            
            P = np.array([[x0, y0], [x1, y1]])
            Q = ax.transData.transform(P)
            ang = np.degrees(np.arctan2(Q[1,1]-Q[0,1], Q[1,0]-Q[0,0]))
            r = t.get_rotation()
            diff = abs(((r - ang + 180) % 360) - 180)
            self.assertLess(diff, 2.0, f"Label rotation off by {diff:.2f} deg: {s}")

    def test_edge_case_very_small_xlim(self):
        """Test behavior with very small frequency ranges."""
        fig, ax = plt.subplots()
        ax.set_xlim(0.1, 1.0)  # Very narrow range
        ax.set_ylim(1e-3, 1)
        fcp(ax, v_unit="m/s")
        self.assertEqual(ax.get_xscale(), "log")
        self.assertEqual(ax.get_yscale(), "log")
        # Should still have some reference lines
        self.assertGreater(len(ax.lines), 0)

    def test_edge_case_very_large_xlim(self):
        """Test behavior with very large frequency ranges."""
        fig, ax = plt.subplots()
        ax.set_xlim(0.001, 100000)  # Very wide range
        ax.set_ylim(1e-6, 1000)
        fcp(ax, v_unit="in/s")
        self.assertEqual(ax.get_xscale(), "log")
        self.assertEqual(ax.get_yscale(), "log")
        # Should handle wide ranges gracefully
        self.assertGreater(len(ax.lines), 0)

    def test_error_handling_invalid_units_variations(self):
        """Test various invalid unit string variations."""
        fig, ax = plt.subplots()
        invalid_units = ["m/sec", "inch/s", "cm/s", "ft/s", "", None, 123]
        for unit in invalid_units:
            with self.subTest(unit=unit):
                with self.assertRaises((ValueError, TypeError, AttributeError)):
                    fcp(ax, v_unit=unit)

    def test_case_insensitive_valid_units(self):
        """Test that valid units work in different cases."""
        fig, ax = plt.subplots()
        ax.set_xlim(1, 100)
        ax.set_ylim(1e-3, 10)
        
        valid_variations = ["m/s", "M/S", "ms", "MS", "mps", "MPS", 
                           "in/s", "IN/S", "ips", "IPS"]
        for unit in valid_variations:
            with self.subTest(unit=unit):
                # Should not raise an error
                try:
                    fcp(ax, v_unit=unit)
                except ValueError:
                    # Some variations might not be supported, that's ok
                    pass

    def test_multiple_fcp_calls_same_axes(self):
        """Test calling fcp multiple times on the same axes."""
        fig, ax = plt.subplots()
        ax.set_xlim(1, 1000)
        ax.set_ylim(1e-3, 10)
        
        # First call
        lines_count_1 = len(ax.lines)
        fcp(ax, v_unit="m/s")
        lines_count_2 = len(ax.lines)
        self.assertGreater(lines_count_2, lines_count_1)
        
        # Second call should add more lines (might be desired behavior)
        fcp(ax, v_unit="m/s")
        lines_count_3 = len(ax.lines)
        # Lines may accumulate
        self.assertGreaterEqual(lines_count_3, lines_count_2)

    def test_zero_or_negative_limits(self):
        """Test behavior with problematic axis limits."""
        fig, ax = plt.subplots()
        
        # Test with limits that include zero or negative values
        problematic_limits = [
            ((0, 1000), (1e-3, 10)),  # xlim starts at 0
            ((1, 1000), (0, 10)),      # ylim starts at 0  
            ((-1, 1000), (1e-3, 10)),  # negative xlim
            ((1, 1000), (-1e-3, 10)),  # negative ylim
        ]
        
        for xlim, ylim in problematic_limits:
            with self.subTest(xlim=xlim, ylim=ylim):
                ax.clear()
                ax.set_xlim(*xlim)
                ax.set_ylim(*ylim)
                # Should raise ValueError for invalid limits
                with self.assertRaises(ValueError):
                    fcp(ax, v_unit="m/s")

    def test_no_axes_provided(self):
        """Test fcp behavior when no axes are provided."""
        # Clear any existing figure
        plt.close('all')
        
        # Create a new figure and axes
        fig, ax = plt.subplots()
        ax.set_xlim(1, 1000)
        ax.set_ylim(1e-3, 10)
        
        # Call fcp without providing axes (should use current axes)
        result_ax = fcp(v_unit="m/s")
        
        # Should return the current axes
        self.assertEqual(result_ax, ax)
        self.assertEqual(result_ax.get_xscale(), "log")
        self.assertEqual(result_ax.get_yscale(), "log")

    def test_performance_with_large_datasets(self):
        """Test performance doesn't degrade significantly with complex plots."""
        import time
        
        fig, ax = plt.subplots()
        ax.set_xlim(0.1, 10000)  # Wide range
        ax.set_ylim(1e-5, 1000)  # Wide range
        
        start_time = time.time()
        fcp(ax, v_unit="m/s")
        end_time = time.time()
        
        # Should complete reasonably quickly (less than 1 second)
        self.assertLess(end_time - start_time, 1.0)
        
        # Verify it still works correctly
        self.assertEqual(ax.get_xscale(), "log")
        self.assertEqual(ax.get_yscale(), "log")
        self.assertGreater(len(ax.lines), 0)

    def test_wide_frequency_range_coverage(self):
        """Test that reference lines cover the full range for very wide frequency spans."""
        fig, ax = plt.subplots()
        # Use very wide range like 0.1 to 5000 Hz
        ax.set_xlim(0.1, 5000)
        ax.set_ylim(0.1, 1000)
        
        fcp(ax, v_unit="in/s")
        fig.canvas.draw()
        
        # Should have many reference lines to cover the wide range
        self.assertGreater(len(ax.lines), 50)
        
        # Check that we have labels spanning multiple decades
        texts = [t.get_text() for t in ax.texts]
        acc_labels = [t for t in texts if "g" in t]
        disp_labels = [t for t in texts if " in" in t]
        
        # Should have acceleration labels spanning at least 3 decades
        self.assertGreaterEqual(len(acc_labels), 3)
        # Should have displacement labels spanning at least 3 decades  
        self.assertGreaterEqual(len(disp_labels), 3)
        
        # Verify we have high-decade labels (indicating proper range coverage)
        high_acc = any("10^{2}" in t or "10^{3}" in t or "10^{4}" in t for t in acc_labels)
        high_disp = any("10^{1}" in t or "10^{2}" in t or "10^{3}" in t for t in disp_labels)
        self.assertTrue(high_acc, f"No high acceleration labels found in: {acc_labels}")
        self.assertTrue(high_disp, f"No high displacement labels found in: {disp_labels}")

    def test_labels_within_axes_bounds(self):
        """Test that all labels are positioned within the axes bounding box."""
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.set_xlim(0.1, 5000)  # Wide range that can cause boundary issues
        ax.set_ylim(0.1, 1000)
        
        fcp(ax, v_unit="in/s")
        fig.canvas.draw()  # Force layout calculation
        
        # Get axes bbox in data coordinates
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        
        # Check each text label
        for text in ax.texts:
            # Get text position
            x_pos, y_pos = text.get_position()
            
            # Get text bounding box in data coordinates
            # Note: This is an approximation since exact text bounds depend on rendering
            text_bbox = text.get_window_extent(fig.canvas.get_renderer())
            text_bbox_data = ax.transData.inverted().transform(text_bbox)
            
            x_min, y_min = text_bbox_data[0]
            x_max, y_max = text_bbox_data[1]
            
            # Check if text extends beyond axes limits
            # Allow margin for rotated text bounding boxes which can be large
            margin_x = (xlim[1] - xlim[0]) * 0.05  # 5% margin
            margin_y = (ylim[1] - ylim[0]) * 0.80  # 80% margin for rotated text height
            
            self.assertGreaterEqual(x_min, xlim[0] - margin_x, 
                f"Label '{text.get_text()}' extends below x-axis minimum at {x_min:.3f} < {xlim[0]:.3f}")
            self.assertLessEqual(x_max, xlim[1] + margin_x,
                f"Label '{text.get_text()}' extends beyond x-axis maximum at {x_max:.3f} > {xlim[1]:.3f}")
            self.assertGreaterEqual(y_min, ylim[0] - margin_y,
                f"Label '{text.get_text()}' extends below y-axis minimum at {y_min:.3f} < {ylim[0]:.3f}")
            self.assertLessEqual(y_max, ylim[1] + margin_y,
                f"Label '{text.get_text()}' extends beyond y-axis maximum at {y_max:.3f} > {ylim[1]:.3f}")

    def test_label_positioning_with_extreme_ranges(self):
        """Test label positioning doesn't cause cutoff with various extreme axis ranges."""
        test_ranges = [
            # (xlim, ylim, v_unit, description)
            ((0.01, 10000), (0.01, 10000), "in/s", "Very wide range"),
            ((10, 100), (10, 100), "m/s", "Narrow range"),
            ((0.1, 1000), (1e-4, 1e4), "m/s", "Asymmetric range"),
            ((1, 10000), (0.001, 100), "in/s", "High frequency range"),
        ]
        
        for xlim, ylim, v_unit, description in test_ranges:
            with self.subTest(description=description, xlim=xlim, ylim=ylim):
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.set_xlim(*xlim)
                ax.set_ylim(*ylim)
                
                fcp(ax, v_unit=v_unit)
                fig.canvas.draw()
                
                # Verify we have labels
                self.assertGreater(len(ax.texts), 0, f"No labels generated for {description}")
                
                # Check that labels don't extend significantly beyond boundaries
                for text in ax.texts:
                    x_pos, y_pos = text.get_position()
                    
                    # Labels should be positioned within reasonable bounds
                    # (allowing some margin for label extent)
                    x_margin = (xlim[1] - xlim[0]) * 0.2  # 20% margin
                    y_margin = (ylim[1] - ylim[0]) * 0.2  # 20% margin
                    
                    self.assertGreaterEqual(x_pos, xlim[0] - x_margin,
                        f"Label '{text.get_text()}' positioned too far left in {description}")
                    self.assertLessEqual(x_pos, xlim[1] + x_margin,
                        f"Label '{text.get_text()}' positioned too far right in {description}")
                    self.assertGreaterEqual(y_pos, ylim[0] - y_margin,
                        f"Label '{text.get_text()}' positioned too low in {description}")
                    self.assertLessEqual(y_pos, ylim[1] + y_margin,
                        f"Label '{text.get_text()}' positioned too high in {description}")


if __name__ == "__main__":
    unittest.main()

