from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from paper_modules.datasets.scr_npz import (
    SDRDSP_V2_PROTOCOL,
    ScrNpzDataset,
    TEST_SCR_VALUES,
    TRAIN_SCR_VALUES,
    validate_sdrdsp_v2_manifest,
)
from paper_modules.models.registry import build_model
from scripts.preprocess_sdrdsp import (
    PHASE_ONLY_TARGET_MODEL,
    RCS_ONLY_TARGET_MODEL,
    REALISTIC_TARGET_MODEL,
    build_scr_samples,
    choose_train_targets,
    paper_cell_to_zero_based,
)


class SdrdspV2PreprocessTests(unittest.TestCase):
    def test_paper_cell_uses_one_based_index(self) -> None:
        self.assertEqual(paper_cell_to_zero_based(2083), 2082)
        with self.assertRaisesRegex(ValueError, "必须从 1 开始"):
            paper_cell_to_zero_based(0)

    def test_train_targets_respect_minimum_gap(self) -> None:
        positions = choose_train_targets(
            np.random.default_rng(42),
            range_cells=256,
            count=5,
            reference_cells=20,
            min_gap=21,
        )
        self.assertEqual(len(positions), 5)
        self.assertGreaterEqual(int(np.diff(positions).min()), 21)

    def test_target_is_continuous_and_scr_uses_reference_power_sum(self) -> None:
        clutter = np.ones((8, 64), dtype=np.complex64)
        x, y, audit = build_scr_samples(
            clutter=clutter,
            target_positions=[32],
            target_speeds=[0.4],
            scr_db=0,
            pulses=4,
            reference_cells=10,
            prt=1.0 / 1600.0,
            wavelength=0.03,
            max_windows=None,
        )

        self.assertEqual(x.shape, (2, 2, 4, 64))
        self.assertTrue(np.all(y[:, 32] == 1))
        self.assertTrue(np.all(y.sum(axis=1) == 1))
        record = audit["target_records"][0]
        self.assertAlmostEqual(record["reference_power_sum"], 10.0, places=6)
        self.assertAlmostEqual(record["target_amplitude"], np.sqrt(10.0), places=6)
        self.assertLessEqual(audit["max_abs_scr_error_db"], 1e-10)

        echoes = x[:, 0] + 1j * x[:, 1]
        injected_target = np.concatenate([echoes[0, :, 32], echoes[1, :, 32]]) - 1.0
        pulse_indices = np.arange(8, dtype=np.float64)
        expected_phase = 4.0 * np.pi * 0.4 * (1.0 / 1600.0) * pulse_indices / 0.03
        expected = np.sqrt(10.0) * np.exp(1j * expected_phase)
        np.testing.assert_allclose(injected_target, expected, rtol=1e-5, atol=1e-5)

    def test_realistic_target_is_deterministic_and_changes_only_target_cells(self) -> None:
        rng = np.random.default_rng(9)
        clutter = (rng.normal(size=(12, 64)) + 1j * rng.normal(size=(12, 64))).astype(np.complex64)
        kwargs = dict(
            clutter=clutter,
            target_positions=[32],
            target_speeds=[0.4],
            scr_db=-12,
            pulses=4,
            reference_cells=10,
            prt=1.0 / 1600.0,
            wavelength=0.03,
            max_windows=None,
        )
        ideal_x, ideal_y, _ = build_scr_samples(**kwargs)
        real_x, real_y, audit = build_scr_samples(
            **kwargs,
            target_model=REALISTIC_TARGET_MODEL,
            target_rng=np.random.default_rng(123),
            phase_noise_std_deg=10.0,
        )
        repeated_x, _, _ = build_scr_samples(
            **kwargs,
            target_model=REALISTIC_TARGET_MODEL,
            target_rng=np.random.default_rng(123),
            phase_noise_std_deg=10.0,
        )

        np.testing.assert_array_equal(real_y, ideal_y)
        np.testing.assert_array_equal(real_x[:, :, :, ideal_y[0] == 0], ideal_x[:, :, :, ideal_y[0] == 0])
        self.assertFalse(np.array_equal(real_x[:, :, :, 32], ideal_x[:, :, :, 32]))
        np.testing.assert_array_equal(real_x, repeated_x)
        self.assertAlmostEqual(audit["target_records"][0]["rcs_mean_power_gain"], 1.0, places=12)

    def test_realistic_target_requires_explicit_rng_and_known_model(self) -> None:
        kwargs = dict(
            clutter=np.ones((8, 64), dtype=np.complex64),
            target_positions=[32],
            target_speeds=[0.4],
            scr_db=0,
            pulses=4,
            reference_cells=10,
            prt=1.0 / 1600.0,
            wavelength=0.03,
            max_windows=None,
        )
        with self.assertRaisesRegex(ValueError, "显式 target_rng"):
            build_scr_samples(**kwargs, target_model=REALISTIC_TARGET_MODEL)
        with self.assertRaisesRegex(ValueError, "未知 target_model"):
            build_scr_samples(**kwargs, target_model="unknown")

    def test_four_domains_reuse_paired_phase_and_rcs_draws(self) -> None:
        rng = np.random.default_rng(31)
        clutter = (rng.normal(size=(12, 64)) + 1j * rng.normal(size=(12, 64))).astype(np.complex64)
        kwargs = dict(
            clutter=clutter,
            target_positions=[32],
            target_speeds=[0.4],
            scr_db=-8,
            pulses=4,
            reference_cells=10,
            prt=1.0 / 1600.0,
            wavelength=0.03,
            max_windows=None,
        )
        ideal_x, ideal_y, _ = build_scr_samples(**kwargs)
        phase_x, phase_y, phase_audit = build_scr_samples(
            **kwargs,
            target_model=PHASE_ONLY_TARGET_MODEL,
            target_rng=np.random.default_rng(777),
        )
        rcs_x, rcs_y, rcs_audit = build_scr_samples(
            **kwargs,
            target_model=RCS_ONLY_TARGET_MODEL,
            target_rng=np.random.default_rng(777),
        )
        combined_x, combined_y, combined_audit = build_scr_samples(
            **kwargs,
            target_model=REALISTIC_TARGET_MODEL,
            target_rng=np.random.default_rng(777),
        )

        for labels in (phase_y, rcs_y, combined_y):
            np.testing.assert_array_equal(labels, ideal_y)
        non_target = ideal_y[0] == 0
        for values in (phase_x, rcs_x, combined_x):
            np.testing.assert_array_equal(values[:, :, :, non_target], ideal_x[:, :, :, non_target])

        clutter_target = clutter[:12, 32].reshape(3, 4)
        target_signals = []
        for values in (ideal_x, phase_x, rcs_x, combined_x):
            echoes = values[:, 0, :, 32] + 1j * values[:, 1, :, 32]
            target_signals.append(echoes - clutter_target)
        ideal_s, phase_s, rcs_s, combined_s = target_signals
        np.testing.assert_allclose(np.abs(phase_s), np.abs(ideal_s), rtol=2e-5, atol=2e-5)
        np.testing.assert_allclose(np.abs(rcs_s), np.abs(combined_s), rtol=2e-5, atol=2e-5)
        np.testing.assert_allclose(
            phase_s / np.abs(phase_s), combined_s / np.abs(combined_s), rtol=2e-5, atol=2e-5
        )
        np.testing.assert_allclose(rcs_s / np.abs(rcs_s), ideal_s / np.abs(ideal_s), rtol=2e-5, atol=2e-5)

        phase_draws = phase_audit["target_records"][0]["component_draws"]
        rcs_draws = rcs_audit["target_records"][0]["component_draws"]
        combined_draws = combined_audit["target_records"][0]["component_draws"]
        self.assertEqual(phase_draws["phase_bundle_sha256"], combined_draws["phase_bundle_sha256"])
        self.assertEqual(rcs_draws["rcs_gain_sha256"], combined_draws["rcs_gain_sha256"])


class SdrdspV2DatasetTests(unittest.TestCase):
    def test_protocol_dataset_keeps_raw_values_and_validates_manifest(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd(), prefix=".tmp_sdrdsp_v2_") as tmp:
            data_dir = Path(tmp)
            self._write_protocol_dataset(data_dir)
            manifest = validate_sdrdsp_v2_manifest(data_dir, expected_pulses=4, expected_range_cells=256)
            self.assertEqual(manifest["protocol"]["id"], SDRDSP_V2_PROTOCOL)

            dataset = ScrNpzDataset(
                data_dir,
                "train",
                normalization="none",
                protocol=SDRDSP_V2_PROTOCOL,
                expected_pulses=4,
                expected_range_cells=256,
            )
            self.assertIsNone(dataset.norm)
            self.assertEqual(float(dataset.real[0, 0, 0]), 7.5)

    def test_protocol_rejects_hidden_normalization(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd(), prefix=".tmp_sdrdsp_v2_") as tmp:
            data_dir = Path(tmp)
            self._write_protocol_dataset(data_dir)
            with self.assertRaisesRegex(ValueError, "必须使用 normalization=none"):
                ScrNpzDataset(
                    data_dir,
                    "train",
                    normalization="train_standardize_clip",
                    protocol=SDRDSP_V2_PROTOCOL,
                    expected_pulses=4,
                    expected_range_cells=256,
                )

    def test_protocol_rejects_manifest_drift(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd(), prefix=".tmp_sdrdsp_v2_") as tmp:
            data_dir = Path(tmp)
            self._write_protocol_dataset(data_dir)
            manifest_path = data_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["protocol"]["scr_reference_power"] = "mean_power"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "scr_reference_power"):
                validate_sdrdsp_v2_manifest(data_dir, expected_pulses=4, expected_range_cells=256)

    @staticmethod
    def _write_protocol_dataset(data_dir: Path) -> None:
        train_x = np.zeros((len(TRAIN_SCR_VALUES), 2, 4, 256), dtype=np.float32)
        train_x[0, 0, 0, 0] = 7.5
        train_y = np.zeros((len(TRAIN_SCR_VALUES), 256), dtype=np.int32)
        train_y[:, [10, 40, 70, 100, 130]] = 1
        np.savez_compressed(
            data_dir / "train.npz",
            X=train_x,
            y=train_y,
            scr=np.asarray(TRAIN_SCR_VALUES, dtype=np.int16),
        )

        for scr in TEST_SCR_VALUES:
            test_x = np.zeros((1, 2, 4, 256), dtype=np.float32)
            test_y = np.zeros((1, 256), dtype=np.int32)
            test_y[:, 128] = 1
            np.savez_compressed(
                data_dir / f"test_scr_{scr}.npz",
                X=test_x,
                y=test_y,
                scr=np.asarray([scr], dtype=np.int16),
            )

        test_positive = {
            str(scr): {"min": 1.0, "mean": 1.0, "max": 1.0}
            for scr in TEST_SCR_VALUES
        }
        manifest = {
            "dataset": "SDRDSP Fig. 9 local-crop reproduction",
            "protocol": {
                "id": SDRDSP_V2_PROTOCOL,
                "scope": "local_crop",
                "paper_experiment": "Fig. 9",
                "train_background_name": "20210106155330_01_staring.mat",
                "test_background_name": "20210106155432_01_staring.mat",
                "train_scr_db": TRAIN_SCR_VALUES,
                "test_scr_db": TEST_SCR_VALUES,
                "pulses": 4,
                "range_cells": 256,
                "reference_cells": 20,
                "scr_reference_power": "sum_of_full_profile_per_cell_mean_power",
                "target_injection_order": "full_profile_before_non_overlapping_segmentation",
                "train_targets_per_scr": 5,
                "min_target_gap": 21,
                "test_target_cell_one_based": 2083,
                "test_speed_mps": 0.4,
                "pulse_window": "non_overlapping_step_equals_P",
                "label_rule": "only_true_injected_range_cells_are_positive",
                "normalization": "none",
            },
            "crop": {
                "paper_target_cell_one_based": 2083,
                "paper_target_index_zero_based": 2082,
                "crop_start_zero_based": 1954,
                "crop_end_exclusive_zero_based": 2210,
                "local_target_index_zero_based": 128,
            },
            "audit": {
                "min_train_target_gap": 30,
                "max_abs_injected_scr_error_db": 0.0,
                "train_positive_bins_per_window": {"min": 5.0, "mean": 5.0, "max": 5.0},
                "test_positive_bins_per_window": test_positive,
            },
        }
        (data_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


class SdrdspV2ModelTests(unittest.TestCase):
    def test_default_baseline_dropout_is_unchanged(self) -> None:
        model = build_model(
            {
                "model": {
                    "name": "original_stgnn",
                    "pulses": 4,
                    "range_cells": 256,
                }
            }
        )
        self.assertEqual(model.backbone.sfe1.gat.dropout_val, 0.1)
        self.assertEqual(model.backbone.sfe2.gat.dropout_val, 0.1)

    def test_v2_disables_gat_dropout_without_changing_interface(self) -> None:
        model = build_model(
            {
                "model": {
                    "name": "original_stgnn",
                    "pulses": 4,
                    "range_cells": 256,
                    "gat_dropout": 0.0,
                }
            }
        )
        self.assertEqual(model.backbone.sfe1.gat.dropout_val, 0.0)
        self.assertEqual(model.backbone.sfe2.gat.dropout_val, 0.0)
        logits = model(torch.randn(2, 4, 256, dtype=torch.complex64))
        self.assertEqual(tuple(logits.shape), (2, 2, 256))


if __name__ == "__main__":
    unittest.main()
