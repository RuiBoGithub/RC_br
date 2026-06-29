import importlib.util
import sys
import unittest
from pathlib import Path


HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import numpy as np
    import pandas as pd
    import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
ETHLIB_ROOT = REPO_ROOT / "ETHlib"
sys.path.insert(0, str(ETHLIB_ROOT))


class TestRuntimeRCShadowPCNN(unittest.TestCase):
    @unittest.skipUnless(HAS_TORCH, "PyTorch is not installed in this environment.")
    def test_original_model_fit_with_runtime_rc_branch(self):
        from _BR_ import RCCase
        from _zone_ import DebugZone as Zone
        import emission_system
        import supply_system
        import EPFLpiml.model as pcnn_model
        from EPFLpiml.module_shadow import PCNN as RuntimeRCPCNN

        pcnn_model.PCNN = RuntimeRCPCNN
        pcnn_model.check_initialization_physical_parameters = lambda **kwargs: None

        case = RCCase(
            year=2023,
            loc_json=REPO_ROOT / "_json/location_params.json",
            geo_json=REPO_ROOT / "_json/geo_params.json",
            default_json=REPO_ROOT / "_json/default_params.json",
            epw_path=ETHLIB_ROOT / "auxiliary/Zurich-Kloten_2013.epw",
            occupancy_profile_csv=REPO_ROOT / "_data/default_occ.csv",
            Zone=Zone,
            supply_system=supply_system,
            emission_system=emission_system,
        )

        index = pd.date_range(
            "2023-01-20 00:00:00",
            periods=96,
            freq="h",
        )
        hours = np.arange(len(index), dtype=float)
        data = pd.DataFrame(
            {
                "Measured heating": 0.02 + 0.004 * np.sin(hours / 8.0),
                "Internal gains": 3000.0 + 500.0 * np.sin(hours / 6.0),
                "Solar gains": np.maximum(0.0, 2000.0 * np.sin(hours / 24.0)),
                "Outside temperature": 2.0 + 3.0 * np.sin(hours / 24.0),
                "Calendar": np.sin(2.0 * np.pi * index.hour / 24.0),
                "Case": 1.0,
            },
            index=index,
        )

        data_params = {
            "temperature_column": ["Measured heating"],
            "power_column": ["Internal gains", "Solar gains"],
            "case_column": "Case",
            "out_column": "Outside temperature",
            "neigh_column": None,
            "inputs_D": ["Calendar"],
            "outside_walls": [0],
            "neighboring_rooms": [],
            "seed": 0,
        }

        def weighted_mse(input, target, weight):
            return torch.mean(weight * (input - target) ** 2)

        model_params = {
            "name": "runtime_rc_shadow_smoke",
            "seed": 0,
            "save": False,
            "batch_size": 8,
            "shuffle": False,
            "n_epochs": 1,
            "learning_rate": 1e-3,
            "decrease_learning_rate": False,
            "heating": True,
            "cooling": False,
            "loss": weighted_mse,
            "warm_start_length": 2,
            "minimum_sequence_length": 8,
            "maximum_sequence_length": 16,
            "overlapping_distance": 8,
            "validation_percentage": 0.2,
            "test_percentage": 0.2,
            "learn_initial_hidden_states": True,
            "feed_input_through_nn": True,
            "input_nn_hidden_sizes": [8],
            "lstm_hidden_size": 8,
            "lstm_num_layers": 1,
            "layer_norm": False,
            "output_nn_hidden_sizes": [8],
            "division_factor": 10.0,
            "device": "cpu",
            "verbose": 0,
            "eps": 1e-6,
            "initial_values_physical_parameters": {
                "a": [1.0],
                "b": [1.0],
                "c": [1.0],
                "d": [1.0],
            },
            "rc_case": case,
            "rc_initial_state": 20.0,
            "include_previous_target_in_D": True,
            "include_physics_in_D": False,
            "include_out_column_in_D": True,
            "correction_mode": "relative_to_physics",
            "max_correction_fraction": 0.5,
            "minimum_correction_fraction": 0.05,
            "clamp_physics_nonnegative": True,
            "enforce_nonnegative_output": True,
        }

        model = pcnn_model.Model(
            data=data,
            module="PCNN",
            model_params=model_params,
            data_params=data_params,
            load=False,
        )

        model.fit(
            n_epochs=1,
            number_sequences=1,
            output_best=False,
        )

        self.assertEqual(len(model.train_losses), 2)
        self.assertIsNotNone(model.model.last_rc_state)


if __name__ == "__main__":
    unittest.main()
