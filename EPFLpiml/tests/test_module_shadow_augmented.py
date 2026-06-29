import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[2]
ETHLIB_ROOT = REPO_ROOT / "ETHlib"
sys.path.insert(0, str(ETHLIB_ROOT))


@unittest.skipUnless(HAS_TORCH, "PyTorch is not installed in this environment.")
class TestAugmentedRuntimeRCPCNN(unittest.TestCase):
    def _kwargs(self, rc_cell):
        return {
            "device": "cpu",
            "temperature_column": [0],
            "power_column": [1, 2, 3, 4, 6],
            "case_column": [8],
            "out_column": 5,
            "inputs_D": [7],
            "X_columns": None,
            "learn_initial_hidden_states": True,
            "feed_input_through_nn": True,
            "input_nn_hidden_sizes": [4],
            "lstm_hidden_size": 4,
            "lstm_num_layers": 1,
            "layer_norm": False,
            "output_nn_hidden_sizes": [4],
            "baseline_hidden_sizes": [4],
            "discrepancy_hidden_sizes": [4],
            "division_factor": 10.0,
            "eps": 1e-6,
            "temperature_min": [0.0],
            "temperature_range": [1.0],
            "rc_cell": rc_cell,
            "rc_initial_state": 20.0,
            "rc_internal_gains_column": 1,
            "rc_solar_gains_column": 2,
            "rc_t_set_heating_column": 3,
            "rc_h_ve_adj_column": 4,
            "rc_t_out_column": 5,
            "d_extra_input_columns": [6],
            "include_previous_target_in_D": True,
            "include_runtime_E_in_D": True,
            "include_physical_inputs_in_D": True,
            "include_out_column_in_D": True,
            "use_baseline_branch": True,
            "use_discrepancy_branch": True,
            "discrepancy_fraction_of_target_range": 0.1,
            "baseline_scale": 0.01,
            "trainable_rc": False,
            "enforce_nonnegative_output": True,
        }

    def test_baseline_heat_updates_next_rc_state(self):
        from EPFLpiml.module_shadow_augmented import PCNN

        class FakeRC(nn.Module):
            def __init__(self):
                super().__init__()
                self.dtype = torch.float32
                self.register_buffer("floor_area", torch.tensor(100.0))
                self.register_buffer("base_h_ve_adj", torch.tensor([[25.0]]))

            def step(
                self,
                *,
                internal_gains,
                solar_gains,
                t_out,
                previous_state,
                t_set_heating=None,
                h_ve_adj=None,
            ):
                heating_energy = torch.zeros_like(previous_state)
                return SimpleNamespace(heating_energy=heating_energy)

            def _calc_temperatures(
                self,
                *,
                energy_demand,
                internal_gains,
                solar_gains,
                t_out,
                previous_state,
                h_ve_adj,
            ):
                return {
                    "t_m_next": previous_state + 1e-6 * energy_demand,
                }

        model = PCNN(self._kwargs(FakeRC()))
        x = torch.tensor(
            [
                [
                    0.2,
                    100.0,
                    0.0,
                    16.0,
                    25.0,
                    12.0,
                    0.0,
                    0.5,
                    0.9,
                ]
            ],
            dtype=torch.float32,
        )

        prediction, states = model(x, states=None, warm_start=False)
        self.assertEqual(prediction.shape, (1, 1))
        self.assertGreater(float(model.last_B), 0.0)
        self.assertGreater(float(model.last_q_prediction), 0.0)
        self.assertGreater(float(states[1]), 20.0)

        prediction.sum().backward()
        baseline_grad = any(
            parameter.grad is not None
            for parameter in model.baseline_head.parameters()
        )
        discrepancy_grad = any(
            parameter.grad is not None
            for parameter in model.discrepancy_head.parameters()
        )
        lstm_grad = any(
            parameter.grad is not None
            for parameter in model.lstm.parameters()
        )

        self.assertTrue(baseline_grad)
        self.assertTrue(discrepancy_grad)
        self.assertTrue(lstm_grad)


if __name__ == "__main__":
    unittest.main()
