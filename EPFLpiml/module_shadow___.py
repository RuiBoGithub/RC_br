from __future__ import annotations

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

try:
    from torch_rc import TorchRC5R1C
except ImportError:
    from ETHlib.torch_rc import TorchRC5R1C


def build_ml_lstm(input_size: int, hidden_size: int, num_layers: int, kwargs: dict) -> nn.Module:
    return nn.LSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        batch_first=True,
    )


def build_ml_gru(input_size: int, hidden_size: int, num_layers: int, kwargs: dict) -> nn.Module:
    return nn.GRU(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        batch_first=True,
    )


def build_ml_rnn(input_size: int, hidden_size: int, num_layers: int, kwargs: dict) -> nn.Module:
    return nn.RNN(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        nonlinearity=kwargs.get("rnn_nonlinearity", "tanh"),
        batch_first=True,
    )


def build_ml_mlp(input_size: int, hidden_size: int, num_layers: int, kwargs: dict) -> nn.Module:
    if input_size == hidden_size:
        return nn.Identity()
    return nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
    )


def build_ml_pcnn(input_size: int, hidden_size: int, num_layers: int, kwargs: dict) -> nn.Module:
    return _PCNNFeatureBackend(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        accumulator_decay=float(kwargs.get("pcnn_accumulator_decay", 0.98)),
        accumulator_scale=float(kwargs.get("pcnn_accumulator_scale", 0.05)),
    )


def build_ml_transformer_placeholder(
    input_size: int,
    hidden_size: int,
    num_layers: int,
    kwargs: dict,
) -> nn.Module:
    factory = kwargs.get("neural_backend_factory")
    if factory is None:
        raise NotImplementedError(
            "ML injector received neural_backend='transformer'. Provide "
            "neural_backend_factory(input_size, hidden_size, kwargs) "
            "returning an nn.Module whose forward accepts "
            "(embedding, states=None) and returns "
            "(neural_sequence_or_state, next_states)."
        )
    module = factory(input_size, hidden_size, kwargs)
    if not isinstance(module, nn.Module):
        raise TypeError("neural_backend_factory must return an nn.Module.")
    return module


ML_BACKEND_BUILDERS = {
    "lstm": build_ml_lstm,
    "gru": build_ml_gru,
    "rnn": build_ml_rnn,
    "mlp": build_ml_mlp,
    "pcnn": build_ml_pcnn,
    "plugin": build_ml_transformer_placeholder,
    "transformer": build_ml_transformer_placeholder,
}


def inject_ml_backend(
    ml: str,
    *,
    input_size: int,
    hidden_size: int,
    num_layers: int,
    kwargs: dict,
) -> nn.Module:
    """Build the requested ML backend for the shadow neural state."""

    key = str(ml).lower()
    try:
        builder = ML_BACKEND_BUILDERS[key]
    except KeyError as exc:
        raise ValueError(
            "Unknown ML backend {!r}. Choose from {}.".format(
                ml,
                sorted(ML_BACKEND_BUILDERS),
            )
        ) from exc

    return builder(
        input_size,
        hidden_size,
        num_layers,
        kwargs,
    )


class PCNN(nn.Module):
    """
    Closed-loop runtime ETHlib-assisted PCNN with pluggable neural backends.

    The fixed RC controller first computes the ideal heating demand E.
    The neural branches then add a non-negative operational baseline B
    and a bounded signed discrepancy D. The final heat prediction is fed
    back into the RC state update, so the next timestep state reflects
    the model prediction rather than the ideal controller alone.
    """

    def __init__(self, kwargs: dict):
        super().__init__()

        self.kwargs = kwargs
        self.device = kwargs["device"]
        self.temperature_column = list(kwargs["temperature_column"])
        self.power_column = list(kwargs["power_column"])
        self.case_column = list(kwargs["case_column"])
        self.out_column = kwargs.get("out_column")
        self.inputs_D = list(kwargs["inputs_D"])
        self._x_columns = kwargs.get("X_columns")

        self.include_previous_target_in_D = bool(
            kwargs.get("include_previous_target_in_D", True)
        )
        self.include_runtime_E_in_D = bool(
            kwargs.get("include_runtime_E_in_D", True)
        )
        self.include_physical_inputs_in_D = bool(
            kwargs.get("include_physical_inputs_in_D", True)
        )
        self.include_out_column_in_D = bool(
            kwargs.get("include_out_column_in_D", True)
        )
        self.use_baseline_branch = bool(
            kwargs.get("use_baseline_branch", True)
        )
        self.use_discrepancy_branch = bool(
            kwargs.get("use_discrepancy_branch", True)
        )
        self.neural_backend = kwargs.get("neural_backend", "lstm")
        self.neural_backend_factory = kwargs.get("neural_backend_factory")
        self.strict_required_inputs = bool(
            kwargs.get("strict_required_inputs", True)
        )
        self.validate_neural_input_variation = bool(
            kwargs.get("validate_neural_input_variation", True)
        )
        self.trainable_rc = bool(kwargs.get("trainable_rc", False))
        self.enforce_nonnegative_output = bool(
            kwargs.get("enforce_nonnegative_output", True)
        )

        self.learn_initial_hidden_states = kwargs[
            "learn_initial_hidden_states"
        ]
        self.feed_input_through_nn = kwargs["feed_input_through_nn"]
        self.input_nn_hidden_sizes = list(kwargs["input_nn_hidden_sizes"])
        self.lstm_hidden_size = int(kwargs["lstm_hidden_size"])
        self.lstm_num_layers = int(kwargs["lstm_num_layers"])
        self.neural_hidden_size = int(
            kwargs.get("neural_hidden_size", self.lstm_hidden_size)
        )
        self.neural_num_layers = int(
            kwargs.get("neural_num_layers", self.lstm_num_layers)
        )
        self.layer_norm = bool(kwargs["layer_norm"])
        self.output_nn_hidden_sizes = list(kwargs["output_nn_hidden_sizes"])
        self.baseline_hidden_sizes = list(
            kwargs.get(
                "baseline_hidden_sizes",
                self.output_nn_hidden_sizes,
            )
        )
        self.discrepancy_hidden_sizes = list(
            kwargs.get(
                "discrepancy_hidden_sizes",
                self.output_nn_hidden_sizes,
            )
        )
        self.division_factor = float(
            np.asarray(
                kwargs.get("division_factor", 10.0),
                dtype=float,
            ).reshape(-1)[0]
        )
        self.eps = float(kwargs.get("eps", 1e-8))

        self.target_min = torch.as_tensor(
            kwargs["temperature_min"],
            dtype=torch.float32,
            device=self.device,
        ).reshape(1, -1)
        self.target_range = torch.as_tensor(
            kwargs["temperature_range"],
            dtype=torch.float32,
            device=self.device,
        ).reshape(1, -1)
        self.discrepancy_fraction_of_target_range = float(
            kwargs.get("discrepancy_fraction_of_target_range", 0.5)
        )

        baseline_scale = kwargs.get("baseline_scale")
        if baseline_scale is None:
            baseline_scale = (
                float(kwargs.get("baseline_fraction_of_target_range", 0.10))
                *
                self.target_range
            )
        self.baseline_scale = torch.as_tensor(
            baseline_scale,
            dtype=torch.float32,
            device=self.device,
        ).reshape(1, -1)

        rc_case = kwargs.get("rc_case")
        rc_cell = kwargs.get("rc_cell")
        if rc_cell is None:
            if rc_case is None:
                raise ValueError(
                    "module_shadow_augmented.PCNN requires rc_case or "
                    "rc_cell for runtime RC evaluation."
                )
            rc_dtype = kwargs.get("rc_dtype", torch.float32)
            if isinstance(rc_dtype, str):
                rc_dtype = getattr(torch, rc_dtype)
            rc_cell = TorchRC5R1C.from_case(
                rc_case,
                sampled_params=kwargs.get("rc_sampled_params"),
                dtype=rc_dtype,
            )

        self.rc_cell = rc_cell.to(self.device)
        for parameter in self.rc_cell.parameters():
            parameter.requires_grad_(self.trainable_rc)

        self.rc_initial_state = torch.as_tensor(
            kwargs.get("rc_initial_state", 20.0),
            dtype=torch.float32,
            device=self.device,
        ).reshape(1, 1)
        self.rc_output_scale = torch.as_tensor(
            kwargs.get(
                "rc_output_scale",
                1.0 / (1000.0 * float(self.rc_cell.floor_area)),
            ),
            dtype=torch.float32,
            device=self.device,
        )

        self.rc_internal_gains_column = self._resolve_column(
            kwargs.get(
                "rc_internal_gains_column",
                self.power_column[0] if len(self.power_column) > 0 else None,
            )
        )
        self.rc_solar_gains_column = self._resolve_column(
            kwargs.get(
                "rc_solar_gains_column",
                self.power_column[1] if len(self.power_column) > 1 else None,
            )
        )
        self.rc_t_set_heating_column = self._resolve_column(
            kwargs.get("rc_t_set_heating_column")
        )
        self.rc_h_ve_adj_column = self._resolve_column(
            kwargs.get("rc_h_ve_adj_column")
        )
        self.rc_t_out_column = self._resolve_column(
            kwargs.get("rc_t_out_column", self.out_column)
        )
        self.d_extra_input_columns = [
            self._resolve_column(column)
            for column in kwargs.get("d_extra_input_columns", [])
        ]
        self._validate_required_columns()

        self.last_E = None
        self.last_B = None
        self.last_D = None
        self.last_q_prediction = None
        self.last_rc_state = None
        self.last_output = None
        self.last_neural_inputs = None
        self.last_neural_input_std = None
        self.last_baseline_input_std = None

        self._build_model()

    def _validate_required_columns(self) -> None:
        if not self.strict_required_inputs:
            return

        required = {
            "rc_internal_gains_column": self.rc_internal_gains_column,
            "rc_solar_gains_column": self.rc_solar_gains_column,
            "rc_t_out_column": self.rc_t_out_column,
            "rc_t_set_heating_column": self.rc_t_set_heating_column,
            "rc_h_ve_adj_column": self.rc_h_ve_adj_column,
        }
        missing = [
            name
            for name, column in required.items()
            if column is None
        ]
        if missing:
            raise ValueError(
                "Missing required augmented runtime RC input columns: "
                + ", ".join(missing)
            )

        if self.include_physical_inputs_in_D:
            physical_neural = [
                self.rc_internal_gains_column,
                self.rc_solar_gains_column,
                self.rc_t_set_heating_column,
                self.rc_h_ve_adj_column,
                self.rc_t_out_column,
            ]
            missing_neural = [
                str(column)
                for column in physical_neural
                if column is None
            ]
            if missing_neural:
                raise ValueError(
                    "Missing physical/controller inputs for the neural "
                    "branches."
                )


    def _build_model(self) -> None:
        d_indices: list[int] = []
        if self.include_previous_target_in_D:
            d_indices.extend(self.temperature_column)
        d_indices.extend(self.inputs_D)

        if self.include_physical_inputs_in_D:
            d_indices.extend(
                [
                    self.rc_internal_gains_column,
                    self.rc_solar_gains_column,
                    self.rc_t_set_heating_column,
                    self.rc_h_ve_adj_column,
                ]
            )

        if self.include_out_column_in_D and self.out_column is not None:
            d_indices.append(int(self.out_column))

        d_indices.extend(self.d_extra_input_columns)
        d_indices = [
            int(column)
            for column in d_indices
            if column is not None
        ]
        self.d_input_indices = list(dict.fromkeys(d_indices))
        if len(self.d_input_indices) == 0 and not self.include_runtime_E_in_D:
            raise ValueError("The neural branch has no input columns.")

        n_d_inputs = len(self.d_input_indices)
        if self.include_runtime_E_in_D:
            n_d_inputs += 1

        if self.feed_input_through_nn:
            sizes = [n_d_inputs] + self.input_nn_hidden_sizes
            self.input_nn = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(sizes[i], sizes[i + 1]),
                        nn.ReLU(),
                    )
                    for i in range(len(sizes) - 1)
                ]
            )
            neural_input_size = sizes[-1]
        else:
            self.input_nn = nn.ModuleList()
            neural_input_size = n_d_inputs

        self._build_neural_backend(neural_input_size)
        if self.layer_norm:
            self.norm = nn.LayerNorm(self.neural_hidden_size)

        self.baseline_head = (
            self._make_head(self.baseline_hidden_sizes)
            if self.use_baseline_branch
            else None
        )
        self.discrepancy_head = (
            self._make_head(self.discrepancy_hidden_sizes)
            if self.use_discrepancy_branch
            else None
        )

        if self.learn_initial_hidden_states:
            if self.neural_backend in {"lstm", "gru", "rnn"}:
                self.initial_h = nn.Parameter(
                    torch.zeros(
                        self.neural_num_layers,
                        self.neural_hidden_size,
                    )
                )
            if self.neural_backend == "lstm":
                self.initial_c = nn.Parameter(
                    torch.zeros(
                        self.neural_num_layers,
                        self.neural_hidden_size,
                    )
                )

        for name, parameter in self.named_parameters():
            if "bias" in name:
                nn.init.constant_(parameter, 0.0)
            elif "weight" in name and parameter.ndim >= 2:
                nn.init.xavier_normal_(parameter)

        if self.baseline_head is not None:
            self._initialise_near_zero_baseline()

    def _build_neural_backend(self, input_size: int) -> None:
        self.sequence_model = inject_ml_backend(
            self.neural_backend,
            input_size=input_size,
            hidden_size=self.neural_hidden_size,
            num_layers=self.neural_num_layers,
            kwargs=self.kwargs,
        )

    def _make_head(self, hidden_sizes: list[int]) -> nn.Sequential:
        layers: list[nn.Module] = []
        previous_size = self.neural_hidden_size
        for hidden_size in hidden_sizes:
            layers.extend(
                [
                    nn.Linear(previous_size, hidden_size),
                    nn.ReLU(),
                ]
            )
            previous_size = hidden_size
        layers.append(nn.Linear(previous_size, 1))
        return nn.Sequential(*layers)

    def _initialise_near_zero_baseline(self) -> None:
        output_layer = None
        for module in self.baseline_head.modules():
            if isinstance(module, nn.Linear):
                output_layer = module

        if output_layer is None:
            raise RuntimeError("baseline_head has no Linear output layer.")

        nn.init.constant_(output_layer.weight, 1e-4)
        nn.init.constant_(output_layer.bias, -6.0)

    def _resolve_column(self, column):
        if column is None or isinstance(column, int):
            return column
        if (
            isinstance(column, str)
            and self._x_columns is not None
            and column in self._x_columns
        ):
            return self._x_columns.index(column)
        return column

    def _initialise_states(self, batch_size: int):
        if self.neural_backend == "mlp":
            return None

        if self.neural_backend == "pcnn":
            return self.sequence_model.initial_state(
                batch_size=batch_size,
                device=self.device,
            )

        if self.learn_initial_hidden_states:
            h = (
                self.initial_h
                .unsqueeze(1)
                .repeat(1, batch_size, 1)
                .to(self.device)
            )
            if self.neural_backend == "lstm":
                c = (
                    self.initial_c
                    .unsqueeze(1)
                    .repeat(1, batch_size, 1)
                    .to(self.device)
                )
                return h, c
            return h
        else:
            h = torch.zeros(
                self.neural_num_layers,
                batch_size,
                self.neural_hidden_size,
                device=self.device,
            )
            if self.neural_backend == "lstm":
                c = torch.zeros(
                    self.neural_num_layers,
                    batch_size,
                    self.neural_hidden_size,
                    device=self.device,
                )
                return h, c
            return h

    def _neural_forward(self, embedding: torch.Tensor, neural_states):
        if self.neural_backend == "lstm":
            neural_sequence, next_states = self.sequence_model(
                embedding,
                neural_states,
            )
        elif self.neural_backend in {"gru", "rnn"}:
            neural_sequence, next_states = self.sequence_model(
                embedding,
                neural_states,
            )
        elif self.neural_backend == "mlp":
            neural_sequence = self.sequence_model(embedding)
            next_states = None
        elif self.neural_backend == "pcnn":
            neural_sequence, next_states = self.sequence_model(
                embedding,
                states=neural_states,
            )
        else:
            result = self.sequence_model(
                embedding,
                states=neural_states,
            )
            if not isinstance(result, tuple) or len(result) != 2:
                raise TypeError(
                    "Plugin neural backend must return "
                    "(neural_sequence_or_state, next_states)."
                )
            neural_sequence, next_states = result
            if neural_sequence.ndim == 2:
                neural_sequence = neural_sequence.unsqueeze(1)

        if self.layer_norm:
            neural_sequence = self.norm(neural_sequence)

        return neural_sequence[:, -1, :], next_states

    def _initialise_rc_state(self, batch_size: int) -> torch.Tensor:
        return self.rc_initial_state.repeat(batch_size, 1)

    def _normalise_target(self, values: torch.Tensor) -> torch.Tensor:
        return (
            0.1
            + 0.8
            * (values - self.target_min)
            / (self.target_range + self.eps)
        )

    def forward(self, x_: torch.Tensor, states=None, warm_start: bool = False):
        x = x_.clone()
        if x.ndim == 2:
            x = x.unsqueeze(1)

        batch_size = x.shape[0]
        if states is None:
            neural_states = self._initialise_states(batch_size)
            rc_state = self._initialise_rc_state(batch_size)
            self.last_output = x[:, -1, self.temperature_column].clone()
        else:
            if (
                isinstance(states, tuple)
                and len(states) == 2
            ):
                neural_states, rc_state = states
            else:
                neural_states = states
                rc_state = self._initialise_rc_state(batch_size)

        if (
            not warm_start
            and self.last_output is not None
            and self.include_previous_target_in_D
        ):
            x[:, -1, self.temperature_column] = self.last_output

        rc_inputs = self._rc_inputs(x)
        E = self._compute_physical_demand(
            rc_inputs=rc_inputs,
            rc_state=rc_state,
        )

        embedding = x[:, :, self.d_input_indices]
        if self.include_runtime_E_in_D:
            embedding = torch.cat(
                [
                    embedding,
                    E.reshape(batch_size, 1, 1).to(
                        dtype=embedding.dtype,
                        device=embedding.device,
                    ),
                ],
                dim=2,
            )
        self.last_neural_inputs = embedding.clone()
        self.last_neural_input_std = self._feature_std_sum(embedding)

        for layer in self.input_nn:
            embedding = layer(embedding)

        neural_state, neural_states_next = self._neural_forward(
            embedding,
            neural_states,
        )

        if self.baseline_head is None:
            B = torch.zeros_like(E)
        else:
            self.last_baseline_input_std = self._feature_std_sum(neural_state)
            if (
                self.training
                and self.validate_neural_input_variation
                and neural_state.shape[0] > 1
                and self.last_baseline_input_std <= self.eps
            ):
                raise ValueError(
                    "Baseline branch received non-varying neural inputs. "
                    "Check feature columns, shifting and normalisation."
                )
            B = F.softplus(self.baseline_head(neural_state)) * self.baseline_scale

        if self.discrepancy_head is None:
            D = torch.zeros_like(E)
        else:
            D = (
                self.discrepancy_fraction_of_target_range
                * self.target_range
                * torch.tanh(self.discrepancy_head(neural_state))
            )

        q_prediction = E + B + D
        if self.enforce_nonnegative_output:
            q_prediction = F.relu(q_prediction)

        rc_state_next = self._update_rc_state(
            q_prediction=q_prediction,
            rc_inputs=rc_inputs,
            rc_state=rc_state,
        )

        output = self._normalise_target(q_prediction)
        padded = torch.all(
            x[:, -1, self.case_column].abs() < self.eps,
            dim=1,
        )
        output[padded] = 0.0

        self.last_E = E.clone()
        self.last_B = B.clone()
        self.last_D = D.clone()
        self.last_q_prediction = q_prediction.clone()
        self.last_rc_state = rc_state_next.clone()
        self.last_output = output.clone()

        return output, (neural_states_next, rc_state_next)

    def _feature_std_sum(self, values: torch.Tensor) -> torch.Tensor:
        flattened = values.reshape(-1, values.shape[-1])
        if flattened.shape[0] <= 1:
            return torch.zeros((), dtype=values.dtype, device=values.device)
        return flattened.std(dim=0, unbiased=False).sum()

    def _rc_inputs(self, x: torch.Tensor) -> dict[str, torch.Tensor | None]:
        return {
            "internal_gains": self._column_or_zero(
                x,
                self.rc_internal_gains_column,
            ),
            "solar_gains": self._column_or_zero(
                x,
                self.rc_solar_gains_column,
            ),
            "t_out": self._column_or_zero(
                x,
                self.rc_t_out_column,
            ),
            "t_set_heating": self._optional_column(
                x,
                self.rc_t_set_heating_column,
            ),
            "h_ve_adj": self._optional_column(
                x,
                self.rc_h_ve_adj_column,
            ),
        }

    def _compute_physical_demand(
        self,
        *,
        rc_inputs: dict[str, torch.Tensor | None],
        rc_state: torch.Tensor,
    ) -> torch.Tensor:
        rc_result = self.rc_cell.step(
            internal_gains=rc_inputs["internal_gains"],
            solar_gains=rc_inputs["solar_gains"],
            t_out=rc_inputs["t_out"],
            previous_state=rc_state,
            t_set_heating=rc_inputs["t_set_heating"],
            h_ve_adj=rc_inputs["h_ve_adj"],
        )
        E = (
            rc_result.heating_energy
            .to(dtype=torch.float32, device=self.device)
            .reshape(-1, 1)
            * self.rc_output_scale
        )
        return torch.clamp(E, min=0.0)

    def _update_rc_state(
        self,
        *,
        q_prediction: torch.Tensor,
        rc_inputs: dict[str, torch.Tensor | None],
        rc_state: torch.Tensor,
    ) -> torch.Tensor:
        if not hasattr(self.rc_cell, "_calc_temperatures"):
            raise AttributeError(
                "Closed-loop RC update requires rc_cell._calc_temperatures."
            )

        h_ve_adj = rc_inputs["h_ve_adj"]
        if h_ve_adj is None:
            h_ve_adj = self.rc_cell.base_h_ve_adj

        heating_input = (
            q_prediction
            / self.rc_output_scale
        ).to(dtype=self.rc_cell.dtype)

        final = self.rc_cell._calc_temperatures(
            energy_demand=heating_input,
            internal_gains=rc_inputs["internal_gains"],
            solar_gains=rc_inputs["solar_gains"],
            t_out=rc_inputs["t_out"],
            previous_state=rc_state,
            h_ve_adj=h_ve_adj,
        )

        return (
            final["t_m_next"]
            .to(dtype=torch.float32, device=self.device)
            .reshape(-1, 1)
        )

    def _column_or_zero(
        self,
        x: torch.Tensor,
        column,
    ) -> torch.Tensor:
        if column is None:
            return torch.zeros(
                x.shape[0],
                1,
                dtype=torch.float32,
                device=self.device,
            )
        return x[:, -1, [int(column)]].to(dtype=torch.float32)

    def _optional_column(self, x: torch.Tensor, column):
        if column is None:
            return None
        return self._column_or_zero(x, column)

    @property
    def physics_weights(self):
        return np.asarray([], dtype=float)

    @property
    def E_parameters(self):
        return [
            list(np.asarray(self.physics_weights, dtype=float)),
            [],
            [],
            [],
        ]


class _PCNNFeatureBackend(nn.Module):
    """
    PCNN-style neural-state backend for the augmented shadow module.

    This is not the original EPFL final-output PCNN copied verbatim. It is
    adapted to the augmented runtime architecture: it returns a neural
    feature state for the B/D heads while the runtime RC branch remains the
    physical state update.
    """

    def __init__(
        self,
        *,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        accumulator_decay: float = 0.98,
        accumulator_scale: float = 0.05,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.accumulator_decay = accumulator_decay
        self.accumulator_scale = accumulator_scale

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.log_accumulator_weight = nn.Parameter(
            torch.empty(hidden_size, input_size)
        )
        self.accumulator_bias = nn.Parameter(
            torch.zeros(hidden_size)
        )
        self.mix = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_normal_(self.log_accumulator_weight)
        nn.init.constant_(self.accumulator_bias, 0.0)

    def initial_state(self, *, batch_size: int, device):
        h = torch.zeros(
            self.num_layers,
            batch_size,
            self.hidden_size,
            device=device,
        )
        c = torch.zeros(
            self.num_layers,
            batch_size,
            self.hidden_size,
            device=device,
        )
        accumulator = torch.zeros(
            batch_size,
            self.hidden_size,
            device=device,
        )
        return (h, c), accumulator

    def forward(self, embedding: torch.Tensor, states=None):
        batch_size = embedding.shape[0]
        if states is None:
            states = self.initial_state(
                batch_size=batch_size,
                device=embedding.device,
            )

        lstm_states, accumulator = states
        lstm_output, lstm_states_next = self.lstm(
            embedding,
            lstm_states,
        )

        positive_weight = self.log_accumulator_weight.exp()
        bounded_inputs = torch.tanh(embedding[:, -1, :])
        accumulator_input = F.linear(
            bounded_inputs,
            positive_weight,
            self.accumulator_bias,
        )
        accumulator_next = (
            self.accumulator_decay * accumulator
            + self.accumulator_scale * accumulator_input
        )

        mixed = self.mix(
            torch.cat(
                [
                    lstm_output[:, -1, :],
                    torch.tanh(accumulator_next),
                ],
                dim=1,
            )
        )

        return mixed.unsqueeze(1), (lstm_states_next, accumulator_next)
