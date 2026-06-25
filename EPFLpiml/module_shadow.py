from __future__ import annotations

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class PositiveLinear(nn.Module):
    """
    Linear layer with strictly positive weights.

    This is used to combine one or more externally calculated
    physics predictions, for example 5R1C and 3R2C outputs.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int = 1,
        initial_weight: float | list[float] | None = None,
        require_bias: bool = False,
    ):
        super().__init__()

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.require_bias = bool(require_bias)

        if initial_weight is None:
            initial = torch.full(
                (self.out_features, self.in_features),
                1.0 / max(self.in_features, 1),
                dtype=torch.float32,
            )
        else:
            initial = torch.as_tensor(
                initial_weight,
                dtype=torch.float32,
            )

            if initial.ndim == 0:
                initial = initial.repeat(
                    self.out_features,
                    self.in_features,
                )

            elif initial.ndim == 1:
                if len(initial) != self.in_features:
                    raise ValueError(
                        "initial_weight must have one value per "
                        "physics input column."
                    )
                initial = initial.reshape(1, -1).repeat(
                    self.out_features,
                    1,
                )

        initial = initial.clamp_min(1e-8)

        self.log_weight = nn.Parameter(
            torch.log(initial)
        )

        if self.require_bias:
            self.bias = nn.Parameter(
                torch.zeros(self.out_features)
            )
        else:
            self.register_parameter("bias", None)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.linear(
            inputs,
            self.log_weight.exp(),
            self.bias,
        )


class PCNN(nn.Module):
    """
    Editable ETHlib-assisted PCNN for heating-energy prediction.

    The module preserves the interface expected by the original
    EPFL Model class:

        prediction, states = model(x, states, warm_start)

    It has two jointly evaluated branches at every timestep:

    E: physical branch
        A positive combination of one or more externally calculated
        RC heating predictions, such as ETHlib 5R1C or 3R2C outputs.

    D: data-driven branch
        An LSTM correction based on an editable set of measured,
        weather, calendar and RC-state features.

    Final prediction:

        Q_piml = max(0, E + D)

    ETHlib itself remains external and is run before training. Its
    outputs are fixed inputs. The positive physical mixing weights and
    all neural-network parameters are trained together.
    """

    def __init__(self, kwargs: dict):
        super().__init__()

        self.device = kwargs["device"]

        # Original EPFL data-interface names retained for compatibility.
        self.temperature_column = list(
            kwargs["temperature_column"]
        )
        self.power_column = list(
            kwargs["power_column"]
        )
        self.case_column = list(
            kwargs["case_column"]
        )
        self.out_column = kwargs.get("out_column")
        self.inputs_D = list(kwargs["inputs_D"])

        # Editable architecture switches.
        self.include_previous_target_in_D = bool(
            kwargs.get(
                "include_previous_target_in_D",
                True,
            )
        )
        self.include_physics_in_D = bool(
            kwargs.get(
                "include_physics_in_D",
                True,
            )
        )
        self.include_out_column_in_D = bool(
            kwargs.get(
                "include_out_column_in_D",
                True,
            )
        )

        self.correction_mode = kwargs.get(
            "correction_mode",
            "relative_to_physics",
        )

        valid_modes = {
            "relative_to_physics",
            "target_range",
            "unbounded",
        }

        if self.correction_mode not in valid_modes:
            raise ValueError(
                f"Unknown correction_mode={self.correction_mode!r}. "
                f"Choose from {sorted(valid_modes)}."
            )

        self.max_correction_fraction = float(
            kwargs.get(
                "max_correction_fraction",
                0.50,
            )
        )
        self.minimum_correction_fraction = float(
            kwargs.get(
                "minimum_correction_fraction",
                0.05,
            )
        )
        self.clamp_physics_nonnegative = bool(
            kwargs.get(
                "clamp_physics_nonnegative",
                True,
            )
        )
        self.enforce_nonnegative_output = bool(
            kwargs.get(
                "enforce_nonnegative_output",
                True,
            )
        )

        # Original EPFL neural-network settings.
        self.learn_initial_hidden_states = kwargs[
            "learn_initial_hidden_states"
        ]
        self.feed_input_through_nn = kwargs[
            "feed_input_through_nn"
        ]
        self.input_nn_hidden_sizes = list(
            kwargs["input_nn_hidden_sizes"]
        )
        self.lstm_hidden_size = int(
            kwargs["lstm_hidden_size"]
        )
        self.lstm_num_layers = int(
            kwargs["lstm_num_layers"]
        )
        self.layer_norm = bool(
            kwargs["layer_norm"]
        )
        self.output_nn_hidden_sizes = list(
            kwargs["output_nn_hidden_sizes"]
        )
        division_factor = kwargs.get(
            "division_factor",
            10.0,
        )

        self.division_factor = float(
            np.asarray(
                division_factor,
                dtype=float,
            ).reshape(-1)[0]
        )
        self.eps = float(
            kwargs.get("eps", 1e-8)
        )

        # The original names are temperature_min/range, but here they
        # describe the measured-heating target.
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

        self.initial_physics_weights = kwargs.get(
            "initial_physics_weights"
        )

        self.last_output = None
        self.last_D = None
        self.last_E = None

        self._build_model()

    def _build_model(self) -> None:
        d_indices: list[int] = []

        if self.include_previous_target_in_D:
            d_indices.extend(
                self.temperature_column
            )

        d_indices.extend(
            self.inputs_D
        )

        if self.include_physics_in_D:
            d_indices.extend(
                self.power_column
            )

        if (
            self.include_out_column_in_D
            and self.out_column is not None
        ):
            d_indices.append(
                int(self.out_column)
            )

        # Remove repeated tensor indices while preserving order.
        self.d_input_indices = list(
            dict.fromkeys(d_indices)
        )

        if len(self.d_input_indices) == 0:
            raise ValueError(
                "The data-driven branch D has no input columns."
            )

        n_d_inputs = len(
            self.d_input_indices
        )

        if self.feed_input_through_nn:
            sizes = (
                [n_d_inputs]
                + self.input_nn_hidden_sizes
            )

            self.input_nn = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(
                            sizes[i],
                            sizes[i + 1],
                        ),
                        nn.ReLU(),
                    )
                    for i in range(
                        len(sizes) - 1
                    )
                ]
            )

            lstm_input_size = sizes[-1]

        else:
            self.input_nn = nn.ModuleList()
            lstm_input_size = n_d_inputs

        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=self.lstm_hidden_size,
            num_layers=self.lstm_num_layers,
            batch_first=True,
        )

        if self.layer_norm:
            self.norm = nn.LayerNorm(
                self.lstm_hidden_size
            )

        output_layers: list[nn.Module] = []
        previous_size = self.lstm_hidden_size

        for hidden_size in self.output_nn_hidden_sizes:
            output_layers.extend(
                [
                    nn.Linear(
                        previous_size,
                        hidden_size,
                    ),
                    nn.ReLU(),
                ]
            )
            previous_size = hidden_size

        output_layers.append(
            nn.Linear(previous_size, 1)
        )

        self.output_nn = nn.Sequential(
            *output_layers
        )

        self.physics_layer = PositiveLinear(
            in_features=len(
                self.power_column
            ),
            out_features=1,
            initial_weight=self.initial_physics_weights,
            require_bias=False,
        )

        if self.learn_initial_hidden_states:
            self.initial_h = nn.Parameter(
                torch.zeros(
                    self.lstm_num_layers,
                    self.lstm_hidden_size,
                )
            )
            self.initial_c = nn.Parameter(
                torch.zeros(
                    self.lstm_num_layers,
                    self.lstm_hidden_size,
                )
            )

        for name, parameter in self.named_parameters():
            if "physics_layer.log_weight" in name:
                continue

            if "bias" in name:
                nn.init.constant_(
                    parameter,
                    0.0,
                )

            elif (
                "weight" in name
                and parameter.ndim >= 2
            ):
                nn.init.xavier_normal_(
                    parameter
                )

    def _initialise_states(
        self,
        batch_size: int,
    ):
        if self.learn_initial_hidden_states:
            h = (
                self.initial_h
                .unsqueeze(1)
                .repeat(
                    1,
                    batch_size,
                    1,
                )
                .to(self.device)
            )
            c = (
                self.initial_c
                .unsqueeze(1)
                .repeat(
                    1,
                    batch_size,
                    1,
                )
                .to(self.device)
            )

        else:
            h = torch.zeros(
                self.lstm_num_layers,
                batch_size,
                self.lstm_hidden_size,
                device=self.device,
            )
            c = torch.zeros(
                self.lstm_num_layers,
                batch_size,
                self.lstm_hidden_size,
                device=self.device,
            )

        return h, c

    def _normalise_target(
        self,
        values: torch.Tensor,
    ) -> torch.Tensor:
        return (
            0.1
            + 0.8
            * (values - self.target_min)
            / (self.target_range + self.eps)
        )

    def _correction(
        self,
        raw_D: torch.Tensor,
        E: torch.Tensor,
    ) -> torch.Tensor:
        if self.correction_mode == "relative_to_physics":
            minimum_scale = (
                self.minimum_correction_fraction
                * self.target_range
            )

            scale = torch.maximum(
                E.abs(),
                minimum_scale,
            )

            return (
                self.max_correction_fraction
                * scale
                * torch.tanh(raw_D)
            )

        if self.correction_mode == "target_range":
            return (
                self.max_correction_fraction
                * self.target_range
                * torch.tanh(raw_D)
            )

        # "unbounded": still scale to target units for stable training.
        return (
            self.target_range
            / max(
                self.division_factor,
                self.eps,
            )
            * raw_D
        )

    def forward(
        self,
        x_: torch.Tensor,
        states=None,
        warm_start: bool = False,
    ):
        x = x_.clone()

        if x.ndim == 2:
            x = x.unsqueeze(1)

        batch_size = x.shape[0]

        if states is None:
            h, c = self._initialise_states(
                batch_size
            )

            self.last_output = x[
                :,
                -1,
                self.temperature_column,
            ].clone()

        else:
            h, c = states

        if (
            not warm_start
            and self.last_output is not None
            and self.include_previous_target_in_D
        ):
            x[
                :,
                -1,
                self.temperature_column,
            ] = self.last_output

        # -----------------------------
        # Data-driven branch D
        # -----------------------------
        embedding = x[
            :,
            :,
            self.d_input_indices,
        ]

        for layer in self.input_nn:
            embedding = layer(
                embedding
            )

        lstm_output, (h, c) = self.lstm(
            embedding,
            (h, c),
        )

        if self.layer_norm:
            lstm_output = self.norm(
                lstm_output
            )

        raw_D = self.output_nn(
            lstm_output[:, -1, :]
        )

        # -----------------------------
        # Physical branch E
        # -----------------------------
        physics_inputs = x[
            :,
            -1,
            self.power_column,
        ]

        if self.clamp_physics_nonnegative:
            physics_inputs = torch.clamp(
                physics_inputs,
                min=0.0,
            )

        E = self.physics_layer(
            physics_inputs
        )

        # -----------------------------
        # Combined prediction
        # -----------------------------
        D = self._correction(
            raw_D=raw_D,
            E=E,
        )

        q_prediction = E + D

        if self.enforce_nonnegative_output:
            q_prediction = F.relu(
                q_prediction
            )

        output = self._normalise_target(
            q_prediction
        )

        padded = torch.all(
            x[
                :,
                -1,
                self.case_column,
            ].abs() < self.eps,
            dim=1,
        )

        output[padded] = 0.0

        self.last_D = D.clone()
        self.last_E = E.clone()
        self.last_output = output.clone()

        return output, (h, c)

    @property
    def physics_weights(self):
        return (
            self.physics_layer.log_weight
            .exp()
            .detach()
            .cpu()
            .numpy()
            .reshape(-1)
        )

    @property
    def E_parameters(self):
        """
        Retain the four-list interface used by the original Model.fit().
        Only the first list is meaningful here.
        """

        return [
            list(
                np.asarray(
                    self.physics_weights,
                    dtype=float,
                )
            ),
            [],
            [],
            [],
        ]
