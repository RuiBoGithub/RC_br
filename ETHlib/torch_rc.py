from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from _BR_ import make_ach, merge_params


@dataclass(frozen=True)
class RCStepResult:
    heating_energy: torch.Tensor
    next_state: torch.Tensor
    heating_demand: torch.Tensor
    energy_demand: torch.Tensor
    t_air: torch.Tensor
    t_m: torch.Tensor
    t_s: torch.Tensor
    has_heating_demand: torch.Tensor
    has_cooling_demand: torch.Tensor


class TorchRC5R1C(nn.Module):
    """
    Torch implementation of the existing ETHlib 5R1C hourly heating step.

    Physical parameters are loaded from an RCCase and registered as buffers,
    so they move with the module but are not trainable.
    """

    def __init__(
        self,
        *,
        floor_area,
        mass_area,
        total_internal_area,
        c_m,
        h_tr_em,
        h_tr_w,
        h_tr_ms,
        h_tr_is,
        h_ve_adj,
        t_set_heating,
        t_set_cooling,
        max_heating_energy,
        max_cooling_energy,
        dtype=torch.float64,
    ):
        super().__init__()
        self.dtype = dtype

        for name, value in {
            "floor_area": floor_area,
            "mass_area": mass_area,
            "total_internal_area": total_internal_area,
            "c_m": c_m,
            "h_tr_em": h_tr_em,
            "h_tr_w": h_tr_w,
            "h_tr_ms": h_tr_ms,
            "h_tr_is": h_tr_is,
            "base_h_ve_adj": h_ve_adj,
            "base_t_set_heating": t_set_heating,
            "t_set_cooling": t_set_cooling,
            "max_heating_energy": max_heating_energy,
            "max_cooling_energy": max_cooling_energy,
        }.items():
            self.register_buffer(
                name,
                torch.as_tensor(value, dtype=dtype),
            )

    @classmethod
    def from_case(
        cls,
        case,
        sampled_params=None,
        dtype=torch.float64,
    ) -> "TorchRC5R1C":
        if sampled_params is None:
            sampled_params = {}

        p = merge_params(sampled_params, case.default_params)
        geometry = case.geometry
        ach_vent, ach_infl = make_ach(
            p=p,
            geometry=geometry,
            calc_ach=_calc_ach_like_ethlib,
        )

        floor_area = geometry["FLOOR_AREA"]
        mass_area = floor_area * 2.5
        total_internal_area = floor_area * p["_alpha"]
        room_vol = geometry["VOLUME"]
        c_m = p["thermal_capacitance_per_floor_area"] * floor_area
        h_tr_em = p["u_walls"] * geometry["WALL_AREA"]
        h_tr_w = p["u_windows"] * geometry["WINDOW_AREA"]
        h_tr_ms = 9.1 * mass_area
        h_tr_is = total_internal_area * 3.45
        h_ve_adj = _h_ve_adj(
            ach_vent=ach_vent,
            ach_infl=ach_infl,
            ventilation_efficiency=p["ventilation_efficiency"],
            room_vol=room_vol,
        )

        return cls(
            floor_area=floor_area,
            mass_area=mass_area,
            total_internal_area=total_internal_area,
            c_m=c_m,
            h_tr_em=h_tr_em,
            h_tr_w=h_tr_w,
            h_tr_ms=h_tr_ms,
            h_tr_is=h_tr_is,
            h_ve_adj=h_ve_adj,
            t_set_heating=p["t_set_heating"],
            t_set_cooling=p["t_set_cooling"],
            max_heating_energy=float("inf") * floor_area,
            max_cooling_energy=-float("inf") * floor_area,
            dtype=dtype,
        )

    def step(
        self,
        *,
        internal_gains,
        solar_gains,
        t_out,
        previous_state,
        t_set_heating=None,
        h_ve_adj=None,
    ) -> RCStepResult:
        internal_gains = self._tensor(internal_gains)
        solar_gains = self._tensor(solar_gains)
        t_out = self._tensor(t_out)
        previous_state = self._tensor(previous_state)

        if t_set_heating is None:
            t_set_heating = self.base_t_set_heating
        else:
            t_set_heating = self._tensor(t_set_heating)

        if h_ve_adj is None:
            h_ve_adj = self.base_h_ve_adj
        else:
            h_ve_adj = self._tensor(h_ve_adj)

        zero = torch.zeros_like(t_out)
        floor_ax10 = 10.0 * self.floor_area

        free = self._calc_temperatures(
            energy_demand=zero,
            internal_gains=internal_gains,
            solar_gains=solar_gains,
            t_out=t_out,
            previous_state=previous_state,
            h_ve_adj=h_ve_adj,
        )

        has_heating = free["t_air"] < t_set_heating
        has_cooling = free["t_air"] > self.t_set_cooling
        has_demand = has_heating | has_cooling

        t_air_set = torch.where(
            has_heating,
            t_set_heating,
            self.t_set_cooling,
        )

        test = self._calc_temperatures(
            energy_demand=torch.zeros_like(t_out) + floor_ax10,
            internal_gains=internal_gains,
            solar_gains=solar_gains,
            t_out=t_out,
            previous_state=previous_state,
            h_ve_adj=h_ve_adj,
        )

        energy_unrestricted = (
            floor_ax10
            * (t_air_set - free["t_air"])
            / (test["t_air"] - free["t_air"])
        )

        energy_demand = torch.clamp(
            energy_unrestricted,
            min=self.max_cooling_energy,
            max=self.max_heating_energy,
        )
        energy_demand = torch.where(
            has_demand,
            energy_demand,
            zero,
        )

        final = self._calc_temperatures(
            energy_demand=energy_demand,
            internal_gains=internal_gains,
            solar_gains=solar_gains,
            t_out=t_out,
            previous_state=previous_state,
            h_ve_adj=h_ve_adj,
        )

        heating_demand = torch.where(
            has_heating,
            energy_demand,
            zero,
        )
        heating_energy = heating_demand

        return RCStepResult(
            heating_energy=heating_energy,
            next_state=final["t_m_next"],
            heating_demand=heating_demand,
            energy_demand=energy_demand,
            t_air=final["t_air"],
            t_m=final["t_m"],
            t_s=final["t_s"],
            has_heating_demand=has_heating,
            has_cooling_demand=has_cooling,
        )

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
        phi_ia = 0.5 * internal_gains + energy_demand
        phi_st = (
            1.0
            - (self.mass_area / self.total_internal_area)
            - (self.h_tr_w / (9.1 * self.total_internal_area))
        ) * (0.5 * internal_gains + solar_gains)
        phi_m = (
            self.mass_area
            / self.total_internal_area
            * (0.5 * internal_gains + solar_gains)
        )

        h_tr_1 = 1.0 / (1.0 / h_ve_adj + 1.0 / self.h_tr_is)
        h_tr_2 = h_tr_1 + self.h_tr_w
        h_tr_3 = 1.0 / (1.0 / h_tr_2 + 1.0 / self.h_tr_ms)
        t_supply = t_out

        phi_m_tot = (
            phi_m
            + self.h_tr_em * t_out
            + h_tr_3
            * (
                phi_st
                + self.h_tr_w * t_out
                + h_tr_1 * ((phi_ia / h_ve_adj) + t_supply)
            )
            / h_tr_2
        )

        t_m_next = (
            previous_state
            * ((self.c_m / 3600.0) - 0.5 * (h_tr_3 + self.h_tr_em))
            + phi_m_tot
        ) / ((self.c_m / 3600.0) + 0.5 * (h_tr_3 + self.h_tr_em))
        t_m = (t_m_next + previous_state) / 2.0
        t_s = (
            self.h_tr_ms * t_m
            + phi_st
            + self.h_tr_w * t_out
            + h_tr_1 * (t_supply + phi_ia / h_ve_adj)
        ) / (self.h_tr_ms + self.h_tr_w + h_tr_1)
        t_air = (
            self.h_tr_is * t_s
            + h_ve_adj * t_supply
            + phi_ia
        ) / (self.h_tr_is + h_ve_adj)

        return {
            "t_m_next": t_m_next,
            "t_m": t_m,
            "t_s": t_s,
            "t_air": t_air,
        }

    def _tensor(self, value):
        return torch.as_tensor(
            value,
            dtype=self.dtype,
            device=self.floor_area.device,
        )


def _calc_ach_like_ethlib(
    n_people,
    fresh_air_lps,
    atrium_ach,
    atrium_volume,
    infl_rate_m3ph_m2,
    geometry,
    window_opening_ach=0.0,
):
    mech_m3s = n_people * fresh_air_lps / 1000.0
    ach_vent = 3600.0 * mech_m3s / geometry["VOLUME"]
    nat_vent_m3s = atrium_ach * atrium_volume / 3600.0
    ach_atrium = 3600.0 * nat_vent_m3s / geometry["VOLUME"]
    infl_m3ph = infl_rate_m3ph_m2 * geometry["WALL_AREA"]
    ach_background_infl = infl_m3ph / geometry["VOLUME"]
    ach_infl = ach_background_infl + ach_atrium + window_opening_ach
    return ach_vent, ach_infl


def _h_ve_adj(
    *,
    ach_vent,
    ach_infl,
    ventilation_efficiency,
    room_vol,
):
    ach_tot = ach_infl + ach_vent
    b_ek = 1 - (ach_vent / ach_tot) * ventilation_efficiency
    return 1200 * b_ek * room_vol * (ach_tot / 3600)
