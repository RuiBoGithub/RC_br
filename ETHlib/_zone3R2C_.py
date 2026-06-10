# zone_r3c2.py
import numpy as np

class R3C2Zone:
    """
    Simplified 3‑resistance, 2‑capacitance building model.
    Replaces the 5R1C star‑network while keeping the same public interface
    as the original 'Zone' class (solve_energy, solve_lighting, ...).
    """
    def __init__(self,
                 window_area=4.0,
                 walls_area=11.0,
                 floor_area=35.0,
                 room_vol=105,
                 total_internal_area=142.0,
                 lighting_load=11.7,
                 lighting_control=300.0,
                 lighting_utilisation_factor=0.45,
                 lighting_maintenance_factor=0.9,
                 u_walls=0.2,        # W/m²K (opaque envelope U‑value)
                 u_windows=1.1,      # W/m²K (window U‑value)
                 ach_vent=1.5,
                 ach_infl=0.5,
                 ventilation_efficiency=0.6,
                 thermal_capacitance_per_floor_area=165000,  # J/K per m²
                 t_set_heating=20.0,
                 t_set_cooling=26.0,
                 max_cooling_energy_per_floor_area=-float("inf"),
                 max_heating_energy_per_floor_area=float("inf"),
                 heating_supply_system=None,
                 cooling_supply_system=None,
                 heating_emission_system=None,
                 cooling_emission_system=None,
                 ):
        # Geometry
        self.floor_area = floor_area
        self.room_vol = room_vol
        self.window_area = window_area
        self.walls_area = walls_area

        # Resistances (converted to conductances in W/K)
        self.U_win = u_windows * window_area      # window conductance
        self.U_ea  = u_walls * walls_area          # opaque envelope to ambient
        # Interior <-> envelope conductance: typical value from ISO 13790
        self.U_ie  = 3.45 * total_internal_area     # (W/K) – surface to air
        # Envelope capacitance (J/K)
        self.C_e = thermal_capacitance_per_floor_area * floor_area
        # Indoor air + fast‑response mass capacitance
        # Physical air: 1200 J/m³K, but we inflate it for stability (e.g. factor 5)
        self.C_i = 1200 * room_vol * 5.0

        # Ventilation parameters (store for dynamic update)
        self._ach_vent = ach_vent
        self._ach_infl = ach_infl
        self._ventilation_efficiency = ventilation_efficiency
        self.update_ventilation_conductance()

        # Setpoints
        self.t_set_heating = t_set_heating
        self.t_set_cooling = t_set_cooling

        # Capacity limits (W)
        self.max_heating_energy = max_heating_energy_per_floor_area * floor_area
        self.max_cooling_energy = max_cooling_energy_per_floor_area * floor_area

        # State variables (will be set in solve_energy)
        self.T_i = 20.0      # indoor air temperature
        self.T_e = 20.0      # envelope temperature

        # Lighting
        self.lighting_load = lighting_load
        self.lighting_control = lighting_control
        self.lighting_utilisation_factor = lighting_utilisation_factor
        self.lighting_maintenance_factor = lighting_maintenance_factor
        self.lighting_demand = 0.0

        # Supply/Emission placeholders (not used in this simple model,
        # but stored for compatibility)
        self.heating_supply_system = heating_supply_system
        self.cooling_supply_system = cooling_supply_system
        self.heating_emission_system = heating_emission_system
        self.cooling_emission_system = cooling_emission_system

        # Outputs (to mimic Zone interface)
        self.heating_demand = 0.0
        self.cooling_demand = 0.0
        self.heating_sys_electricity = 0.0
        self.heating_sys_fossils = 0.0
        self.cooling_sys_electricity = 0.0
        self.cooling_sys_fossils = 0.0
        self.electricity_out = 0.0
        self.cop = float('nan')
        self.sys_total_energy = 0.0
        self.heating_energy = 0.0
        self.cooling_energy = 0.0
        self.has_heating_demand = False
        self.has_cooling_demand = False
        self.t_m_next = self.T_e  # compatibility alias

    # --- Property to allow dynamic ventilation update ---
    @property
    def ach_vent(self):
        return self._ach_vent

    @ach_vent.setter
    def ach_vent(self, value):
        self._ach_vent = value
        self.update_ventilation_conductance()

    @property
    def ach_infl(self):
        return self._ach_infl

    def update_ventilation_conductance(self):
        """Recalculate U_vent (W/K) from current ach values."""
        ach_tot = self._ach_infl + self._ach_vent
        # temperature adjustment factor (from ISO 13790)
        b_ek = 1 - (self._ach_vent / ach_tot) * self._ventilation_efficiency
        self.U_vent = 1200 * b_ek * self.room_vol * (ach_tot / 3600)

    # --- Lighting (unchanged logic) ---
    def solve_lighting(self, illuminance, occupancy):
        lux = (illuminance * self.lighting_utilisation_factor *
               self.lighting_maintenance_factor) / self.floor_area
        if lux < self.lighting_control and occupancy > 0:
            self.lighting_demand = self.lighting_load * self.floor_area
        else:
            self.lighting_demand = 0.0

    # --- Main thermal solver ---
    def solve_energy(self, internal_gains, solar_gains, t_out, t_m_prev):
        """
        Run one time step (1 hour) of the 3R2C model.
        Uses forward Euler with Δt = 3600 s.
        `t_m_prev` is used as the previous envelope temperature (T_e).
        """
        dt = 3600   # seconds
        # Previous state
        T_i_prev = self.T_i
        T_e_prev = self.T_e if t_m_prev is None else t_m_prev   # allow override

        # Total heat flow to indoor air (excluding heating/cooling power)
        # Gains are split: 50% radiative to envelope, 50% convective to air
        phi_i = (0.5 * internal_gains + solar_gains * self.window_area / self.floor_area
                 )  # (W) simplified solar distribution
        phi_e = 0.5 * internal_gains   # (W) to envelope

        # 1. Free‑floating temperatures (no heating/cooling)
        T_i_0 = T_i_prev + dt / self.C_i * (
            (t_out - T_i_prev) * (self.U_win + self.U_vent) +
            (T_e_prev - T_i_prev) * self.U_ie +
            phi_i
        )
        T_e_0 = T_e_prev + dt / self.C_e * (
            (t_out - T_e_prev) * self.U_ea +
            (T_i_prev - T_e_prev) * self.U_ie +
            phi_e
        )

        # 2. Determine if heating/cooling is needed
        self.has_heating_demand = T_i_0 < self.t_set_heating
        self.has_cooling_demand = T_i_0 > self.t_set_cooling

        if not self.has_heating_demand and not self.has_cooling_demand:
            # No demand – use free‑floating temperatures
            self.T_i = T_i_0
            self.T_e = T_e_0
            self.energy_demand = 0.0
            self.heating_demand = 0.0
            self.cooling_demand = 0.0
        else:
            # 3. Compute the power needed to bring T_i to setpoint
            if self.has_heating_demand:
                T_set = self.t_set_heating
                sign = 1.0
            else:
                T_set = self.t_set_cooling
                sign = -1.0

            # Linear extrapolation: test with Q = 10 W/m² × floor_area
            Q_test = 10.0 * self.floor_area
            T_i_test = T_i_prev + dt / self.C_i * (
                (t_out - T_i_prev) * (self.U_win + self.U_vent) +
                (T_e_prev - T_i_prev) * self.U_ie +
                phi_i + Q_test
            )
            # Unrestricted demand
            if abs(T_i_test - T_i_0) > 1e-6:
                Q_unrestricted = Q_test * (T_set - T_i_0) / (T_i_test - T_i_0)
            else:
                Q_unrestricted = 0.0

            # Apply capacity limits
            if self.has_heating_demand:
                Q_actual = min(Q_unrestricted, self.max_heating_energy)
                Q_actual = max(Q_actual, 0.0)
                self.heating_demand = Q_actual
                self.cooling_demand = 0.0
            else:
                # Cooling demand is negative
                Q_actual = max(Q_unrestricted, self.max_cooling_energy)
                Q_actual = min(Q_actual, 0.0)
                self.cooling_demand = Q_actual
                self.heating_demand = 0.0

            # 4. Apply actual power and update states
            self.T_i = T_i_prev + dt / self.C_i * (
                (t_out - T_i_prev) * (self.U_win + self.U_vent) +
                (T_e_prev - T_i_prev) * self.U_ie +
                phi_i + Q_actual
            )
            self.T_e = T_e_0   # envelope not directly affected by HVAC power in this simple model
            self.energy_demand = Q_actual

        # Set aliases for compatibility with the original run_model
        self.t_air = self.T_i
        self.t_m = self.T_e
        self.t_m_next = self.T_e
        # Simple COP = 1.0 (no supply/emission models), or you can attach real systems later
        self.cop = 1.0
        # For the original output logging: assume heating/cooling energy = demand (no losses)
        if self.has_heating_demand:
            self.heating_energy = abs(self.heating_demand)
            self.cooling_energy = 0.0
        else:
            self.heating_energy = 0.0
            self.cooling_energy = abs(self.cooling_demand)
        self.sys_total_energy = self.heating_energy + self.cooling_energy