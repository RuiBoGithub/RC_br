# zone_debug.py
from building_physics import Zone


class DebugZone(Zone):
    """
    A Zone subclass that:
    - Stores ach_vent and ach_infl as attributes.
    - Allows dynamic update of ach_vent (property setter recalculates h_ve_adj).
    - Records intermediate calculation values for debugging.
    """
    def __init__(self, *args, **kwargs):
        # Extract ventilation parameters from kwargs before passing to super()
        self._ach_vent = kwargs.pop('ach_vent', 1.5)
        self._ach_infl = kwargs.pop('ach_infl', 0.5)
        self._occ_area = kwargs.pop('occ_area', None)
        self._ventilation_efficiency = kwargs.get('ventilation_efficiency', 0.6)

        super().__init__(*args, **kwargs)

        # Recalculate h_ve_adj with the stored values (super().__init__ already
        # computed it using the original args, but we now store them for later updates)
        self.update_ventilation_conductance()

        # Debug logging
        self._debug_log = []
        self._current_debug = {}

    def update_ventilation_conductance(self):
        """Recalculate h_ve_adj from current ach_vent, ach_infl, and efficiency."""
        ach_tot = self._ach_infl + self._ach_vent
        b_ek = (1 - (self._ach_vent / ach_tot) * self._ventilation_efficiency)
        self.h_ve_adj = 1200 * b_ek * self.room_vol * (ach_tot / 3600)
        
    @property
    def occ_area(self):
        if self._occ_area is None:
            return self.floor_area
        return self._occ_area
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

    @ach_infl.setter
    def ach_infl(self, value):
        self._ach_infl = value
        self.update_ventilation_conductance()

    def solve_energy(
        self,
        internal_gains,
        solar_gains,
        t_out,
        t_m_prev,
        t_air_prev=None,
    ):
        """
        Override solve_energy to capture t_air_0 and t_air_10 before
        the supply systems are called.

        Updated for 3R2C, where both t_m_prev and t_air_prev are dynamic states.
        """

        if t_air_prev is None:
            t_air_prev = getattr(self, "t_air", t_m_prev)

        original_calc = self.calc_temperatures_crank_nicolson
        air_temps = {}
        mass_temps = {}

        def wrapper(
            energy_demand,
            ig,
            sg,
            tout,
            t_m_p,
            t_air_p=None,
        ):
            if t_air_p is None:
                t_air_p = getattr(self, "t_air", t_m_p)

            t_m_result, t_air_result, t_op_result = original_calc(
                energy_demand,
                ig,
                sg,
                tout,
                t_m_p,
                t_air_p,
            )

            air_temps[energy_demand] = t_air_result
            mass_temps[energy_demand] = t_m_result

            return t_m_result, t_air_result, t_op_result

        self.calc_temperatures_crank_nicolson = wrapper

        try:
            super().solve_energy(
                internal_gains=internal_gains,
                solar_gains=solar_gains,
                t_out=t_out,
                t_m_prev=t_m_prev,
                t_air_prev=t_air_prev,
            )
        finally:
            self.calc_temperatures_crank_nicolson = original_calc

        t_air_0 = air_temps.get(0.0, air_temps.get(0, None))
        t_m_0 = mass_temps.get(0.0, mass_temps.get(0, None))

        test_power = 10.0 * self.floor_area
        t_air_10 = air_temps.get(test_power, None)
        t_m_10 = mass_temps.get(test_power, None)

        debug_snapshot = {
            "t_m_prev": t_m_prev,
            "t_air_prev": t_air_prev,
            "t_out": t_out,
            "internal_gains": internal_gains,
            "solar_gains": solar_gains,

            "has_heating_demand": self.has_heating_demand,
            "has_cooling_demand": self.has_cooling_demand,

            "t_air_free": t_air_0,
            "t_m_free": t_m_0,
            "t_air_test": t_air_10,
            "t_m_test": t_m_10,

            "delta_t_air": (
                t_air_10 - t_air_0
                if t_air_0 is not None and t_air_10 is not None
                else None
            ),

            "energy_demand_unrestricted": getattr(
                self,
                "energy_demand_unrestricted",
                None,
            ),
            "energy_demand": self.energy_demand,

            "t_air_final": self.t_air,
            "t_m_final": self.t_m,
            "t_air_next": getattr(self, "t_air_next", None),
            "t_m_next": getattr(self, "t_m_next", None),

            "h_ve_adj": self.h_ve_adj,
            "ach_vent": self.ach_vent,
            "ach_infl": self.ach_infl,

            "heating_setpoint": self.t_set_heating,
            "cooling_setpoint": self.t_set_cooling,
        }

        self._current_debug = debug_snapshot
        self._debug_log.append(debug_snapshot)

    def get_debug_dataframe(self):
        """Return the full debug log as a pandas DataFrame."""
        import pandas as pd
        return pd.DataFrame(self._debug_log)

    def print_latest_debug(self):
        """Print the most recent debug snapshot."""
        d = self._current_debug
        print("--- Zone Debug Snapshot ---")
        for key, val in d.items():
            print(f"{key:30s}: {val}")
        print("---------------------------")