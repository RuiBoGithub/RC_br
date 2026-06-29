import os
import sys
import unittest


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from building_physics import Zone
import emission_system
import supply_system


def make_office():
    office = Zone(
        window_area=13.5,
        walls_area=15.19 - 13.5,
        floor_area=34.3,
        room_vol=106.33,
        total_internal_area=142.38,
        lighting_load=11.7,
        u_walls=0.2,
        u_windows=1.1,
        ach_vent=1.5,
        ach_infl=0.5,
        ventilation_efficiency=0,
        thermal_capacitance_per_floor_area=165000,
        t_set_heating=20,
        t_set_cooling=26,
        max_cooling_energy_per_floor_area=-12,
        max_heating_energy_per_floor_area=12,
        heating_supply_system=supply_system.DirectHeater,
        cooling_supply_system=supply_system.DirectCooler,
        heating_emission_system=emission_system.AirConditioning,
        cooling_emission_system=emission_system.AirConditioning,
    )
    office.rc_order = "5R2C"
    return office


class Test5R2CState(unittest.TestCase):
    def test_repeated_temperature_evaluations_are_stateless_for_same_previous_state(self):
        office = make_office()
        args = {
            "energy_demand": 0.0,
            "internal_gains": 10.0,
            "solar_gains": 2000.0,
            "t_out": 10.0,
            "t_m_prev": 22.0,
            "t_air_prev": 19.0,
        }

        first = office.calc_temperatures_crank_nicolson(**args)
        first_next = (office.t_m_next, office.t_air_next)

        office.calc_temperatures_crank_nicolson(
            10.0 * office.floor_area,
            args["internal_gains"],
            args["solar_gains"],
            args["t_out"],
            args["t_m_prev"],
            args["t_air_prev"],
        )

        second = office.calc_temperatures_crank_nicolson(**args)
        second_next = (office.t_m_next, office.t_air_next)

        self.assertEqual(first, second)
        self.assertEqual(first_next, second_next)


if __name__ == "__main__":
    unittest.main()
