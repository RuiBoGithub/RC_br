import supply_system
import emission_system
class Zone(object):
    '''Sets the parameters of the zone. '''

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
                 u_walls=0.2,
                 u_windows=1.1,
                 ach_vent=1.5,
                 ach_infl=0.5,
                 ventilation_efficiency=0.6,
                 thermal_capacitance_per_floor_area=165000,
                 t_set_heating=20.0,
                 t_set_cooling=26.0,
                 max_cooling_energy_per_floor_area=-float("inf"),
                 max_heating_energy_per_floor_area=float("inf"),
                 heating_supply_system=supply_system.OilBoilerMed,
                 cooling_supply_system=supply_system.HeatPumpAir,
                 heating_emission_system=emission_system.NewRadiators,
                 cooling_emission_system=emission_system.AirConditioning,
                 ):


        # Zone Dimensions
        self.window_area = window_area  # [m2] Window Area

        # Fenestration and Lighting Properties
        self.lighting_load = lighting_load  # [kW/m2] lighting load
        self.lighting_control = lighting_control  # [lux] Lighting set point
        # How the light entering the window is transmitted to the working plane
        self.lighting_utilisation_factor = lighting_utilisation_factor
        # How dirty the window is. Section 2.2.3.1 Environmental Science
        # Handbook
        self.lighting_maintenance_factor = lighting_maintenance_factor

        # Calculated Properties
        self.floor_area = floor_area  # [m2] Floor Area

        # [m2] Effective Mass Area assuming a medium weight zone
        self.mass_area = self.floor_area * 2.5

        self.room_vol = room_vol  # [m3] Room Volume

        # Air capacitance for 3R2C model [J/K]
        self.c_air = 1200.0 * self.room_vol

        self.total_internal_area = total_internal_area
        self.A_t = self.total_internal_area

        # Single Capacitance  5 conductance Model Parameters
        # [kWh/K] Room Capacitance. Default based on ISO standard 12.3.1.2 for medium heavy zones
        self.c_m = thermal_capacitance_per_floor_area * self.floor_area
        # Conductance of opaque surfaces to exterior [W/K]
        self.h_tr_em = u_walls * walls_area
        # Conductance to exterior through glazed surfaces [W/K], based on
        # U-wert of 1W/m2K
        self.h_tr_w = u_windows * window_area

        # Determine the ventilation conductance
        ach_tot = ach_infl + ach_vent  # Total Air Changes Per Hour
        # temperature adjustment factor taking ventilation and infiltration
        # [ISO: E -27]
        b_ek = (1 - (ach_vent / (ach_tot)) * ventilation_efficiency)
        self.h_ve_adj = 1200 * b_ek * self.room_vol * \
            (ach_tot / 3600)  # Conductance through ventilation [W/M]
        # transmittance from the internal air to the thermal mass of the
        # zone
        self.h_tr_ms = 9.1 * self.mass_area
        # Conductance from the conditioned air to interior zone surface
        self.h_tr_is = self.total_internal_area * 3.45

        # Thermal set points
        self.t_set_heating = t_set_heating
        self.t_set_cooling = t_set_cooling

        # Thermal Properties
        self.has_heating_demand = False  # Boolean for if heating is required
        self.has_cooling_demand = False  # Boolean for if cooling is required
        self.max_cooling_energy = max_cooling_energy_per_floor_area * \
            self.floor_area  # max cooling load (W/m2)
        self.max_heating_energy = max_heating_energy_per_floor_area * \
            self.floor_area  # max heating load (W/m2)

        # Zone System Properties
        self.heating_supply_system = heating_supply_system
        self.cooling_supply_system = cooling_supply_system
        self.heating_emission_system = heating_emission_system
        self.cooling_emission_system = cooling_emission_system

    @property
    def h_tr_1(self):
        """
        Definition to simplify calc_phi_m_tot
        # (C.6) in [C.3 ISO 13790]
        """
        return 1.0 / (1.0 / self.h_ve_adj + 1.0 / self.h_tr_is)

    @property
    def h_tr_2(self):
        """
        Definition to simplify calc_phi_m_tot
        # (C.7) in [C.3 ISO 13790]
        """
        return self.h_tr_1 + self.h_tr_w

    @property
    def h_tr_3(self):
        """
        Definition to simplify calc_phi_m_tot
        # (C.8) in [C.3 ISO 13790]
        """
        return 1.0 / (1.0 / self.h_tr_2 + 1.0 / self.h_tr_ms)

    @property
    def t_opperative(self):
        """
        Operative temperature approximation for the 3R2C model.

        Since there is no explicit surface node, the mass temperature is used
        as a proxy for mean radiant temperature.
        """
        return 0.3 * self.t_air + 0.7 * self.t_m

    def solve_lighting(self, illuminance, occupancy):
        """
        Calculates the lighting demand for a set timestep

        :param illuminance: Illuminance transmitted through the window [Lumens]
        :type illuminance: float
        :param occupancy: Probability of full occupancy
        :type occupancy: float

        :return: self.lighting_demand, Lighting Energy Required for the timestep
        :rtype: float

        """
        # Cite: Environmental Science Handbook, SV Szokolay, Section 2.2.1.3
        # also, this might be sped up by pre-calculating the constants, but idk. first check with profiler...
        lux = (illuminance * self.lighting_utilisation_factor *
               self.lighting_maintenance_factor) / self.floor_area  # [Lux]

        if lux < self.lighting_control and occupancy > 0:
            # Lighting demand for the hour
            self.lighting_demand = self.lighting_load * self.floor_area
        else:
            self.lighting_demand = 0

    def solve_energy(
        self,
        internal_gains,
        solar_gains,
        t_out,
        t_m_prev,
        t_air_prev=None,
    ):
        """
        Calculates heating and cooling consumption for one timestep.
        """

        if t_air_prev is None:
            t_air_prev = getattr(self, "t_air", t_m_prev)

        self.has_demand(
            internal_gains,
            solar_gains,
            t_out,
            t_m_prev,
            t_air_prev,
        )

        if not self.has_heating_demand and not self.has_cooling_demand:

            self.energy_demand = 0.0

            self.heating_demand = 0.0
            self.cooling_demand = 0.0
            self.heating_sys_electricity = 0.0
            self.heating_sys_fossils = 0.0
            self.cooling_sys_electricity = 0.0
            self.cooling_sys_fossils = 0.0
            self.electricity_out = 0.0
            self.cop = float("nan")

        else:

            self.calc_energy_demand(
                internal_gains,
                solar_gains,
                t_out,
                t_m_prev,
                t_air_prev,
            )

            self.calc_temperatures_crank_nicolson(
                self.energy_demand,
                internal_gains,
                solar_gains,
                t_out,
                t_m_prev,
                t_air_prev,
            )

            supply_director = supply_system.SupplyDirector()

            if self.has_heating_demand:
                supply_director.set_builder(
                    self.heating_supply_system(
                        load=self.energy_demand,
                        t_out=t_out,
                        heating_supply_temperature=self.heating_supply_temperature,
                        cooling_supply_temperature=self.cooling_supply_temperature,
                        has_heating_demand=self.has_heating_demand,
                        has_cooling_demand=self.has_cooling_demand,
                    )
                )

                supplyOut = supply_director.calc_system()

                self.heating_demand = self.energy_demand
                self.heating_sys_electricity = supplyOut.electricity_in
                self.heating_sys_fossils = supplyOut.fossils_in
                self.cooling_demand = 0.0
                self.cooling_sys_electricity = 0.0
                self.cooling_sys_fossils = 0.0
                self.electricity_out = supplyOut.electricity_out

            elif self.has_cooling_demand:
                supply_director.set_builder(
                    self.cooling_supply_system(
                        load=self.energy_demand * -1.0,
                        t_out=t_out,
                        heating_supply_temperature=self.heating_supply_temperature,
                        cooling_supply_temperature=self.cooling_supply_temperature,
                        has_heating_demand=self.has_heating_demand,
                        has_cooling_demand=self.has_cooling_demand,
                    )
                )

                supplyOut = supply_director.calc_system()

                self.heating_demand = 0.0
                self.heating_sys_electricity = 0.0
                self.heating_sys_fossils = 0.0
                self.cooling_demand = self.energy_demand
                self.cooling_sys_electricity = supplyOut.electricity_in
                self.cooling_sys_fossils = supplyOut.fossils_in
                self.electricity_out = supplyOut.electricity_out

            self.cop = supplyOut.cop

        self.sys_total_energy = (
            self.heating_sys_electricity
            + self.heating_sys_fossils
            + self.cooling_sys_electricity
            + self.cooling_sys_fossils
        )

        self.heating_energy = self.heating_sys_electricity + self.heating_sys_fossils
        self.cooling_energy = self.cooling_sys_electricity + self.cooling_sys_fossils

    # TODO: rename. this is expected to return a boolean. instead, it changes state??? you don't want to change state...
    # why not just return has_heating_demand and has_cooling_demand?? then call the function "check_demand"
    # has_heating_demand, has_cooling_demand = self.check_demand(...)
    def has_demand(
        self,
        internal_gains,
        solar_gains,
        t_out,
        t_m_prev,
        t_air_prev=None,
    ):
        """
        Determines whether the building requires heating or cooling.
        """

        if t_air_prev is None:
            t_air_prev = getattr(self, "t_air", t_m_prev)

        energy_demand = 0.0

        self.calc_temperatures_crank_nicolson(
            energy_demand,
            internal_gains,
            solar_gains,
            t_out,
            t_m_prev,
            t_air_prev,
        )

        self.t_air_free = self.t_air

        if self.t_air < self.t_set_heating:
            self.has_heating_demand = True
            self.has_cooling_demand = False

        elif self.t_air > self.t_set_cooling:
            self.has_cooling_demand = True
            self.has_heating_demand = False

        else:
            self.has_heating_demand = False
            self.has_cooling_demand = False

    def calc_temperatures_crank_nicolson(
        self,
        energy_demand,
        internal_gains,
        solar_gains,
        t_out,
        t_m_prev,
        t_air_prev=None,
    ):
        """
        Determines node temperatures using the 3R2C model.
        """

        if t_air_prev is None:
            t_air_prev = getattr(self, "t_air", t_m_prev)

        self.calc_heat_flow_3r2c(
            internal_gains=internal_gains,
            solar_gains=solar_gains,
            energy_demand=energy_demand,
        )

        self.calc_t_air_t_m_next_3r2c(
            t_out=t_out,
            t_air_prev=t_air_prev,
            t_m_prev=t_m_prev,
        )

        self.t_air = self.t_air_next
        self.t_m = self.t_m_next

        # Compatibility with old 5R1C code.
        # 3R2C has no explicit surface node.
        self.t_s = self.t_m

        return self.t_m, self.t_air, self.t_opperative


    def calc_heat_flow_3r2c(self, internal_gains, solar_gains, energy_demand):
        """
        Splits gains for the 3R2C model.

        Compared with the original 5R1C code, the surface node is removed.
        Therefore, surface gains are merged into the thermal mass side.
        """

        # Original ISO-style split retained as far as possible
        phi_ia_base = 0.5 * internal_gains

        phi_st_base = (
            1
            - (self.mass_area / self.A_t)
            - (self.h_tr_w / (9.1 * self.A_t))
        ) * (0.5 * internal_gains + solar_gains)

        phi_m_base = (
            self.mass_area / self.A_t
        ) * (0.5 * internal_gains + solar_gains)

        # In 3R2C there is no surface node, so phi_st is assigned to the mass side.
        self.phi_ia = phi_ia_base
        self.phi_m = phi_m_base + phi_st_base

        emDirector = emission_system.EmissionDirector()

        if energy_demand > 0:
            emDirector.set_builder(
                self.heating_emission_system(energy_demand=energy_demand)
            )
        else:
            emDirector.set_builder(
                self.cooling_emission_system(energy_demand=energy_demand)
            )

        flows = emDirector.calc_flows()

        # No surface node in 3R2C.
        # Surface heat is assigned to the mass node.
        self.phi_ia += flows.phi_ia_plus
        self.phi_m += flows.phi_m_plus + flows.phi_st_plus

        self.heating_supply_temperature = flows.heating_supply_temperature
        self.cooling_supply_temperature = flows.cooling_supply_temperature


    def calc_t_air_t_m_next_3r2c(self, t_out, t_air_prev, t_m_prev):
        """
        Solves the 3R2C state equations using Crank-Nicolson.

        C_air dT_air/dt =
            H_ao (T_out - T_air)
            + H_am (T_m - T_air)
            + phi_ia

        C_m dT_m/dt =
            H_mo (T_out - T_m)
            + H_am (T_air - T_m)
            + phi_m
        """

        dt = 3600.0

        # Capacitances
        c_air = getattr(self, "c_air", 1200.0 * self.room_vol)
        c_m = self.c_m

        # Three conductance paths
        h_ao = self.h_ve_adj + self.h_tr_w
        h_am = self.h_tr_ms
        h_mo = self.h_tr_em

        # State matrix A for dT/dt = A*T + b
        a11 = -(h_ao + h_am) / c_air
        a12 = h_am / c_air
        a21 = h_am / c_m
        a22 = -(h_mo + h_am) / c_m

        b1 = (h_ao * t_out + self.phi_ia) / c_air
        b2 = (h_mo * t_out + self.phi_m) / c_m

        # Crank-Nicolson:
        # (I - 0.5*dt*A) T_next = (I + 0.5*dt*A) T_prev + dt*b

        left_11 = 1.0 - 0.5 * dt * a11
        left_12 = -0.5 * dt * a12
        left_21 = -0.5 * dt * a21
        left_22 = 1.0 - 0.5 * dt * a22

        right_1 = (
            (1.0 + 0.5 * dt * a11) * t_air_prev
            + (0.5 * dt * a12) * t_m_prev
            + dt * b1
        )

        right_2 = (
            (0.5 * dt * a21) * t_air_prev
            + (1.0 + 0.5 * dt * a22) * t_m_prev
            + dt * b2
        )

        det = left_11 * left_22 - left_12 * left_21

        if abs(det) < 1e-12:
            raise ZeroDivisionError("3R2C Crank-Nicolson matrix is singular.")

        self.t_air_next = (right_1 * left_22 - left_12 * right_2) / det
        self.t_m_next = (left_11 * right_2 - right_1 * left_21) / det

    def calc_energy_demand(
        self,
        internal_gains,
        solar_gains,
        t_out,
        t_m_prev,
        t_air_prev=None,
    ):
        """
        Calculates the energy demand of the space if heating/cooling is active.
        """

        if t_air_prev is None:
            t_air_prev = getattr(self, "t_air", t_m_prev)

        # Step 1: air temperature with no heating/cooling
        energy_demand_0 = 0.0

        t_air_0 = self.calc_temperatures_crank_nicolson(
            energy_demand_0,
            internal_gains,
            solar_gains,
            t_out,
            t_m_prev,
            t_air_prev,
        )[1]

        # Step 2: target air temperature
        if self.has_heating_demand:
            t_air_set = self.t_set_heating
        elif self.has_cooling_demand:
            t_air_set = self.t_set_cooling
        else:
            raise NameError(
                "calc_energy_demand() was called although no heating or cooling is required."
            )

        # Trial load: 10 W/m2
        energy_floorAx10 = 10.0 * self.floor_area

        t_air_10 = self.calc_temperatures_crank_nicolson(
            energy_floorAx10,
            internal_gains,
            solar_gains,
            t_out,
            t_m_prev,
            t_air_prev,
        )[1]

        # Avoid division by zero or unstable interpolation
        if abs(t_air_10 - t_air_0) < 1e-9:
            self.energy_demand_unrestricted = 0.0
        else:
            self.calc_energy_demand_unrestricted(
                energy_floorAx10,
                t_air_set,
                t_air_0,
                t_air_10,
            )

        # Step 3: check capacity limits
        if self.max_cooling_energy <= self.energy_demand_unrestricted <= self.max_heating_energy:
            self.energy_demand = self.energy_demand_unrestricted
            self.t_air_ac = t_air_set

        elif self.energy_demand_unrestricted > self.max_heating_energy:
            self.energy_demand = self.max_heating_energy

        elif self.energy_demand_unrestricted < self.max_cooling_energy:
            self.energy_demand = self.max_cooling_energy

        else:
            self.energy_demand = 0.0
            raise ValueError("Unknown heating/cooling system status.")

        # Final temperature calculation with actual heating/cooling demand
        self.calc_temperatures_crank_nicolson(
            self.energy_demand,
            internal_gains,
            solar_gains,
            t_out,
            t_m_prev,
            t_air_prev,
        )

    def calc_energy_demand_unrestricted(self, energy_floorAx10, t_air_set, t_air_0, t_air_10):
        """
        Calculates the energy demand of the system if it has no maximum output restrictions
        # (C.13) in [C.3 ISO 13790]


        Based on the Thales Intercept Theorem.
        Where we set a heating case that is 10x the floor area and determine the temperature as a result
        Assuming that the relation is linear, one can draw a right angle triangle.
        From this we can determine the heating level required to achieve the set point temperature
        This assumes a perfect HVAC control system
        """
        self.energy_demand_unrestricted = energy_floorAx10 * \
            (t_air_set - t_air_0) / (t_air_10 - t_air_0)

    def calc_heat_flow(self, t_out, internal_gains, solar_gains, energy_demand):
        """
        Calculates the heat flow from the solar gains, heating/cooling system, and internal gains into the building

        The input of the building is split into the air node, surface node, and thermal mass node based on
        on the following equations

        #C.1 - C.3 in [C.3 ISO 13790]

        Note that this equation has diverged slightly from the standard
        as the heating/cooling node can enter any node depending on the
        emission system selected

        """

        # Calculates the heat flows to various points of the building based on the breakdown in section C.2, formulas C.1-C.3
        # Heat flow to the air node
        self.phi_ia = 0.5 * internal_gains
        # Heat flow to the surface node
        self.phi_st = (1 - (self.mass_area / self.A_t) -
                       (self.h_tr_w / (9.1 * self.A_t))) * (0.5 * internal_gains + solar_gains)
        # Heatflow to the thermal mass node
        self.phi_m = (self.mass_area / self.A_t) * \
            (0.5 * internal_gains + solar_gains)

        # We call the EmissionDirector to modify these flows depending on the
        # system and the energy demand
        emDirector = emission_system.EmissionDirector()

        # Set the emission system to the type specified by the user
        if energy_demand > 0:
            emDirector.set_builder(self.heating_emission_system(
                energy_demand=energy_demand))
        else:
            emDirector.set_builder(self.cooling_emission_system(
                energy_demand=energy_demand))
        # Calculate the new flows to each node based on the heating/cooling system
        flows = emDirector.calc_flows()

        # Set modified flows to building object
        self.phi_ia += flows.phi_ia_plus
        self.phi_st += flows.phi_st_plus
        self.phi_m += flows.phi_m_plus

        # Set supply temperature to building object
        self.heating_supply_temperature = flows.heating_supply_temperature
        self.cooling_supply_temperature = flows.cooling_supply_temperature

    def calc_t_m_next(self, t_m_prev, t_air_prev):
        """
        Primary Equation, calculates the temperature of the next time step
        # (C.4) in [C.3 ISO 13790]
        """

        self.t_m_next = ((t_m_prev * ((self.c_m / 3600.0) - 0.5 * (self.h_tr_3 + self.h_tr_em))) +
                         self.phi_m_tot) / ((self.c_m / 3600.0) + 0.5 * (self.h_tr_3 + self.h_tr_em))

    def calc_phi_m_tot(self, t_out):
        """
        Calculates a global heat transfer. This is a definition used to simplify equation
        calc_t_m_next so it's not so long to write out
        # (C.5) in [C.3 ISO 13790]
        # h_ve = h_ve_adj and t_supply = t_out [9.3.2 ISO 13790]
        """

        t_supply = t_out  # ASSUMPTION: Supply air comes straight from the outside air

        self.phi_m_tot = self.phi_m + self.h_tr_em * t_out + \
            self.h_tr_3 * (self.phi_st + self.h_tr_w * t_out + self.h_tr_1 *
                           ((self.phi_ia / self.h_ve_adj) + t_supply)) / self.h_tr_2

    def calc_t_m(self, t_m_prev, t_air_prev):
        """
        Temperature used for the calculations, average between newly calculated and previous bulk temperature
        # (C.9) in [C.3 ISO 13790]
        """
        self.t_m = (self.t_m_next + t_m_prev, t_air_prev) / 2.0

    def calc_t_s(self, t_out):
        """
        Calculate the temperature of the inside room surfaces
        # (C.10) in [C.3 ISO 13790]
        # h_ve = h_ve_adj and t_supply = t_out [9.3.2 ISO 13790]
        """

        t_supply = t_out  # ASSUMPTION: Supply air comes straight from the outside air

        self.t_s = (self.h_tr_ms * self.t_m + self.phi_st + self.h_tr_w * t_out + self.h_tr_1 *
                    (t_supply + self.phi_ia / self.h_ve_adj)) / \
                   (self.h_tr_ms + self.h_tr_w + self.h_tr_1)

    def calc_t_air(self, t_out):
        """
        Calculate the temperature of the air node
        # (C.11) in [C.3 ISO 13790]
        # h_ve = h_ve_adj and t_supply = t_out [9.3.2 ISO 13790]
        """

        t_supply = t_out

        # Calculate the temperature of the inside air
        self.t_air = (self.h_tr_is * self.t_s + self.h_ve_adj *
                      t_supply + self.phi_ia) / (self.h_tr_is + self.h_ve_adj)
