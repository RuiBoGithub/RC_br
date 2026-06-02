# ventilation_controller.py

def constant_ach(hour, *args, value=0.5, **kwargs):
    """
    Returns a constant ventilation rate every hour.
    Use this to test a fixed reduced ventilation (e.g., 0.5 ACH).
    """
    return value

def schedule_ach(hour, schedule, **kwargs):
    """
    Use a predefined list/array of hourly ach_vent values.
    :param schedule: array-like of length 8760 with desired ach values.
    """
    return schedule[hour]

def free_cooling_ach(indoor_temp, outdoor_temp, cooling_setpoint, heating_setpoint,
                     min_ach=0.5, max_ach=5.0, margin=1.0, deadband=0.5):
    """
    Simple free-cooling / economizer controller.

    If cooling is needed (indoor_temp > cooling_setpoint + deadband)
    AND outdoor air is cooler than indoor air by at least `margin`,
    then increase ventilation to max_ach. Otherwise, use min_ach.

    :param indoor_temp: current indoor air temperature [°C]
    :param outdoor_temp: outdoor air temperature [°C]
    :param cooling_setpoint: upper bound of comfort band [°C]
    :param heating_setpoint: lower bound (ignored in this simple version)
    :param min_ach: minimum mechanical ventilation rate [ACH]
    :param max_ach: maximum mechanical ventilation rate [ACH]
    :param margin: required positive difference (T_indoor - T_outdoor) to trigger free cooling [K]
    :param deadband: tolerance above cooling setpoint before cooling is considered [K]
    :return: desired mechanical ventilation rate [ACH]
    """
    if indoor_temp > cooling_setpoint + deadband and (indoor_temp - outdoor_temp) > margin:
        return max_ach
    else:
        return min_ach

def demand_controlled_ventilation(indoor_co2, co2_setpoint=1000, min_ach=0.5, max_ach=5.0):
    """
    Example DCV based on CO₂. Not used in the current heating/cooling model,
    but kept for future use.
    """
    # Simple linear control: if CO2 > setpoint, raise ventilation proportionally
    if indoor_co2 <= co2_setpoint:
        return min_ach
    else:
        # Linear ramp between setpoint and double the setpoint
        fraction = min((indoor_co2 - co2_setpoint) / co2_setpoint, 1.0)
        return min_ach + (max_ach - min_ach) * fraction
    
def occupancy_based_ach(hour, occupancy, occupied_ach=1.5, unoccupied_ach=0.3,
                        occupancy_threshold=0.01):
    """
    Occupancy‑based ventilation controller.

    When occupancy is above the threshold (people present), use `occupied_ach`.
    Otherwise (night / unoccupied) use `unoccupied_ach`.

    Parameters
    ----------
    hour : int
        Hour index (0…8759) – not used but kept for a uniform interface.
    occupancy : float
        Occupancy fraction (0 to 1) or number of people.
    occupied_ach : float
        Mechanical ventilation rate during occupied hours [ACH].
    unoccupied_ach : float
        Mechanical ventilation rate during unoccupied hours [ACH].
    occupancy_threshold : float
        Occupancy value above which the space is considered “occupied”.

    Returns
    -------
    float
        Desired mechanical ventilation rate [ACH].
    """
    if occupancy > occupancy_threshold:
        return occupied_ach
    else:
        return unoccupied_ach