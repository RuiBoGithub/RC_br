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


def occupancy_based_ach(hour, occupancy, occupied_ach=1.5, unoccupied_ach=0.3,
                        occupancy_threshold=0.01):
    if occupancy > occupancy_threshold:
        return occupied_ach
    else:
        return unoccupied_ach