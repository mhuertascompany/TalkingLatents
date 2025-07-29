import numpy as np
from typing import Union, List


def classify_stellar_type(temperature: Union[float, np.ndarray],
                          mass: Union[float, np.ndarray]) -> Union[str, List[str]]:
    """
    Classify stellar type based on effective temperature and mass using the main sequence classification.

    Args:
        temperature: Effective temperature in Kelvin (float or array)
        mass: Mass in solar masses (float or array)

    Returns:
        Stellar type classification(s) as string or list of strings

    Note:
        - Temperature ranges are based on main sequence stars
        - Mass ranges help distinguish between different luminosity classes
        - Returns the most likely main sequence type based on both parameters
    """

    # Define stellar classification ranges
    stellar_classes = [
        {
            'type': 'O',
            'temp_min': 33000,
            'temp_max': float('inf'),
            'mass_min': 16.0,
            'mass_max': float('inf'),
            'color': 'blue'
        },
        {
            'type': 'B',
            'temp_min': 10000,
            'temp_max': 33000,
            'mass_min': 2.1,
            'mass_max': 16.0,
            'color': 'bluish white'
        },
        {
            'type': 'A',
            'temp_min': 7300,
            'temp_max': 10000,
            'mass_min': 1.4,
            'mass_max': 2.1,
            'color': 'white'
        },
        {
            'type': 'F',
            'temp_min': 6000,
            'temp_max': 7300,
            'mass_min': 1.04,
            'mass_max': 1.4,
            'color': 'yellowish white'
        },
        {
            'type': 'G',
            'temp_min': 5300,
            'temp_max': 6000,
            'mass_min': 0.8,
            'mass_max': 1.04,
            'color': 'yellow'
        },
        {
            'type': 'K',
            'temp_min': 3900,
            'temp_max': 5300,
            'mass_min': 0.45,
            'mass_max': 0.8,
            'color': 'light orange'
        },
        {
            'type': 'M',
            'temp_min': 2300,
            'temp_max': 3900,
            'mass_min': 0.08,
            'mass_max': 0.45,
            'color': 'light orangish red'
        }
    ]

    def classify_single(temp, mass):
        """Classify a single star"""

        # First, try to match both temperature and mass
        best_matches = []

        for star_class in stellar_classes:
            temp_match = star_class['temp_min'] <= temp < star_class['temp_max']
            mass_match = star_class['mass_min'] <= mass <= star_class['mass_max']

            if temp_match and mass_match:
                return star_class['type']
            elif temp_match:
                best_matches.append((star_class['type'], 'temp_only'))
            elif mass_match:
                best_matches.append((star_class['type'], 'mass_only'))

        # If no perfect match, prioritize temperature over mass for main sequence
        for star_class in stellar_classes:
            if star_class['temp_min'] <= temp < star_class['temp_max']:
                # Check if mass is reasonable (within 2x the range)
                mass_factor = max(mass / star_class['mass_max'], star_class['mass_min'] / mass)
                if mass_factor <= 2.0:  # Allow some flexibility
                    return star_class['type']
                else:
                    # Temperature matches but mass is very off - might be giant/dwarf
                    if mass > star_class['mass_max'] * 2:
                        return f"{star_class['type']}-giant"
                    else:
                        return f"{star_class['type']}-dwarf"

        # If still no match, find closest temperature match
        temp_distances = []
        for star_class in stellar_classes:
            temp_center = (star_class['temp_min'] + star_class['temp_max']) / 2
            distance = abs(temp - temp_center)
            temp_distances.append((distance, star_class['type']))

        # Return closest temperature match
        closest_type = min(temp_distances)[1]
        return f"{closest_type}-uncertain"

    # Handle both single values and arrays
    if isinstance(temperature, (int, float)) and isinstance(mass, (int, float)):
        return classify_single(temperature, mass)
    else:
        # Convert to numpy arrays for vectorized operation
        temp_array = np.atleast_1d(temperature)
        mass_array = np.atleast_1d(mass)

        if len(temp_array) != len(mass_array):
            raise ValueError("Temperature and mass arrays must have the same length")

        results = []
        for t, m in zip(temp_array, mass_array):
            results.append(classify_single(t, m))

        return results


def get_stellar_properties(stellar_type: str) -> dict:
    """
    Get typical properties for a given stellar type

    Args:
        stellar_type: Stellar classification (e.g., 'G', 'K', 'M')

    Returns:
        Dictionary with typical properties
    """

    properties = {
        'O': {
            'temp_range': (33000, 50000),
            'mass_range': (16, 90),
            'radius_range': (6.6, 15),
            'luminosity_range': (30000, 1000000),
            'color': 'blue',
            'hydrogen_lines': 'Weak',
            'percentage': 0.00003
        },
        'B': {
            'temp_range': (10000, 33000),
            'mass_range': (2.1, 16),
            'radius_range': (1.8, 6.6),
            'luminosity_range': (25, 30000),
            'color': 'bluish white',
            'hydrogen_lines': 'Medium',
            'percentage': 0.12
        },
        'A': {
            'temp_range': (7300, 10000),
            'mass_range': (1.4, 2.1),
            'radius_range': (1.4, 1.8),
            'luminosity_range': (5, 25),
            'color': 'white',
            'hydrogen_lines': 'Strong',
            'percentage': 0.61
        },
        'F': {
            'temp_range': (6000, 7300),
            'mass_range': (1.04, 1.4),
            'radius_range': (1.15, 1.4),
            'luminosity_range': (1.5, 5),
            'color': 'yellowish white',
            'hydrogen_lines': 'Medium',
            'percentage': 3.0
        },
        'G': {
            'temp_range': (5300, 6000),
            'mass_range': (0.8, 1.04),
            'radius_range': (0.96, 1.15),
            'luminosity_range': (0.6, 1.5),
            'color': 'yellow',
            'hydrogen_lines': 'Weak',
            'percentage': 7.6
        },
        'K': {
            'temp_range': (3900, 5300),
            'mass_range': (0.45, 0.8),
            'radius_range': (0.7, 0.96),
            'luminosity_range': (0.08, 0.6),
            'color': 'light orange',
            'hydrogen_lines': 'Very weak',
            'percentage': 12
        },
        'M': {
            'temp_range': (2300, 3900),
            'mass_range': (0.08, 0.45),
            'radius_range': (0.1, 0.7),
            'luminosity_range': (0.0001, 0.08),
            'color': 'light orangish red',
            'hydrogen_lines': 'Very weak',
            'percentage': 76
        }
    }

    # Handle suffixes like '-giant', '-dwarf', '-uncertain'
    base_type = stellar_type.split('-')[0]

    if base_type in properties:
        return properties[base_type]
    else:
        return None
