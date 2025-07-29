import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
import re
import json
from tqdm import tqdm


class DataPreparator:
    """
    Utility class for preparing star data for training
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def load_star_data(self, latent_path: str, metadata_path: str) -> Tuple[np.ndarray, List[str]]:
        """
        Load star latent vectors and convert metadata to natural language descriptions

        Args:
            latent_path: Path to numpy file containing latent vectors
            metadata_path: Path to CSV/JSON file containing star parameters

        Returns:
            latent_vectors: numpy array of latent representations
            descriptions: list of natural language descriptions
        """
        # Load latent vectors
        latent_vectors = np.load(latent_path)

        # Load metadata
        if metadata_path.endswith('.csv'):
            metadata = pd.read_csv(metadata_path)
        elif metadata_path.endswith('.json'):
            with open(metadata_path, 'r') as f:
                metadata = pd.DataFrame(json.load(f))
        else:
            raise ValueError("Metadata file must be CSV or JSON")

        # Convert to descriptions
        descriptions = []
        for _, row in metadata.iterrows():
            desc = self.create_star_description(row.to_dict())
            descriptions.append(desc)

        assert len(latent_vectors) == len(descriptions), \
            f"Mismatch: {len(latent_vectors)} latent vectors, {len(descriptions)} descriptions"

        return latent_vectors, descriptions

    def create_star_description(self, star_params: Dict) -> str:
        """
        Convert star parameters to natural language description

        Args:
            star_params: Dictionary of star parameters

        Returns:
            Natural language description of the star
        """
        templates = [
            "Mass: {mass:.2f} solar masses, Temperature: {temperature:.0f} K, Luminosity: {luminosity:.3f} solar luminosities, Radius: {radius:.2f} solar radii",
            "This star has a mass of {mass:.2f} times that of the Sun, surface temperature of {temperature:.0f} Kelvin, luminosity {luminosity:.3f} times solar, and radius {radius:.2f} times solar radius",
            "Stellar parameters: M = {mass:.2f} M☉, T_eff = {temperature:.0f} K, L = {luminosity:.3f} L☉, R = {radius:.2f} R☉",
            "The star exhibits mass {mass:.2f} M_sun, effective temperature {temperature:.0f} degrees K, luminosity {luminosity:.3f} L_sun, and radius {radius:.2f} R_sun"
        ]

        # Randomly select template for variety
        template = np.random.choice(templates)

        # Fill template with parameters
        try:
            description = template.format(**star_params)
        except KeyError as e:
            # Handle missing parameters
            print(f"Warning: Missing parameter {e} for star {star_params}")
            # Use basic template
            description = f"Mass: {star_params.get('mass', 'unknown')}, Temperature: {star_params.get('temperature', 'unknown')}"

        return description

    def augment_data(self, latent_vectors: np.ndarray, descriptions: List[str],
                     augment_factor: int = 2) -> Tuple[np.ndarray, List[str]]:
        """
        Augment training data by adding noise to latent vectors and paraphrasing descriptions
        """
        augmented_latents = []
        augmented_descriptions = []

        # Original data
        augmented_latents.append(latent_vectors)
        augmented_descriptions.extend(descriptions)

        for i in range(augment_factor - 1):
            # Add noise to latent vectors
            noise_scale = 0.01 * (i + 1)  # Increasing noise
            noisy_latents = latent_vectors + np.random.normal(0, noise_scale, latent_vectors.shape)
            augmented_latents.append(noisy_latents)

            # Create paraphrased descriptions (simplified - you might want more sophisticated paraphrasing)
            paraphrased = []
            for desc in descriptions:
                # Simple paraphrasing by reordering parameters
                paraphrased_desc = self.paraphrase_description(desc)
                paraphrased.append(paraphrased_desc)

            augmented_descriptions.extend(paraphrased)

        final_latents = np.concatenate(augmented_latents, axis=0)

        return final_latents, augmented_descriptions

    def paraphrase_description(self, description: str) -> str:
        """Simple paraphrasing by reordering parameters"""
        # Extract parameters using regex
        mass_match = re.search(r'[Mm]ass[:\s]*([0-9.]+)', description)
        temp_match = re.search(r'[Tt]emperature[:\s]*([0-9.]+)', description)
        lum_match = re.search(r'[Ll]uminosity[:\s]*([0-9.]+)', description)
        rad_match = re.search(r'[Rr]adius[:\s]*([0-9.]+)', description)

        if all([mass_match, temp_match, lum_match, rad_match]):
            mass = float(mass_match.group(1))
            temp = float(temp_match.group(1))
            lum = float(lum_match.group(1))