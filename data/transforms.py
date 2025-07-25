import matplotlib.pyplot as plt
import scipy.interpolate as interpolate
from scipy.signal import medfilt
from scipy.optimize import curve_fit
import numpy as np
import torch
import time


class Compose:
    """Composes several transforms together.
    Adapted from https://pytorch.org/vision/master/_modules/torchvision/transforms/transforms.html#Compose

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x, mask=None, info=dict(), step=None):
        new_info = info.copy() if info else {}
        if len(x.shape) == 1:
                x = x[:, np.newaxis]
        out = x
        t0 = time.time()
        # print(f"Initial type: {out.dtype}")
        for t in self.transforms:
            out, mask, info = t(out, mask=mask, info=info)
            # print(f"{t}  shape: {out.shape}")
            # if mask is not None:
            #     print("mask shape: ", mask.shape)
            # else:
            #     print("mask is None")
        return out, mask, info

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string

class RandomMasking:
    """
    Randomly mask elements in the input for self-supervised learning tasks.
    Some masked elements are replaced with a predefined value, others with random numbers.
    """

    def __init__(self, mask_prob=0.15, replace_prob=0.8, mask_value=0,
                 random_low=0, random_high=None):
        """
        Initialize the RandomMasking transformation.

        :param mask_prob: Probability of masking an element
        :param replace_prob: Probability of replacing a masked element with mask_value
        :param random_prob: Probability of replacing a masked element with a random value
        :param mask_value: The value to use for masking
        :param random_low: Lower bound for random replacement (inclusive)
        :param random_high: Upper bound for random replacement (exclusive)
        """
        self.mask_prob = mask_prob
        self.replace_prob = replace_prob
        self.mask_value = mask_value
        self.random_low = random_low
        self.random_high = random_high

        assert 0 <= mask_prob <= 1, "mask_prob must be between 0 and 1"
        assert 0 <= replace_prob <= 1, "replace_prob must be between 0 and 1"

    def __call__(self, x, mask=None, info=dict()):
        if isinstance(x, np.ndarray):
            return self._mask_numpy(x, mask=mask, info=info)
        elif isinstance(x, torch.Tensor):
            return self._mask_torch(x, mask=mask, info=info)
        else:
            raise TypeError("Input must be either a numpy array or a PyTorch tensor")

    def _mask_numpy(self, x, mask=None, info=dict()):
        if len(x.shape) == 1:
            x = x[np.newaxis, :]
        if self.random_high is None:
            self.random_high = x.max()
        mask = np.random.rand(*x[0].shape) < self.mask_prob

        # Create a copy of x to modify
        masked_x = x.copy()

        # Replace with mask_value
        replace_mask = mask & (np.random.rand(*x[0].shape) < self.replace_prob)
        masked_x[:, replace_mask] = self.mask_value

        # Replace with random values
        random_mask = mask & ~replace_mask
        masked_x[:, random_mask] = np.random.uniform(self.random_low, self.random_high, size=random_mask.sum())

        return masked_x, mask, info

    def _mask_torch(self, x, mask=None, info=dict()):
        if self.random_high is None:
            self.random_high = x.max()
        if mask is None:
            mask = torch.rand_like(x) < self.mask_prob

        # Create a copy of x to modify
        masked_x = x.clone()

        # Replace with mask_value
        replace_mask = mask & (torch.rand_like(x) < self.replace_prob)
        masked_x[replace_mask] = self.mask_value

        # Replace with random values
        random_mask = mask & ~replace_mask
        masked_x[random_mask] = torch.rand_like(x[random_mask]) * (self.random_high - self.random_low) + self.random_low

        return masked_x, mask, info

    def __repr__(self):
        return (f"RandomMasking(mask_prob={self.mask_prob}, replace_prob={self.replace_prob}, "
                f"mask_value={self.mask_value}, "
                f"random_low={self.random_low}, random_high={self.random_high})")

class GeneralSpectrumPreprocessor:
    """
    Generalized preprocessing class for spectra implementing wavelength correction,
    resampling, denoising, continuum normalization, and secondary normalization
    across arbitrary wavelength sections.

    This class extends the LAMOST approach to handle any number of wavelength
    sections that are processed independently and then combined.
    """

    def __init__(self,
                 wavelength_sections=[(3841, 5800), (5800, 8798)],
                 resample_step=0.0001,
                 median_filter_size=3,
                 polynomial_order=5,
                 rv_norm=True,
                 continuum_norm=True,
                 resample=True,
                 plot_steps=False,
                 section_names=None):
        """
        Initialize preprocessing parameters.

        Args:
            wavelength_sections (list of tuples): List of (start, end) wavelength ranges
                Example: [(2000, 3500), (4000, 5000), (6000, 8000)]. default is LAMOST sections.
            resample_step (float): Logarithmic resampling step size
            median_filter_size (int): Size of median filter window
            polynomial_order (int): Order of polynomial for continuum estimation
            rv_norm (bool): Whether to apply radial velocity correction
            continuum_norm (bool): Whether to apply continuum normalization
            plot_steps (bool): Whether to generate diagnostic plots
            section_names (list): Optional names for each section for plotting
        """
        self.wavelength_sections = wavelength_sections
        self.resample_step = resample_step
        self.median_filter_size = median_filter_size
        self.polynomial_order = polynomial_order
        self.rv_norm = rv_norm
        self.resample = resample
        self.continuum_norm = continuum_norm
        self.plot_steps = plot_steps

        # Generate section names if not provided
        if section_names is None:
            self.section_names = [f"Section_{i + 1}_({start}-{end})"
                                  for i, (start, end) in enumerate(wavelength_sections)]
        else:
            self.section_names = section_names

        self.n_sections = len(wavelength_sections)

        # Validate inputs
        if len(self.section_names) != self.n_sections:
            raise ValueError("Number of section names must match number of wavelength sections")

    def __call__(self, spectrum, mask=None, info=dict()):
        """
        Apply full preprocessing pipeline to input spectrum.

        Args:
            spectrum (np.ndarray or torch.Tensor): Input spectrum flux values
            mask (np.ndarray): Optional mask for bad pixels
            info (dict): Dictionary containing 'wavelength' and optionally 'RV', 'obsid'

        Returns:
            tuple: (preprocessed_spectrum, mask, updated_info)
        """
        if len(spectrum.shape) == 2 and spectrum.shape[0] == 1:
            spectrum = spectrum.squeeze(0)
        # Convert to numpy if torch tensor
        if torch.is_tensor(spectrum):
            spectrum = spectrum.numpy()

        wavelength = info['wavelength']
        if len(wavelength.shape) == 2 and wavelength.shape[0] == 1:
            wavelength = wavelength.squeeze(0)

        # Initialize plotting if requested
        if self.plot_steps:
            fig, axes = self._setup_plots()
            self._plot_original_spectrum(axes[0], wavelength, spectrum, info.get('obsid', 'Unknown'))

        # 1. Wavelength Correction
        if self.rv_norm and 'RV' in info:
            corrected_wavelength = self._wavelength_correction(wavelength, info['RV'])
        else:
            corrected_wavelength = wavelength.copy()

        # info['corrected_wavelength'] = corrected_wavelength

        # 2. Process each wavelength section independently
        processed_sections = []
        section_info = {}

        for i, (section_range, section_name) in enumerate(zip(self.wavelength_sections, self.section_names)):
            # Extract section
            section_mask = ((corrected_wavelength >= section_range[0]) &
                            (corrected_wavelength <= section_range[1]))

            if not np.any(section_mask):
                print(f"Warning: No data found in section {section_name} ({section_range})")
                continue

            section_wavelength = corrected_wavelength[section_mask]
            section_spectrum = spectrum[section_mask].squeeze()

            # Store section info
            # section_info[f'{section_name}_wavelength'] = section_wavelength
            # section_info[f'{section_name}_original'] = section_spectrum.copy()

            if self.plot_steps:
                self._plot_section_step(axes[1], i, section_wavelength, section_spectrum,
                                        section_name, "After Wavelength Correction")

            # Process this section through the pipeline
            processed_section = self._process_section(
                section_spectrum, section_wavelength, section_range,
                section_name, i, axes if self.plot_steps else None
            )

            processed_sections.append(processed_section)
            # section_info[f'{section_name}_processed'] = processed_section

        # 3. Combine all processed sections
        if processed_sections:
            combined_spectrum = np.concatenate(processed_sections)
            info.update(section_info)

            # 4. Secondary Normalization
            combined_spectrum = self._secondary_normalization(combined_spectrum)

        else:
            raise ValueError("No valid wavelength sections found in the spectrum")

        if self.plot_steps:
            obsid = info['apogee_id'] if 'apogee_id' in info else info.get('obsid', 'Unknown')
            self._finalize_plots(fig, obsid)

        return combined_spectrum[None, :], mask, info

    def _process_section(self, section_spectrum, section_wavelength, section_range,
                         section_name, section_idx, axes=None):
        """
        Process a single wavelength section through the full pipeline.
        """
        current_spectrum = section_spectrum.copy()
        # 1. Linear Interpolation Resampling
        if self.resample:
            current_spectrum = self._linear_interpolation_resample(
                current_spectrum, section_wavelength, section_range
            )

        if axes is not None:
            self._plot_section_step(axes[2], section_idx, np.arange(len(current_spectrum)),
                                    current_spectrum, section_name, "After Resampling")

        # 2. Denoising (Median Filtering)
        current_spectrum = self._median_filter_denoise(current_spectrum)

        if axes is not None:
            self._plot_section_step(axes[3], section_idx, np.arange(len(current_spectrum)),
                                    current_spectrum, section_name, "After Denoising")

        # 3. Continuum Normalization
        if self.continuum_norm:
            current_spectrum = self._continuum_normalization(current_spectrum)

            if axes is not None:
                self._plot_section_step(axes[4], section_idx, np.arange(len(current_spectrum)),
                                        current_spectrum, section_name, "After Continuum Normalization")

        return current_spectrum

    def _wavelength_correction(self, wavelength, radial_velocity):
        """
        Correct wavelength based on radial velocity.
        λ′ = λ * (1 + RV/c)
        """
        c = 299792.458  # Speed of light in km/s
        return wavelength * (1 + radial_velocity / c)

    def _linear_interpolation_resample(self, spectrum, wavelength, wave_range):
        """
        Resample spectrum using linear interpolation in logarithmic space.
        """
        # Create logarithmic wavelength grid
        log_wave_start = np.log10(wave_range[0])
        log_wave_end = np.log10(wave_range[1])
        new_log_wavelengths = np.arange(
            log_wave_start,
            log_wave_end,
            self.resample_step
        )
        new_wavelengths = 10 ** new_log_wavelengths

        # Interpolate spectrum
        interpolator = interpolate.interp1d(
            wavelength,
            spectrum,
            kind='linear',
            fill_value='extrapolate'
        )
        resampled_spectrum = interpolator(new_wavelengths)

        return resampled_spectrum

    def _median_filter_denoise(self, spectrum):
        """
        Apply median filtering for noise reduction.
        """
        return medfilt(spectrum, kernel_size=self.median_filter_size)

    def _continuum_normalization(self, spectrum):
        """
        Estimate and normalize continuum using polynomial fitting.
        """
        x = np.arange(len(spectrum))
        poly_coeffs = np.polyfit(x, spectrum, deg=self.polynomial_order)
        continuum = np.polyval(poly_coeffs, x)

        # Avoid division by zero
        continuum = np.where(continuum == 0, 1e-10, continuum)
        normalized_spectrum = spectrum / continuum

        return normalized_spectrum

    def _secondary_normalization(self, spectrum):
        """
        Secondary normalization with outlier replacement and z-score transformation.
        """
        mu = np.mean(spectrum)
        sigma = np.std(spectrum)

        # Handle case where sigma is zero
        if sigma == 0:
            return np.zeros_like(spectrum)

        # Replace outliers
        spectrum = np.where(
            (spectrum < mu - 3 * sigma) | (spectrum > mu + 3 * sigma),
            mu,
            spectrum
        )

        # Z-score transformation
        normalized_spectrum = (spectrum - mu) / sigma

        return normalized_spectrum

    def _setup_plots(self):
        """
        Setup the plotting grid for diagnostic plots.
        """
        n_rows = 6
        n_cols = max(2, self.n_sections)  # At least 2 columns, more if needed

        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(15 * n_cols, 4 * n_rows))

        # Ensure axes is always 2D
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        return fig, axes

    def _plot_original_spectrum(self, ax_row, wavelength, spectrum, obsid):
        """
        Plot the original spectrum across the full row.
        """
        # Use the first subplot of the row for the full spectrum
        ax_row[0].plot(wavelength.squeeze(), spectrum.squeeze())
        ax_row[0].set_title(f"Original Spectrum - {obsid}")
        ax_row[0].set_xlabel("Wavelength (Å)")
        ax_row[0].set_ylabel("Flux")

        # Hide other subplots in the first row if not needed
        for i in range(1, len(ax_row)):
            ax_row[i].set_visible(False)

    def _plot_section_step(self, ax_row, section_idx, x_data, y_data, section_name, step_name):
        """
        Plot a processing step for a specific section.
        """
        if section_idx < len(ax_row):
            ax_row[section_idx].plot(x_data, y_data, label=section_name)
            ax_row[section_idx].set_title(f"{step_name} - {section_name}")
            ax_row[section_idx].set_xlabel("Index" if step_name != "After Wavelength Correction" else "Wavelength (Å)")
            ax_row[section_idx].set_ylabel("Flux")

    def _finalize_plots(self, fig, obsid):
        """
        Finalize and save the diagnostic plots.
        """
        fig.suptitle(f"General Spectrum Preprocessing - {obsid}", fontsize=16)
        plt.tight_layout()

        # Try to save the plot, but don't fail if the directory doesn't exist
        try:
            plt.savefig(f'/data/lightSpec/images/general_spectrum_{obsid}_preprocessing.png', dpi=150,
                        bbox_inches='tight')
        except:
            print(f"Could not save plot for {obsid}")

        plt.show()

    def get_section_bounds(self):
        """
        Get the wavelength bounds for each section after resampling.

        Returns:
            list: List of tuples with (start_idx, end_idx) for each section in combined spectrum
        """
        bounds = []
        current_idx = 0

        for section_range in self.wavelength_sections:
            # Calculate expected length after resampling
            log_wave_start = np.log10(section_range[0])
            log_wave_end = np.log10(section_range[1])
            n_points = int((log_wave_end - log_wave_start) / self.resample_step)

            bounds.append((current_idx, current_idx + n_points))
            current_idx += n_points

        return bounds

    def __repr__(self):
        return (f"GeneralSpectrumPreprocessor("
                f"wavelength_sections={self.wavelength_sections}, "
                f"n_sections={self.n_sections}, "
                f"resample_step={self.resample_step})")


class ToTensor():
    def __init__(self):
        pass
    def __call__(self, x, mask=None, info=None, step=None):
        x = torch.tensor(x)
        if mask is not None:
           mask = torch.tensor(mask)
        return x, mask, info
    def __repr__(self):
        return "ToTensor"