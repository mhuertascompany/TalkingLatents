import os
import torch
from torch.utils.data import Dataset
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib as mpl
import pandas as pd
from typing import List
import copy
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import traceback
from data.transforms import RandomMasking
import time


def continue_array_linear(input_array, seq_len, scale='linear'):
    """
    Continue a 1D array to a desired sequence length using linear trend extrapolation.

    Parameters:
    -----------
    input_array : array-like
        The input 1D array to continue
    seq_len : int
        Desired total length of the output sequence
    scale : str, default 'linear'
        Scaling method:
        - 'linear': Linear trend in original space (constant additive steps)
        - 'log': Linear trend in log space (constant multiplicative steps)

    Returns:
    --------
    numpy.ndarray
        Extended array of length seq_len using linear trend
    """
    arr = np.array(input_array, dtype=float)
    n = len(arr)
    # Number of points to extrapolate
    extend_len = seq_len - n

    if scale == 'linear':
        # Linear trend in original space: y = mx + b
        # Constant additive steps
        if n >= 2:
            x_orig = np.arange(n)
            slope = np.polyfit(x_orig, arr, 1)[0]  # Linear fit slope
            extended_values = arr[-1] + slope * np.arange(1, extend_len + 1)
        else:
            extended_values = np.full(extend_len, arr[-1])

    elif scale == 'log':
        # Linear trend in log space: log(y) = mx + b
        # Constant multiplicative steps (exponential growth/decay)
        if n >= 2 and np.all(arr > 0):
            x_orig = np.arange(n)
            log_arr = np.log(arr)
            slope = np.polyfit(x_orig, log_arr, 1)[0]  # Linear fit in log space

            # Extend in log space then convert back
            log_extended = np.log(arr[-1]) + slope * np.arange(1, extend_len + 1)
            extended_values = np.exp(log_extended)
        else:
            # Fallback: can't use log scale with non-positive values
            print("Warning: Log scale requires positive values. Using linear scale instead.")
            if n >= 2:
                x_orig = np.arange(n)
                slope = np.polyfit(x_orig, arr, 1)[0]
                extended_values = arr[-1] + slope * np.arange(1, extend_len + 1)
            else:
                extended_values = np.full(extend_len, arr[-1])

    return np.concatenate([arr, extended_values])
class SimulationDataset(Dataset):
    def __init__(self,
                 df,
                 labels=['Period', 'Inclination'],
                 light_transforms=None,
                 spec_transforms=None,
                 npy_path=None,
                 spec_path=None,
                 use_acf=False,
                 use_fft=False,
                 scale_flux=False,
                 meta_columns=None,
                 spec_seq_len=4096,
                 light_seq_len=34560,
                 example_wv_path='/data/lamost/example_wv.npy'
                 ):
        self.df = df
        self.spec_seq_len = spec_seq_len
        self.lc_seq_len = light_seq_len
        self.use_acf = use_acf
        self.use_fft = use_fft
        self.range_dict = dict()
        self.labels = labels
        # self.labels_df = self.df[labels]
        self.update_range_dict()
        # self.lc_dir = os.path.join(data_dir, 'lc')
        # self.spectra_dir = os.path.join(data_dir, 'lamost')
        self.lc_transforms = light_transforms
        self.spectra_transforms = spec_transforms
        self.example_wv = np.load(example_wv_path)
        self.Nlc = len(self.df)
        self.scale_flux = scale_flux
        self.meta_columns = meta_columns

    def fill_nan_inf_np(self, x: np.ndarray, interpolate: bool = True):
        """
        Fill NaN and Inf values in a numpy array

        Args:
            x (np.ndarray): array to fill
            interpolate (bool): whether to interpolate or not

        Returns:
            np.ndarray: filled array
        """
        # Create a copy to avoid modifying the original array
        x_filled = x.copy()

        # Identify indices of finite and non-finite values
        finite_mask = np.isfinite(x_filled)
        non_finite_indices = np.where(~finite_mask)[0]

        finite_indices = np.where(finite_mask)[0]

        # If there are non-finite values and some finite values
        if len(non_finite_indices) > 0 and len(finite_indices) > 0:
            if interpolate:
                # Interpolate non-finite values using linear interpolation
                interpolated_values = np.interp(
                    non_finite_indices,
                    finite_indices,
                    x_filled[finite_mask]
                )
                # Replace non-finite values with interpolated values
                x_filled[non_finite_indices] = interpolated_values
            else:
                # Replace non-finite values with zero
                x_filled[non_finite_indices] = 0

        return x_filled

    def update_range_dict(self):
        for name in self.labels:
            min_val = self.df[name].min()
            max_val = self.df[name].max()
            self.range_dict[name] = (min_val, max_val)

    def __len__(self):
        return len(self.df)

    def _normalize(self, x, label):
        # min_val = float(self.range_dict[key][0])
        # max_val = float(self.range_dict[key][1])
        # return (x - min_val) / (max_val - min_val)
        if 'period' in label.lower():
            x = x / P_MAX
        elif 'age' in label.lower():
            x = x / MAX_AGE
        return x

        # min_val, max_val = self.boundary_values_dict[f'min {label}'], self.boundary_values_dict[f'max {label}']
        # return (x - min_val)/(max_val - min_val)

    def transform_lc_flux(self, flux, info_lc):
        if self.lc_transforms:
            flux, _, info_lc = self.lc_transforms(flux, info=info_lc)
            if self.use_acf:
                acf = torch.tensor(info_lc['acf']).nan_to_num(0)
                flux = torch.cat((flux, acf), dim=0)
            if self.use_fft:
                fft = torch.tensor(info_lc['fft']).nan_to_num(0)
                flux = torch.cat((flux, fft), dim=0)
        if flux.shape[-1] == 1:
            flux = flux.squeeze(-1)
        if len(flux.shape) == 1:
            flux = flux.unsqueeze(0)
        flux = pad_with_last_element(flux, self.lc_seq_len)
        return flux, info_lc

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        padded_idx = f'{idx:d}'.zfill(int(np.log10(self.Nlc)) + 1)
        s = time.time()
        try:
            spec = pd.read_parquet(row['spec_data_path']).values
            # spec = self.fill_nan_np(spec)
        except (FileNotFoundError, OSError, IndexError) as e:
            # print("Error reading file ", idx, e)
            spec = np.zeros((3909, 1))
        try:
            # print(idx, row['lc_data_path'])
            lc = pd.read_parquet(row['lc_data_path']).values
            # lc[:, 1] = self.fill_nan_inf_np(lc[:, 1])
            # max_val = np.max(np.abs(lc[:, 1]))
            # if max_val > 1e2:
            #     lc[np.abs(lc[:, 1]) > 1e2, 1] = np.random.uniform(0, 2, size=lc[np.abs(lc[:, 1]) > 1e2, 1].shape)

            # lc[:, 1] = lc[:, 1] / lc[:, 1].max()
        except (FileNotFoundError, OSError, IndexError) as e:
            print("Error reading file ", idx, e)
            lc = np.zeros((48000, 2))
        spectra = spec[:, -1]
        target_spectra = spectra.copy()
        flux = lc[:, -1]
        target_flux = flux.copy()
        info_s = dict()
        info_lc = dict()
        try:
            # label = row[self.labels].to_dict()
            # print(label)
            # label = {k: self._normalize(v, k) for k, v in label.items()}
            y = torch.tensor([self._normalize(row[k], k) for k in self.labels], dtype=torch.float32)

        except IndexError:
            y = torch.zeros(len(self.labels))
        s1 = time.time()
        info_s['wavelength'] = self.example_wv
        if self.spectra_transforms:
            spectra, _, info_s = self.spectra_transforms(spectra, info=info_s)
        spectra = pad_with_last_element(spectra, self.spec_seq_len)
        spectra = torch.nan_to_num(spectra, nan=0)
        s2 = time.time()
        # print("nans in spectra: ", np.sum(np.isnan(spectra)))
        # spectra = torch.tensor(spectra).float()
        flux, info_lc = self.transform_lc_flux(flux, info_lc)
        target_flux, _ = self.transform_lc_flux(target_flux, info_lc)
        info = {'spectra': info_s, 'lc': info_lc}
        if 'L' in row.keys():
            info['KMAG'] = row['L']
        else:
            info['KMAG'] = 1
        s3 = time.time()
        flux = flux.nan_to_num(0).float()
        spectra = spectra.nan_to_num(0).float()
        target_flux = target_flux.nan_to_num(0).float()
        # print(flux.shape, target_flux.shape, spectra.shape, y.shape)
        # print(s1-s, s2-s1, s3-s2)
        return flux, spectra, y, target_flux, spectra, info


class SpectraDataset(Dataset):
    """
    dataset for spectra data
    Args:
        data_dir: path to the data directory
        transforms: transformations to apply to the data
        df: dataframe containing the data paths
        max_len: maximum length of the spectra
        use_cache: whether to use a cache file
        id: column name for the observation id
    """

    def __init__(self, data_dir,
                 transforms=None,
                 df=None,
                 max_len=3909,
                 use_cache=True,
                 id='obsid',
                 labels=['Teff', 'logg', 'FeH'],
                 ):
        self.data_dir = Path(data_dir)
        self.transforms = transforms
        self.df = df
        self.id = id
        self.labels = labels
        if df is None:
            cache_file = os.path.join(self.data_dir, '.path_cache.txt')

            if use_cache and os.path.exists(cache_file):
                print("Loading cached file paths...")
                with open(cache_file, 'r') as f:
                    self.path_list = np.array([line.strip() for line in f])
            else:
                print("Creating files list...")
                self.path_list = self._file_listing()
                if use_cache:
                    with open(cache_file, 'w') as f:
                        f.write('\n'.join(self.path_list))
        else:
            self.path_list = None
        self.max_len = max_len
        self.mask_transform = RandomMasking()

    def _file_listing(self):

        def process_chunk(file_names):
            return [self.data_dir / name for name in file_names]

        file_names = os.listdir(self.data_dir)
        chunk_size = 100000
        chunks = [file_names[i:i + chunk_size] for i in range(0, len(file_names), chunk_size)]

        with ThreadPoolExecutor() as executor:
            paths = []
            for chunk_paths in executor.map(process_chunk, chunks):
                paths.extend(chunk_paths)

        return np.array(paths)

    def __len__(self):
        if self.df is not None:
            return len(self.df)
        return len(self.path_list) if self.path_list is not None else 0

    def read_lamost_spectra(self, filename):
        with fits.open(filename) as hdulist:
            binaryext = hdulist[1].data
            header = hdulist[0].header
        x = binaryext['FLUX'].astype(np.float32)
        wv = binaryext['WAVELENGTH'].astype(np.float32)
        rv = header['HELIO_RV']
        meta = {'RV': rv, 'wavelength': wv}
        return x, meta

    def read_apogee_spectra(self, filename):
        with fits.open(filename) as hdul:
            data = hdul[1].data.astype(np.float32).squeeze()[None]
        meta = {}
        header = hdul[1].header
        # Create pixel array (1-indexed for FITS convention)
        pixels = np.arange(1, data.shape[-1] + 1)

        # Calculate log10(wavelength):
        # log_wave = CRVAL1 + CDELT1 * (pixel - CRPIX1)
        log_wavelength = header['CRVAL1'] + header['CDELT1'] * (pixels - header['CRPIX1'])

        # Convert to linear wavelength in Angstroms
        wv = 10 ** log_wavelength
        meta = {'wavelength': wv}
        return data, meta

    def __getitem__(self, idx):
        start = time.time()

        try:
            row = self.df.iloc[idx]
            obsid = row[self.id]
            info = row.to_dict()
            if self.id == 'obsid':
                spectra_filename = os.path.join(self.data_dir, f'{obsid}.fits')
                spectra, meta = self.read_lamost_spectra(spectra_filename)
                info.update(meta)
                info['obsid'] = obsid
            elif self.id == 'APOGEE_ID':
                spectra_filename = f"/data/apogee/data/aspcapStar-dr17-{obsid}.fits"
                spectra, meta = self.read_apogee_spectra(spectra_filename)
                info.update(meta)
                info['apogee_id'] = obsid
            else:
                raise ValueError(f"Unknown id type: {self.id}")
        except (OSError, IndexError) as e:
            print("Error reading file ", obsid, e)
            return (torch.zeros(self.max_len),
                    torch.zeros(self.max_len),
                    torch.zeros(len(self.labels)),
                    torch.zeros(self.max_len, dtype=torch.bool),
                    torch.zeros(self.max_len, dtype=torch.bool),
                    info
                    )

        t1 = time.time()
        if self.transforms:
            spectra, _, info = self.transforms(spectra, None, info)
        spectra_masked, mask, _ = self.mask_transform(spectra, None, info)
        if spectra_masked.shape[-1] < self.max_len:
            pad = torch.zeros(1, self.max_len - spectra_masked.shape[-1])
            spectra_masked = torch.cat([spectra_masked, pad], dim=-1)
            pad_mask = torch.zeros(1, self.max_len - mask.shape[-1], dtype=torch.bool)
            mask = torch.cat([mask, pad_mask], dim=-1)
            pad_spectra = torch.zeros(1, self.max_len - spectra.shape[-1])
            spectra = torch.cat([spectra, pad_spectra], dim=-1)
            wv = info['wavelength']
            wv = continue_array_linear(wv.squeeze(), self.max_len - wv.squeeze().shape[-1], scale='log')
            info['wavelength'] = wv
        spectra = torch.nan_to_num(spectra, nan=0)
        spectra_masked = torch.nan_to_num(spectra_masked, nan=0)
        t4 = time.time()
        target = torch.tensor([info[t] for t in self.labels], dtype=torch.float32)
        # print("read spectra: ", t1-start, "transform: ", t2-t1, "target: ", t3-t2, "pad: ", t4-t3)
        return (spectra_masked.float().squeeze(0), spectra.float().squeeze(0), \
                target.float(), mask.squeeze(0), mask.squeeze(0), info)
