import os
import torch
from torch.utils.data import Dataset
from astropy.io import fits
import numpy as np
from pathlib import Path
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
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
                 file_type='fits',
                 wv_arr=None
                 ):
        self.data_dir = Path(data_dir)
        self.transforms = transforms
        self.df = df
        self.id = id
        self.labels = labels
        self.file_type = file_type
        self.wv_arr = wv_arr
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
        if self.file_type == 'fits':
            with fits.open(filename) as hdulist:
                binaryext = hdulist[1].data
                header = hdulist[0].header
            x = binaryext['FLUX'].astype(np.float32)
            wv = binaryext['WAVELENGTH'].astype(np.float32)
            rv = header['HELIO_RV']
            meta = {'RV': rv, 'wavelength': wv}
        elif self.file_type == 'pqt':
            data = pd.read_parquet(filename)
            x = data['flux'].astype(np.float32).values
            wv = data['wavelength'].astype(np.float32).values
            meta = {'wavelength': wv}
        else:
            raise NotImplementedError(f'File type {self.file_type} not supported')
        return x, meta

    def read_apogee_spectra(self, filename):
        if self.file_type == 'fits':
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
        elif self.file_type == 'pqt':
            data = pd.read_parquet(filename)
            data = data['flux'].astype(np.float32).values
            meta = {'wavelength': data['wavelength'].astype(np.float32)}.values
        else:
            raise NotImplementedError(f'File type {self.file_type} not supported')
        return data, meta

    def __getitem__(self, idx):
        start = time.time()

        try:
            row = self.df.iloc[idx]
            obsid = row[self.id]
            info = row.to_dict()
            if self.id == 'obsid' or self.id == 'Simulation Number': # simulations are of LAMOST
                spectra_filename = os.path.join(self.data_dir, f'{obsid}.{self.file_type}')
                spectra, meta = self.read_lamost_spectra(spectra_filename)
                info.update(meta)
                info['obsid'] = obsid
            elif self.id == 'APOGEE_ID':
                spectra_filename = f"/data/apogee/data/aspcapStar-dr17-{obsid}.{self.file_type}"
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
