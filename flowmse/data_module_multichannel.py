import os
from os.path import join
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from glob import glob
import numpy as np
import torch.nn.functional as F
import torchaudio

def get_window(window_type, window_length):
    if window_type == 'sqrthann':
        return torch.sqrt(torch.hann_window(window_length, periodic=True))
    elif window_type == 'hann':
        return torch.hann_window(window_length, periodic=True)
    else:
        raise NotImplementedError(f"Window type {window_type} not implemented!")


class Specs(Dataset):
    def __init__(self, data_dir, dummy, shuffle_spec, num_frames, sampling_rate=16000,
            format='default', normalize="noisy", spec_transform=None,
            stft_kwargs=None, subset_ratio=1.0, mix_dir=None, **ignored_kwargs):
        """
        mix_dir: (Optional) Separate root path to find Mixture files. 
                 If provided, mixture files are found here, while clean/noisy files are found in data_dir.
        """

        self.mixture_files = []
        self.noisy_files = []  # Separator Output (_p)
        self.clean_files = []  # Ground Truth (Clean)

        if format == "default":
            # (Keep original code)
            noisy_files1 = sorted(glob(os.path.join(data_dir, '*_source1hatP.wav')))
            clean_files1 = [item.replace('_source1hatP.wav', '_source1.wav') for item in noisy_files1]
            mixture_files1 = [item.replace('_source1hatP.wav', '_mix.wav') for item in noisy_files1]
            noisy_files2 = sorted(glob(os.path.join(data_dir, '*_source2hatP.wav')))
            clean_files2 = [item.replace('_source2hatP.wav', '_source2.wav') for item in noisy_files2]
            mixture_files2 = [item.replace('_source2hatP.wav', '_mix.wav') for item in noisy_files2]
            
            self.mixture_files = [*mixture_files1, *mixture_files2]
            self.noisy_files = [*noisy_files1, *noisy_files2]
            self.clean_files = [*clean_files1, *clean_files2]

        elif format == "wsj_hierarchical":
            # Target folder list
            target_spks = ['spk_1', 'spk_2', 'spk_3']
            
            # Dictionary mapping spk_N folder names to Nspeaker_reverb folder names
            spk_to_mix_folder = {
                'spk_1': '1speaker_reverb',
                'spk_2': '2speaker_reverb',
                'spk_3': '3speaker_reverb'
            }
            
            for spk_name in target_spks:
                spk_path = os.path.join(data_dir, spk_name)
                if not os.path.exists(spk_path):
                    continue
                
                # Extract the number of speakers from the folder name
                try:
                    num_speakers = int(spk_name.split('_')[-1])
                except ValueError:
                    print(f"Warning: Could not parse speaker count from {spk_name}. Skipping.")
                    continue

                # List case folders (e.g., 01aa0108_m0p2374)
                case_dirs = sorted([d for d in os.listdir(spk_path) if os.path.isdir(os.path.join(spk_path, d))])
                
                # Apply Subset Ratio
                if subset_ratio < 1.0:
                    num_use = int(len(case_dirs) * subset_ratio)
                    case_dirs = case_dirs[:num_use]
                
                for case_dir in case_dirs:
                    full_case_path = os.path.join(spk_path, case_dir)
                    
                    # 1. Extract prefix (e.g., 95) from the existing folder to find Clean/Est files.
                    #    Derive the prefix based on the _mix.wav file in the existing folder.
                    mix_candidates_original = glob(os.path.join(full_case_path, '*_mix.wav'))
                    if not mix_candidates_original:
                        continue
                    
                    # e.g.: 457_mix.wav -> 457
                    original_mix_filename = os.path.basename(mix_candidates_original[0])
                    file_prefix = original_mix_filename.replace('_mix.wav', '')
                    
                    # 2. Set the actual Mixture path to be used
                    if mix_dir is not None:
                        # If mix_dir is provided: find mix.wav in a different path
                        # Structure: mix_dir / 1speaker_reverb / case_dir / mix.wav
                        mix_folder_name = spk_to_mix_folder.get(spk_name)
                        if mix_folder_name is None:
                             print(f"Warning: No mapping for {spk_name}. Skipping mixture path override.")
                             continue
                             
                        # e.g.: .../wav16k_/cv/1speaker_reverb/01aa0108_m0p2374/mix.wav
                        final_mix_path = os.path.join(mix_dir, mix_folder_name, case_dir, 'mix.wav')
                    else:
                        # Default behavior (original way)
                        final_mix_path = mix_candidates_original[0]

                    # Check if the Mix file actually exists (Required)
                    if not os.path.exists(final_mix_path):
                        # print(f"Missing mix file: {final_mix_path}") # For debugging
                        continue

                    # Iterate for the number of speakers (spk1, spk2, spk3 ...)
                    for k in range(1, num_speakers + 1):
                        # Clean/Est filename pattern (using existing folder)
                        est_filename = f"{file_prefix}_spk{k}_p.wav"
                        clean_filename = f"{file_prefix}_spk{k}.wav"
                        
                        est_file_path = os.path.join(full_case_path, est_filename)
                        clean_file_path = os.path.join(full_case_path, clean_filename)
                        
                        # Add only if both files in the pair exist
                        if os.path.exists(est_file_path) and os.path.exists(clean_file_path):
                            self.mixture_files.append(final_mix_path)  # Replaced path or original path
                            self.noisy_files.append(est_file_path)     # Original path
                            self.clean_files.append(clean_file_path)   # Original path

        else:
            raise NotImplementedError(f"Directory format {format} unknown!")

        self.dummy = dummy
        self.num_frames = num_frames
        self.shuffle_spec = shuffle_spec
        self.normalize = normalize
        self.spec_transform = spec_transform
        self.sampling_rate = sampling_rate

        assert all(k in stft_kwargs.keys() for k in ["n_fft", "hop_length", "center", "window"]), "misconfigured STFT kwargs"
        self.stft_kwargs = stft_kwargs
        self.hop_length = self.stft_kwargs["hop_length"]
        assert self.stft_kwargs.get("center", None) == True, "'center' must be True for current implementation"

    def __getitem__(self, i):
        # Clean Load
        x, sr = torchaudio.load(self.clean_files[i])
        x = x.float()
        if sr != self.sampling_rate:
            x = torchaudio.transforms.Resample(sr, self.sampling_rate)(x)
            
        # Noisy (Est) Load
        y, sr = torchaudio.load(self.noisy_files[i])
        y = y.float()
        if sr != self.sampling_rate:
            y = torchaudio.transforms.Resample(sr, self.sampling_rate)(y)
            
        # Mixture Load (might be an overridden path)
        m, sr = torchaudio.load(self.mixture_files[i])
        m = m.float()
        if sr != self.sampling_rate:
            m = torchaudio.transforms.Resample(sr, self.sampling_rate)(m)
            
        min_leng = min(x.shape[-1], y.shape[-1], m.shape[-1])
        x = x[...,:min_leng]
        y = y[...,:min_leng]
        m = m[...,:min_leng]

        target_len = (self.num_frames - 1) * self.hop_length
        current_len = x.size(-1)
        pad = max(target_len - current_len, 0)
        if pad == 0:
            if self.shuffle_spec:
                start = int(np.random.uniform(0, current_len-target_len))
            else:
                start = int((current_len-target_len)/2)
                
            if y[..., start:start+target_len].abs().max() < 0.05:
                start = 0
                
            x = x[..., start:start+target_len]
            y = y[..., start:start+target_len]
            m = m[..., start:start+target_len]
        else:
            x = F.pad(x, (pad//2, pad//2+(pad%2)), mode='constant')
            y = F.pad(y, (pad//2, pad//2+(pad%2)), mode='constant')
            m = F.pad(m, (pad//2, pad//2+(pad%2)), mode='constant')

        if self.normalize == "noisy":
            normfac = y.abs().max()
        elif self.normalize == "clean":
            normfac = x.abs().max()
        elif self.normalize == "not":
            normfac = 1.0
        x = x / normfac
        y = y / normfac
        m = m / normfac
        X = torch.stft(x, **self.stft_kwargs)
        Y = torch.stft(y, **self.stft_kwargs)
        M = torch.stft(m, **self.stft_kwargs)
        X, Y, M = self.spec_transform(X), self.spec_transform(Y), self.spec_transform(M)      
        return X, Y, M

    def __len__(self):
        if self.dummy:
            return int(len(self.clean_files)/200)
        else:
            return len(self.clean_files)


class SpecsDataModule(pl.LightningDataModule):
    @staticmethod
    def add_argparse_args(parser):
        # Original paths (for Clean/Est files and metadata)

        parser.add_argument("--train_dir", type=str, default='')
        parser.add_argument("--val_dir", type=str, default='') 
        parser.add_argument("--test_dir", type=str, default='')



        parser.add_argument("--train_mix_dir", type=str, default='')
        parser.add_argument("--val_mix_dir", type=str, default='') 
        parser.add_argument("--test_mix_dir", type=str, default='')

        parser.add_argument("--format", type=str, default="wsj_hierarchical", help="Read file paths according to file naming format.")
        parser.add_argument("--sampling_rate", type=int, default=16000, help="The sampling rate.")
        parser.add_argument("--batch_size", type=int, default=16, help="The batch size.")
        parser.add_argument("--n_fft", type=int, default=510, help="Number of FFT bins.")
        parser.add_argument("--hop_length", type=int, default=64, help="Window hop length.")
        parser.add_argument("--num_frames", type=int, default=256, help="Number of frames for the dataset.")
        parser.add_argument("--window", type=str, choices=("sqrthann", "hann"), default="hann", help="The window function.")
        parser.add_argument("--num_workers", type=int, default=8, help="Number of workers.")
        parser.add_argument("--dummy", action="store_true", help="Use reduced dummy dataset.")
        parser.add_argument("--spec_factor", type=float, default=0.15, help="Factor to multiply complex STFT.")
        parser.add_argument("--spec_abs_exponent", type=float, default=0.5, help="Exponent e for the transformation.")
        parser.add_argument("--normalize", type=str, choices=("clean", "noisy", "not"), default="noisy", help="Normalize strategy.")
        parser.add_argument("--transform_type", type=str, choices=("exponent", "log", "none"), default="exponent", help="Spectogram transformation.")
        
        parser.add_argument("--train_subset_ratio", type=float, default=0.3, help="Ratio of data to use for training (e.g. 0.5 for 50%).")
        
        return parser

    def __init__(
        self, train_dir, val_dir, test_dir, 
        train_mix_dir, val_mix_dir, test_mix_dir,
        format='wsj_hierarchical', sampling_rate=16000, batch_size=8,
        n_fft=510, hop_length=64, num_frames=256, window='hann',
        num_workers=4, dummy=False, spec_factor=0.15, spec_abs_exponent=0.5,
        gpu=True, normalize='noisy', transform_type="exponent", train_subset_ratio=0.5, **kwargs
    ):
        super().__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        
        # Save Mixture paths
        self.train_mix_dir = train_mix_dir
        self.val_mix_dir = val_mix_dir
        self.test_mix_dir = test_mix_dir
        
        self.format = format
        self.sampling_rate = sampling_rate
        self.batch_size = batch_size
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_frames = num_frames
        self.window = get_window(window, self.n_fft)
        self.windows = {}
        self.num_workers = num_workers
        self.dummy = dummy
        self.spec_factor = spec_factor
        self.spec_abs_exponent = spec_abs_exponent
        self.gpu = gpu
        self.normalize = normalize
        self.transform_type = transform_type
        self.train_subset_ratio = train_subset_ratio
        self.kwargs = kwargs
        print(self.train_subset_ratio)

    def setup(self, stage=None):
        specs_kwargs = dict(
            stft_kwargs=self.stft_kwargs, num_frames=self.num_frames,
            spec_transform=self.spec_fwd, **self.kwargs
        )
        if stage == 'fit' or stage is None:
            self.train_set = Specs(data_dir=self.train_dir,
                dummy=self.dummy, shuffle_spec=True, format=self.format, 
                normalize=self.normalize, sampling_rate=self.sampling_rate, 
                subset_ratio=self.train_subset_ratio, 
                mix_dir=self.train_mix_dir,  # [NEW]
                **specs_kwargs)
            
            self.valid_set = Specs(data_dir=self.val_dir,
                dummy=self.dummy, shuffle_spec=False, format=self.format,
                normalize=self.normalize, sampling_rate=self.sampling_rate, 
                subset_ratio=1.0, 
                mix_dir=self.val_mix_dir,  # [NEW]
                **specs_kwargs)

        if stage == 'test' or stage is None:
            self.test_set = Specs(data_dir=self.test_dir,
                dummy=self.dummy, shuffle_spec=False, format=self.format,
                normalize=self.normalize, sampling_rate=self.sampling_rate, 
                subset_ratio=1.0, 
                mix_dir=self.test_mix_dir,  # [NEW]
                **specs_kwargs)

    # ... (The rest remains the same) ...
    def spec_fwd(self, spec):
        if self.transform_type == "exponent":
            if self.spec_abs_exponent != 1:
                e = self.spec_abs_exponent
                spec = spec.abs()**e * torch.exp(1j * spec.angle())
            spec = spec * self.spec_factor
        elif self.transform_type == "log":
            spec = torch.log(1 + spec.abs()) * torch.exp(1j * spec.angle())
            spec = spec * self.spec_factor
        elif self.transform_type == "none":
            spec = spec
        return spec

    def spec_back(self, spec):
        if self.transform_type == "exponent":
            spec = spec / self.spec_factor
            if self.spec_abs_exponent != 1:
                e = self.spec_abs_exponent
                spec = spec.abs()**(1/e) * torch.exp(1j * spec.angle())
        elif self.transform_type == "log":
            spec = spec / self.spec_factor
            spec = (torch.exp(spec.abs()) - 1) * torch.exp(1j * spec.angle())
        elif self.transform_type == "none":
            spec = spec
        return spec

    @property
    def stft_kwargs(self):
        return {**self.istft_kwargs, "return_complex": True}

    @property
    def istft_kwargs(self):
        return dict(
            n_fft=self.n_fft, hop_length=self.hop_length,
            window=self.window, center=True
        )

    def _get_window(self, x):
        window = self.windows.get(x.device, None)
        if window is None:
            window = self.window.to(x.device)
            self.windows[x.device] = window
        return window

    def stft(self, sig):
        window = self._get_window(sig)
        return torch.stft(sig, **{**self.stft_kwargs, "window": window})

    def istft(self, spec, length=None):
        window = self._get_window(spec)
        return torch.istft(spec, **{**self.istft_kwargs, "window": window, "length": length})

    def train_dataloader(self):
        return DataLoader(
            self.train_set, batch_size=self.batch_size,
            num_workers=self.num_workers, pin_memory=self.gpu, shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_set, batch_size=self.batch_size,
            num_workers=self.num_workers, pin_memory=self.gpu, shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set, batch_size=self.batch_size,
            num_workers=self.num_workers, pin_memory=self.gpu, shuffle=False
        )