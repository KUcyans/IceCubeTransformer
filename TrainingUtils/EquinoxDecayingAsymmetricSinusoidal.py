import torch
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np
import math
from scipy.optimize import minimize_scalar
from Enum.LrDecayMode import LrDecayMode


class EquinoxDecayingAsymmetricSinusoidal(_LRScheduler):
    def __init__(self, 
                optimizer: torch.optim.Optimizer,
                lr_max: float,
                lr_min: float,
                total_steps: int,
                frequency_per_section: int,
                n_sections: int,
                lr_decay: LrDecayMode = LrDecayMode.LINEAR,
                last_epoch: int = -1):

        self.total_steps = total_steps
        self.n_sections = max(n_sections, 1)
        self.section_length = max(total_steps // self.n_sections, 1)

        self.lr_max = lr_max
        self.lr_min = lr_min
        self.frequency_per_section = frequency_per_section
        self.lr_decay = lr_decay
        self.current_section = 0
        self.n = 10  # Fix self.n initialization

        super().__init__(optimizer, last_epoch)


    def get_lr(self):
        """Return the updated learning rate for each optimizer parameter group."""
        lr = self._compute_lr(self.last_epoch)  # ✅ FIXED: Use self.last_epoch
        return [lr for _ in self.optimizer.param_groups]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        new_lr = self.get_lr()[0]  # Get the computed LR
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

    def _compute_lr(self, step):
        section_index, x = self._get_section_and_x(step)
        self.current_section = min(section_index, self.n_sections - 1)

        ceiling = self._get_section_ceiling(self.current_section)
        floor = self._get_section_floor(self.current_section)
        amplitude, height = self._get_section_amplitude_and_height(ceiling, floor)

        x_scaled = (x / self.section_length) * self.frequency_per_section
        lr = self.adapted_asymmetric_sinusoidal(self.n, x_scaled, 1, amplitude, height)

        return lr
    
    def _get_section_ceiling(self, section: int):
        log_lr_max = math.log(self.lr_max)
        log_lr_min = math.log(self.lr_min)
        
        if self.lr_decay == LrDecayMode.LINEAR:
            ceiling = self.lr_max * ((min(self.n_sections/10, 0.9)) ** section)
        elif self.lr_decay == LrDecayMode.EXPONENTIAL:
            log_ceiling = log_lr_max - (log_lr_max - log_lr_min) * (section / self.n_sections)
            ceiling = math.exp(log_ceiling)
        elif self.lr_decay == LrDecayMode.COSINE:
            log_ceiling = log_lr_min + (log_lr_max - log_lr_min) * \
                        (1 + math.cos(math.pi * section / self.n_sections)) / 2
            ceiling = math.exp(log_ceiling)
        else:
            raise ValueError(f"Invalid lr_decay mode: {self.lr_decay}")

        return ceiling

    def _get_section_floor(self, section: int):
        return self._get_section_ceiling(section + 1)
    
    def _get_section_and_x(self, step):
        section_index = step // self.section_length
        x = step % self.section_length
        return section_index, x

    def _get_section_amplitude_and_height(self, ceiling, floor):
        amplitude = (ceiling - floor) / 2
        height = ceiling - amplitude
        return amplitude, height

    def adapted_asymmetric_sinusoidal(self, n, x, a, amplitude, height):
        y = self.asymmetric_sinusoidal_unit(n, x, a)
        return height + y * amplitude

    def asymmetric_sinusoidal_core(self, n, x, a, shift=0):
        total = np.zeros_like(x, dtype=np.float64)
        bin_coeff_denominator = math.comb(2 * n, n)

        for k in range(1, n + 1):
            bin_coeff = math.comb(2 * n, n - k)
            term = (bin_coeff / (bin_coeff_denominator * k)) * np.sin(a * k * (x + shift))
            total += term

        return total

    def find_peak_x(self, n, a, shift=0):
        return minimize_scalar(lambda x: -self.asymmetric_sinusoidal_core(n, x, a, shift=shift), bounds=(0, 10), method='bounded').x

    def asymmetric_sinusoidal_unit(self, n, x, a):
        first_peak_x = self.find_peak_x(n, a, shift=0)
        wavelength = self.find_peak_x(n, a, shift=first_peak_x)

        amplitude_norm = self.asymmetric_sinusoidal_core(n, first_peak_x, a, shift=0)
        return self.asymmetric_sinusoidal_core(n, x, wavelength*a, shift=first_peak_x) / amplitude_norm
