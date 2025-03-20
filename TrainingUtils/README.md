# 2025.03.20 Equinox Decaying Asymmetric Sinusoidal LR Scheduler
1. [`EquinoxDecayingAsymmetricSinusoidalLRScheduler`](https://github.com/KUcyans/IceCubeTransformer/blob/main/TrainingUtils/EquinoxDecayingAsymmetricSinusoidal.py)
    * a learning rate scheduler which decays with transient asymmetric sinusoidal oscillations
      * the unit oscillation component is calculated by 
      $$S_n(x) = \sum_{k=1}^{n} \frac{\binom{2n}{n-k}}{\binom{2n}{n} \cdot k} \sin(kx)$$
      * the first peak to peak for `n=10` case will be used as the core of the scheduler - the blue curve in the figure below
      ![alt text](image.png)
    * the overall profile is divided into phases whose value is determined by `n_sections`
    * incorporating with `n_sections`, and `total_steps`, member functions will calculate the ceiling, floor, and amplitude at each phase
    * `frequency_per_section` will decide the frequency of the sinusoidal oscillation within each phase
    ![alt text](image-1.png)  
    `n_sections=8`, `frequency_per_section=4`, `total_steps=1000`, `decay_mode='linear'`  
    ![alt text](image-2.png)
    `n_sections=8`, `frequency_per_section=8`, `total_steps=1000`, `decay_mode='linear'`  
    * there are different decay modes, each of which has a different way of calculating the ceiling, the floor and the amplitude. See [`LrDecayMode.py`](https://github.com/KUcyans/IceCubeTransformer/blob/main/Enum/LrDecayMode.py)
        1. `linear` decay: the ceiling lr value diminishes by `(min(self.n_sections/10, 0.9)` at each phase  
        ![alt text](image-3.png)  
        `n_sections=10`, `frequency_per_section=8`, `total_steps=1000`, `decay_mode='linear'`  
        2. `exponential` decay: logarithm of the storey height(`ceiling - floor`) of each phase is constant, hence the exponential decay
        ![alt text](image-4.png)  
        `n_sections=10`, `frequency_per_section=8`, `total_steps=1000`, `decay_mode='exponential'`  
        3. `cosine` decay: the logarithm of the storey height passed to $(\frac{1 + cos(\frac{\pi x}{n_{section}})}{2})$ to redefine ceiling, floor, and amplitude
        ![alt text](image-5.png)  
        `n_sections=10`, `frequency_per_section=8`, `total_steps=1000`, `decay_mode='cosine'`  

2. [`KaturaCosineAnnealingWarmupRestarts`](https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup/blob/master/cosine_annealing_warmup/scheduler.py)

3. 