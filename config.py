from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class Route1Config:
    # runtime
    seed: int = 42
    device: str = "cuda"
    num_workers: int = 0

    # dataset
    dataset_name: str = "CWRU"  # CWRU | JNU | SELF_A | SELF_B
    cwru_root: str = "data/CWRU"
    jnu_root: str = "data/JNU"
    self_processed_root: str = "data/SELF/Processed_Data_Numpy"
    self_a_subdir: str = "SelfA"
    self_b_subdir: str = "SelfB"
    self_norm_mode: str = "train_global"  # train_global | per_sample
    self_norm_mean: float = 0.0
    self_norm_std: float = 1.0
    self_use_saved_norm: bool = False
    sequence_length: int = 1024
    stride: int = 512
    known_classes: List[int] = field(default_factory=lambda: [0, 1, 3])
    unknown_classes: List[int] = field(default_factory=lambda: [2])
    source_domains: List[str] = field(default_factory=list)
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # model
    latent_dim: int = 128
    m_max: int = 15
    routing_temp: float = 0.2
    expert_active_threshold: float = 0.02

    # optimization
    epochs: int = 30
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 5.0

    # route1 losses
    loss_w_cls: float = 1.0
    loss_w_rec: float = 1.0
    loss_w_adv: float = 0.2
    loss_w_balance: float = 0.005
    loss_w_entropy: float = 0.01
    loss_w_collapse: float = 0.05
    grl_alpha: float = 0.2

    # route1 schedule
    route1_pretrain_epochs: int = 5
    conf_threshold_start: float = 0.08
    conf_threshold_end: float = 0.55
    balance_decay_epochs: int = 8
    balance_min_factor: float = 0.1
    routing_hard_after_epoch: int = 3
    routing_noise_std: float = 0.05
    target_k: int = 7
    sparsify_start_epoch: int = 4
    sparsify_end_epoch: int = 16
    ema_mse_momentum: float = 0.9

    # output
    output_dir: str = "outputs"

    # baselines
    baseline_method: str = "ERM"
    cvar_alpha: float = 0.2
    groupdro_eta: float = 0.01
    jtt_stage1_epochs: int = 5
    jtt_upweight: float = 20.0
    diversify_k: int = 4
    diversify_grl_alpha: float = 0.2
    diversify_adv_weight: float = 0.2
    aminet_supcon_weight: float = 0.1
    dacn_supcon_weight: float = 0.1
    dacn_adv_weight: float = 0.2
    afrdn_num_env: int = 4
    afrdn_rec_weight: float = 1.0
    afrdn_adv_weight: float = 0.2
    wpa_aug_weight: float = 0.5
    wpa_adv_weight: float = 0.2
    irm_num_env: int = 2
    irm_penalty_weight: float = 0.1
    eiil_warmup_epochs: int = 5
    eiil_num_env: int = 2
    eiil_reweight: float = 3.0
    mixup_alpha: float = 0.4
    mixstyle_alpha: float = 0.3
    mixstyle_prob: float = 0.5
    ood_score_type: str = "msp"  # msp | energy | maxlogit
    ood_energy_temp: float = 1.0

    # route1 ablations
    ablation_disable_natural_collapse: bool = False
    ablation_disable_confidence_weighting: bool = False
    ablation_disable_worst_step: bool = False
    ablation_disable_reconstruction_routing: bool = False
    ablation_soft_only_routing: bool = False
    ablation_equal_capacity_no_env: bool = False

    @property
    def num_known_classes(self) -> int:
        return len(self.known_classes)

    def resolve_paths(self, project_root: Path) -> None:
        self.cwru_root = str((project_root / self.cwru_root).resolve())
        self.jnu_root = str((project_root / self.jnu_root).resolve())
        self.self_processed_root = str((project_root / self.self_processed_root).resolve())
        self.output_dir = str((project_root / self.output_dir).resolve())
