
import torch
from torch.utils.tensorboard import SummaryWriter


SUM_FREQ = 100 # TODO invariant to gpu_nums

# Short display names for known metric keys
_METRIC_LABELS = {
    "loss/train":            "L_total",
    "loss/pose_train":       "L_pose",
    "loss/rotation_train":   "L_rot",
    "loss/translation_train":"L_trans",
    "loss/flow_train":       "L_flow",
    "loss/scores_train":     "L_scorer",
    "scorer/regression_train":"sc_reg",
    "scorer/logbarrier_train":"sc_log",
    "scorer/rank_frames_train":"rank_f",
    "scorer/rank_pairs_train":"rank_p",
    "scorer/utility_mean_train":"u_mean",
    "scorer/pos_utility_mean_train":"u_pos",
    "scorer/neg_utility_mean_train":"u_neg",
    "scorer/valid_patch_fraction_train":"p_valid",
    "scorer/isotropy_mean_train":"iso",
    "scorer/survival_mean_train":"surv",
    "scorer/repeatability_train":"rep",
    "scorer/diversity_penalty_train":"div",
    "scorer/motion_spread_train":"mot",
    "scorer/teacher_alignment_train":"teach",
    "scorer/teacher_weight_train":"teach_w",
    "scorer/info_head_train":"info_h",
    "scorer/conditioning_head_train":"cond_h",
    "scorer/replay_valid_fraction_train":"rep_v",
    "scorer/fb_cycle_error_px_train":"fb_px",
    "scorer/replay_stability_px_train":"rep_px",
    "loss/cm_train":         "L_cm",
    "loss/cm_weight":        "cm_w",
    "val/epe_feature_px":    "val_epe",
    "val/patch_survival_rate":"val_surv",
    "val/rejection_rate":    "val_rej",
    "val/selected_patch_isotropy":"val_iso",
    "val/selected_patch_anisotropy":"val_aniso",
    "val/selection_diversity":"val_div",
    "val/score_dynamic_range":"val_rng",
    "val/fb_cycle_error_px":"val_fb",
    "val/replay_stability_px":"val_rep",
    "val/replay_valid_fraction":"val_rv",
    "px1":                   "px<.25",
    "r1":                    "r<.001",
    "r2":                    "r<.01",
    "t1":                    "t<.001",
    "t2":                    "t<.01",
}

class Logger:
    def __init__(self, name, scheduler, total_steps=0, step=1):
        self.total_steps = total_steps
        self.step = step
        self.running_loss = {}
        self.writer = None
        self.name = name
        self.scheduler = scheduler

    def _print_training_status(self):
        if self.writer is None:
            self.writer = SummaryWriter("runs/{}".format(self.name))

        lr = self.scheduler.get_lr().pop() # TODO use get_last_lr()

        parts = ["step={:<6d}  lr={:.2e}".format(
            self.total_steps * self.step + 1, lr)]
        for k in self.running_loss:
            val = self.running_loss[k] / SUM_FREQ
            label = _METRIC_LABELS.get(k, k.split("/")[-1])
            parts.append("{}={:.4f}".format(label, val))

        print("  ".join(parts))

        for key in self.running_loss:
            val = self.running_loss[key] / SUM_FREQ
            # TODO all losses in one diagram (add_scalars)
            self.writer.add_scalar(key, val, self.total_steps * self.step)
            self.running_loss[key] = 0.0
        self.writer.add_scalar("lr", lr, self.total_steps * self.step)

    def push(self, metrics):

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % SUM_FREQ == SUM_FREQ - 1:
            self._print_training_status()
            self.running_loss = {}

        self.total_steps += 1

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter("runs/{}".format(self.name))
            print([k for k in self.running_loss])
            
        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps * self.step)

    def write_figures(self, figures):
        if self.writer is None:
            self.writer = SummaryWriter("runs/{}".format(self.name))
            
        for key in figures:
            self.writer.add_figure(key, figures[key], self.total_steps * self.step)

    def close(self):
        if self.writer is not None:
            self.writer.close()
