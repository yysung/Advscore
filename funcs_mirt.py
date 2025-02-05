# %%
import os.path as osp
import pickle

import numpy as np

SKILL_COLS = ["skill_0"]
REL_COLS = ["disc_0"]
DIFF_COLS = ["diff"]

CKPT_DIR = "<ckpt_dir>/"
# 1 dim exp_names
exp_names = {
    "advqa_combined": "mirt-advqa_combined-1-dim_diff-values_imp-values_fit-imp_Adam-lr=5e-03_c-reg-skill=1e-5-diff=1e-5-imp=1e-6_sampler=none-w=7.0_bs=512",
    "bamboogle": "mirt-bamboogle-1-dim_diff-values_imp-values_fit-imp_Adam-lr=5e-03_c-reg-skill=1e-5-diff=1e-5-imp=1e-6_sampler=none-w=7.0_bs=512",
    "fm2": "mirt-fm2-1-dim_diff-values_imp-values_fit-imp_Adam-lr=2e-03_c-reg-skill=1e-5-diff=1e-5-imp=1e-6_sampler=weighted-w=4.0_bs=512",
    "trickme": "mirt-trickme-1-dim_diff-values_imp-values_fit-imp_Adam-lr=5e-03_c-reg-skill=1e-5-diff=1e-5-imp=1e-6_sampler=weighted-w=7.0_bs=256",
}


def load_dataframe_dict(dataset_name: str, tag: str = "best"):
    filepath = osp.join(CKPT_DIR, exp_names[dataset_name], f"{tag}.pkl")
    with open(filepath, "rb") as f:
        all_data = pickle.load(f)
    all_data["agents"].set_index("subject_id", inplace=True)
    all_data["agents"]["subject_id"] = all_data["agents"].index
    return all_data

def irt_logit_func(skills, diff, rels):
    # skills: (n_agents, n_dim)
    if len(skills.shape) == 2:
        logits = (
            np.sum(skills[:, None, :] * rels[None, :, :], axis=-1) - diff[None, :, 0]
        )
        logits = logits.astype(np.float32)
        return logits
    else:  # Shape of skills = (n_dim)
        logits = np.sum(skills * rels, axis=-1) - diff[:, 0]
        logits = logits.astype(np.float32)
        return logits
