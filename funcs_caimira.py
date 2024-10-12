# %%
import os.path as osp
import pickle

import numpy as np

SKILL_COLS = ["skill_0", "skill_1"]
REL_COLS = ["rel_0", "rel_1"]
DIFF_COLS = ["dif_0", "dif_1"]

DATASET_NAMES = ["advqa_combined", "fm2", "bamboogle", "trickme"]

CKPT_DIR = "/fs/clip-quiz/mgor/qa-difficulty/outputs/irt/model_ckpts"

exp_names = {
    "advqa_combined": "caimira-advqa_combined-2-dim_diff-kernel_imp-kernel_aqcevce-emb-advqa-qar-cohere-embed-v3-classification-embed_fit-imp_Adam-lr=5e-03_c-reg-skill=1e-5-diff=1e-5-imp=1e-6_sampler=none-w=4.0_bs=512",
    "bamboogle": "caimira-bamboogle-2-dim_diff-kernel_imp-kernel_bqcevce-emb-bamboogle-qar-cohere-embed-v3-classification-embed_fit-imp_Adam-lr=5e-03_c-reg-skill=1e-5-diff=1e-5-imp=1e-6_sampler=none-w=4.0_bs=512",
    "trickme": "caimira-trickme-2-dim_diff-kernel_imp-kernel_tqcevce-emb-trickme-qar-cohere-embed-v3-classification-embed_fit-imp_Adam-lr=5e-03_c-reg-skill=1e-5-diff=1e-5-imp=1e-6_sampler=weighted-w=4.0_bs=256",
    "fm2": "caimira-fm2-2-dim_diff-kernel_imp-kernel_fqcevce-emb-fm2-qar-cohere-embed-v3-classification-embed_fit-imp_Adam-lr=2e-03_c-reg-skill=1e-5-diff=1e-5-imp=1e-6_sampler=weighted-w=4.0_bs=256",
}

# exp_names = {
#     "fm2": "caimira-fm2-1-dim_diff-kernel_imp-kernel_fqcevce-emb-fm2-qar-cohere-embed-v3-classification-embed_fit-imp_Adam-lr=5e-03_c-reg-skill=1e-5-diff=1e-5-imp=1e-6_sampler=weighted-w=4.0_bs=256",
#     "trickme": "caimira-trickme-1-dim_diff-kernel_imp-kernel_tqcevce-emb-trickme-qar-cohere-embed-v3-classification-embed_fit-imp_Adam-lr=5e-03_c-reg-skill=1e-5-diff=1e-5-imp=1e-6_sampler=weighted-w=4.0_bs=256",
#     "bamboogle": "caimira-bamboogle-1-dim_diff-kernel_imp-kernel_bqcevce-emb-bamboogle-qar-cohere-embed-v3-classification-embed_fit-imp_Adam-lr=5e-03_c-reg-skill=1e-5-diff=1e-5-imp=1e-6_sampler=none-w=4.0_bs=512",
#     "advqa_combined": "caimira-advqa_combined-1-dim_diff-kernel_imp-kernel_aqcevce-emb-advqa-qar-cohere-embed-v3-classification-embed_fit-imp_Adam-lr=5e-03_c-reg-skill=1e-5-diff=1e-5-imp=1e-6_sampler=none-w=4.0_bs=512",
# }


def load_dataframe_dict(dataset_name: str, tag: str = "final"):
    filepath = osp.join(CKPT_DIR, exp_names[dataset_name], f"{tag}.pkl")
    with open(filepath, "rb") as f:
        all_data = pickle.load(f)
    all_data["agents"]["subject_id"] = all_data["agents"].index
    return all_data


def irt_logit_func(skills, diff, rels):
    # skills: (n_agents, n_dim)
    if len(skills.shape) == 2:
        latent_scores = skills[:, None, :] - diff[None, :, :]
        logits = (latent_scores * rels[None, :, :]).sum(axis=-1)
        logits = logits.astype(np.float32)
        return logits
    else:
        latent_scores = skills - diff
        logits = (latent_scores * rels).sum(axis=-1)
        logits = logits.astype(np.float32)
        return logits


# %%
