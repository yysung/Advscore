# %%
import os
import os.path as osp
from tqdm import tqdm
from llm_utils import categorize_answer

import pandas as pd
import datasets
from llm_utils import find_focus_entity, generate_wiki_summary
from concurrent.futures import ProcessPoolExecutor, as_completed

os.environ["HF_TOKEN"] = "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"

# %%
data_dir = "data"


dataset_paths = {
    "advqa": osp.join(data_dir, "Advqa_text.csv"),
    "bamboogle": osp.join(data_dir, "Bamboogle_text.csv"),
    "trickme": osp.join(data_dir, "Trickme_text.csv"),
    "fm2": osp.join(data_dir, "Fm2_text.csv"),
}

ds_columns = {
    "advqa": ["Question", "Answer"],
    "bamboogle": ["Question", "Answer"],
    "trickme": ["Question", "Answer"],
    "fm2": ["Claim", "Answer", "Gold_evidence", "Gold_page"],
}


def load_dataset_frame(dataset_name):
    path = dataset_paths[dataset_name]
    df = pd.read_csv(path)[ds_columns[dataset_name]]
    df.rename(
        columns={"Question": "question", "Claim": "claim", "Answer": "answer"},
        inplace=True,
    )
    if dataset_name == "fm2":
        df.rename(
            columns={"Gold_evidence": "evidence", "Gold_page": "page"}, inplace=True
        )
        df["answer"] = df["answer"].apply(
            lambda x: "Correct" if x == "SUPPORTS" else "Incorrect"
        )
    return df


def map_focus_entity_and_reference(entry: dict):
    ret = {}
    ret["answer_type"] = categorize_answer(entry["answer"])["object_type"]

    if ret["answer_type"].startswith("Entity/"):
        ret["focus_entity"] = entry["answer"]
    else:
        ret["focus_entity"] = find_focus_entity(entry["question"])

    if ret["focus_entity"] != "None":
        ret["wiki_summary"] = generate_wiki_summary(ret["focus_entity"])
    else:
        ret["wiki_summary"] = ""
    return ret


def map_focus_entity_and_reference_fm2(entry: dict):
    ret = {}
    ret["answer_type"] = categorize_answer(entry["page"])["object_type"]
    ret["focus_entity"] = entry["page"]
    ret["wiki_summary"] = entry["evidence"]
    return ret


def prepare_dataset(dataset_name: str):
    if osp.exists(f"datasets/{dataset_name}"):
        print(f"Dataset {dataset_name} already exists.")
        return
    df = load_dataset_frame(dataset_name)
    q_key = "claim" if dataset_name == "fm2" else "question"
    a_key = "answer"
    data = {
        q_key: df[q_key].tolist(),
        "answer": df[a_key].tolist(),
    }
    print(df.columns)
    if dataset_name == "fm2":
        data["evidence"] = df["evidence"].tolist()
        data["page"] = df["page"].tolist()

    dataset = datasets.Dataset.from_dict(data)
    dataset = dataset.map(
        lambda x, idx: {"id": f"q{idx+1}"}, with_indices=True, keep_in_memory=True)
    if dataset_name == "fm2":
        dataset = dataset.map(map_focus_entity_and_reference_fm2)
    else:
        dataset = dataset.map(map_focus_entity_and_reference)
    dataset.save_to_disk(f"datasets/{dataset_name}")





def parallel_prepare_dataset(dataset_name):
    prepare_dataset(dataset_name)
    print(f"Dataset {dataset_name} prepared.")


with ProcessPoolExecutor() as executor:
    futures = [
        executor.submit(parallel_prepare_dataset, dataset_name)
        for dataset_name in dataset_paths.keys()
    ]
    for future in as_completed(futures):
        future.result()
