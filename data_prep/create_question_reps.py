# %%
import argparse
import os
from typing import Optional

import cohere
import torch
from datasets_plus import load_dataset
from loguru import logger
from sentence_transformers import SentenceTransformer

embedding_models = [
    "all-mpnet-base-v2",
    "cohere-embed-english-v3.0",  # Cohere model
    "gritlm-7b",
    "gritlm-7b-custom",
    "e5-mistral-7b-instruct",
]


E5_MISTRAL_INST = (
    "Instruct: Given a question and the answer context, identify the the primary topics,"
    " entities and different challenges: linguistic, factual, relational and "
    "reasoning based, and return questions that pose similar challenges.\n"
)


# class InstSentenceTransformer(SentenceTransformer):
#     def __init__(self, model_name: str, instruction: str):
#         super().__init__(model_name)
#         self.instruction = instruction
#         self.max_seq_length = 4096

#     def encode(self, sentences: list[str]):
#         return encode(sentences, prompt=self.instruction)


GRIT_INST = (
    "Given a question and the answer context, identify the primary topics, "
    "entities and different challenges: linguistic, factual, "
    "relational and reasoning based."
)

GRIT_INST_2 = (
    "Given a question, the answer, and the associated Wikipedia Page, fetch questions that "
    "pose similar challenges in terms of high level topics, focus entities, sentence structure, semantics, "
    "and different kinds of reasoning skills required to find the correct answer: knowledge based, "
    "linguistic, relational, temporal and spatial."
)

# Create a prompt  to get the embedding that captures.


class GritEmbeddingModel:
    def __init__(
        self, model_name: str = "GritLM/GritLM-7B", instruction: str = GRIT_INST
    ):
        self.model = GritLM(model_name, mode="embedding", torch_dtype=torch.bfloat16)
        self.instruction = self._gritlm_instruction(instruction)
        self.model_name = model_name

    def _gritlm_instruction(self, instruction):
        return (
            "<|user|>\n" + instruction + "\n<|embed|>\n"
            if instruction
            else "<|embed|>\n"
        )

    def encode(self, documents: list[str]):
        vec = self.model.encode(documents, instruction=self.instruction)
        return torch.tensor(vec)


class CohereEmbeddingModel:
    def __init__(self, model_name: str, input_type: Optional[str] = None):
        self.model_name = model_name
        self.input_type = input_type
        self.co = cohere.Client(os.environ["COHERE_API_KEY"])

    def encode(self, documents: list[str]):
        resp = self.co.embed(
            texts=documents,
            model=self.model_name,
            input_type=self.input_type,
        )
        return torch.tensor(resp.embeddings)


def get_model(model_name: str, embed_task: Optional[str] = None):
    """Get a model for embedding. embed_task is only used for Cohere models."""
    if model_name.startswith("cohere"):
        return CohereEmbeddingModel(model_name.split("-", 1)[1], embed_task)
    if model_name.startswith("gritlm"):
        instruction = GRIT_INST_2 if "custom" in model_name else GRIT_INST
        return GritEmbeddingModel(instruction=instruction)
    if model_name.startswith("e5-mistral"):
        return SentenceTransformer(f"intfloat/{model_name}")
    return SentenceTransformer(model_name)


def make_model_inputs(entry: dict[str, str], strategy: str):
    question = entry["question"] if "question" in entry else entry["claim"]
    if strategy == "question":
        return (question,)
    answer = entry["answer"]
    if strategy == "question-answer":
        return (f"Question: {question} Answer:{answer}",)
    if entry["wiki_summary"]:
        page_content = f"Page: {entry['focus_entity']} Summary: {entry['wiki_summary']}"
    else:
        page_content = "No page summary (Not an entity question)"
    if strategy == "question-answer-ref":
        return (f"Question: {question} Answer:{answer} {page_content}",)
    if strategy == "question-answer-and-ref":
        return (f"Question: {question} Answer:{answer}", f"{page_content}")


def get_strategy_code(strategy: str):
    if strategy == "question":
        return "q"
    if strategy == "question-answer":
        return "qa"
    if strategy == "question-answer-ref":
        return "qar"
    if strategy == "question-answer-and-ref":
        return "qa,r"


def make_embeddings(dataset, model, strategy: str, batch_size=32):
    def encode_batch(batch):
        batch_entries = [dict(zip(batch, t)) for t in zip(*batch.values())]
        docs = [make_model_inputs(entry, strategy) for entry in batch_entries]

        # Flatten the list of docs
        flat_docs = [doc for sublist in docs for doc in sublist]
        embeddings = model.encode(flat_docs)

        # Reshape embeddings if strategy returns multiple docs per entry
        if strategy in ["question-answer-and-ref"]:
            embeddings = embeddings.reshape(
                len(batch_entries), -1, embeddings.shape[-1]
            )
        return {"embedding": embeddings}

    encoded_dataset = dataset.map(
        encode_batch,
        batched=True,
        batch_size=batch_size,
        desc="Encoding documents",
    ).with_format("torch")

    return encoded_dataset


def main(args: argparse.Namespace):
    logger.info("Loading Model[{}]", args.model_name)
    model = get_model(args.model_name, args.embed_task)
    if "mpnet" in args.model_name:
        model_name = "mpnet-base-v2"
    elif "cohere" in args.model_name and "v3" in args.model_name:
        model_name = f"cohere-embed-v3-{args.embed_task}"
    else:
        model_name = args.model_name

    strategy_code = get_strategy_code(args.strategy)
    output_path = (
        f"data/embeddings/{args.dataset_name}-{strategy_code}-{model_name}-embed.pt"
    )

    logger.info("Output path: {}", output_path)
    dataset_full_name = f"datasets/{args.dataset_name}"
    dataset = load_dataset(dataset_full_name)

    logger.info(f"{args.dataset_name}: Encoding documents...")
    encoded_dataset = make_embeddings(dataset, model, args.strategy, args.batch_size)
    state_dict = dict(zip(encoded_dataset["id"], encoded_dataset["embedding"]))

    logger.info(f"{args.dataset_name}: Saving embeddings to {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(state_dict, output_path)


def add_arguments(
    parser: Optional[argparse.ArgumentParser] = None,
) -> argparse.ArgumentParser:
    if parser is None:
        parser = argparse.ArgumentParser("Make question embeddings.")

    parser.add_argument("--dataset_name", "-d", type=str, default="advqa")

    parser.add_argument(
        "--model-name", "-m", default="all-mpnet-base-v2", choices=embedding_models
    )

    parser.add_argument(
        "--embed-task",
        "-t",
        default="clustering",
        help="Cohere embedding task, only used for Cohere v3 embed models. Ignored otherwise.",
    )

    parser.add_argument(
        "--strategy",
        "-s",
        default="question-answer-ref",
        choices=[
            "question",
            "question-answer",
            "question-answer-ref",
            "question-answer-and-ref",
        ],
    )

    parser.add_argument(
        "--batch-size",
        "-bs",
        default=32,
        type=int,
    )

    return parser


if __name__ == "__main__":
    parser = add_arguments()
    args = parser.parse_args()
    main(args)

# %%
import os
import re


def rename_embedding_files():
    embeddings_dir = "data/embeddings"

    for filename in os.listdir(embeddings_dir):
        if "question-answer-ref" in filename:
            new_filename = filename.replace("question-answer-ref", "qar")
            old_path = os.path.join(embeddings_dir, filename)
            new_path = os.path.join(embeddings_dir, new_filename)
            os.rename(old_path, new_path)
            print(f"Renamed: {filename} -> {new_filename}")


if __name__ == "__main__":
    rename_embedding_files()

# %%
