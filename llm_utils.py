# %%
import os
from typing import Optional

import json_repair
import litellm
from litellm.caching import Cache

litellm.cache = Cache()

entity_categorization_system_prompt = """As an NLP expert, categorize the given string into the most specific object type. Consider the following categories:

1. Common types: Date, Time, Year, Number, Currency, Measurement
2. Entities (use a hierarchical structure): Entity/Person, Entity/Location, Entity/Organization, Entity/Concept, Entity/Event, Entity/Product, Entity/Language, Entity/Movie, Entity/Music, Entity/Book, etc.
3. Simple responses: BooleanResponse (for "Yes", "No", "True", "False"), ImpossibilityStatement (for "Impossible", "Cannot be determined", "Unknown", etc.)

Create appropriate categories if needed.

Return your answer in the following JSON format:
{
  "object_type": "Category name (use hierarchical structure for entities)",
  "confidence": 0-100
}"""

focus_entity_system_prompt = """As an NLP expert, analyze the input question and determine if there's a specific named entity that is the focus. The entity should be explicitly mentioned in the question and directly related to the main topic being asked about. Also, disambiguate between multiple entities with the same name if necessary. E.g., Purple Rain (movie) vs Purple Rain (song).

If there is a clear named entity that's central to the question, extract it. If the question is about a general concept, event, or category without mentioning a specific named entity, respond with "None".

Examples:
1. "Who wrote the novel 'To Kill a Mockingbird'?" -> "To Kill a Mockingbird"
2. "What is the capital of France?" -> "France"
3. "How many people died in the second most powerful earthquake ever recorded?" -> "None"
4. "This artist was nominated for a Grammy for Toxicity" -> "Toxicity (album)"

Return your answer in the following JSON format:
{
"entity": "Entity name/None"
}"""

adv_type_categorization_system_prompt = """
Annotation Task: Identify features and tactics used in creating adversarial questions for language models.

Instructions:
1. Analyze each question carefully, considering both the question text and the provided correct answer.
2. Select up to 3 features/tactics from the list below that best describe the adversarial nature of the question.
3. If no features/tactics apply, respond with "None".
4. Provide a detailed explanation for your choices, including specific text highlights.

Features/Tactics:
1. Composing Seen Clues: Requires integrating multiple explicit clues from the question to derive the correct answer.
2. Logic and Calculation: Involves mathematical operations, logical deductions, or manipulations of partial information.
3. Multi-Step Reasoning: Necessitates a sequence of inferential steps between different entities or concepts to reach the correct conclusion.
4. Negation: Employs "not", "non-", "no", or other forms of negation that could potentially misdirect the model's understanding or response.
5. Temporal Misdirection: Incorporates specific dates, time periods, or temporal relationships that may lead the model to an incorrect temporal context.
6. Location Misdirection: Utilizes geographical information or spatial relationships in a way that could mislead the model's spatial reasoning.
7. Commonsense: Requires application of general world knowledge not explicitly stated in the question. Specify the particular commonsense knowledge needed when applicable.
8. Domain Expertise: Demands specialized knowledge in a specific field (e.g., science, history, literature, technology) to accurately interpret and answer the question.
9. Irrelevant Clues: Includes extraneous or redundant information designed to distract the model from the core inference required for the correct answer.
10. Crosslingual: Incorporates multilingual elements or cross-lingual references that may challenge the model's language understanding or translation capabilities.

Response Format (JSON):
{
    "categories": ["Feature1", "Feature2", "Feature3"],
    "explanation": "Detailed reasoning for the chosen features/tactics. Blue highlight: [text indicating adversarial elements]. Yellow highlight: [text indicating question type or key elements]."
}

Example:
Q: "In 2025, which city will not host the Olympics that it hosted in 1964?"
A: "Tokyo"

Annotation:
{
    "categories": ["Temporal Misdirection", "Negation"],
    "explanation": "This question employs two key adversarial tactics. Temporal Misdirection: The question mentions a future date ('2025') and a past date ('1964'), potentially confusing a model about the temporal context. Blue highlight: [In 2025], [1964]. Negation: The use of 'not' in the question could mislead a model into providing an affirmative answer instead of identifying the city that won't host. Blue highlight: [will not host]. The combination of these elements makes it challenging for a model to correctly identify Tokyo as the answer. Yellow highlight: [which city will not host the Olympics that it hosted]."
}

If no features/tactics apply:
{
    "categories": ["None"],
    "explanation": "After careful analysis, no specific adversarial features or tactics were identified in this question. The question appears to be straightforward and does not employ any of the listed adversarial techniques."
}

Note: Ensure your explanation is thorough, directly referencing the question text and explaining how each identified feature/tactic contributes to the question's adversarial nature. Consider how these elements might challenge a language model in producing the correct answer provided.
"""


def query_model(
    input_text: str,
    system_prompt: Optional[str] = None,
    return_json: bool = False,
    model: str = "gpt-4o-mini",
):
    if system_prompt is None:
        messages = [{"content": input_text, "role": "user"}]
    else:
        messages = [
            {"content": system_prompt, "role": "system"},
            {"content": input_text, "role": "user"},
        ]
    response = litellm.completion(
        model="gpt-4o-mini",
        response_format={"type": "json_object"} if return_json else None,
        messages=messages,
    )

    if return_json:
        return json_repair.loads(response["choices"][0]["message"]["content"])
    return response["choices"][0]["message"]["content"]


def categorize_answer(answer: str):
    return query_model(answer, entity_categorization_system_prompt, return_json=True)


def find_focus_entity(question: str):
    entity = query_model(question, focus_entity_system_prompt, return_json=True)[
        "entity"
    ]
    if entity != "None" and entity.lower() not in question.lower():
        print(f"Warning: entity {entity} not in question {question}")
    return entity


def generate_wiki_summary(
    entity_name: str, word_limit: int = 200, model: str = "gpt-4o"
):
    prompt = f"Generate a summary of the Wikipedia page for {entity_name} in {word_limit} words or less."
    return query_model(prompt, model=model)


def find_wikipedia_page(entity_name: str):
    import requests

    base_url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "opensearch",
        "search": entity_name,
        "limit": 10,
        "namespace": 0,
        "format": "json",
    }

    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        data = response.json()
        if data[1]:  # Check if there are any search results
            return data[1][0]  # Return the URL of the first result
    return None  # Return None if no results or if the request failed


def get_wikipedia_summary(page_title: str, max_chars: int = 1000):
    import requests

    base_url = "https://en.wikipedia.org/w/api.php"
    params = {
        "format": "json",
        "action": "query",
        "prop": "extracts",
        "exintro": True,
        "explaintext": True,
        "redirects": 1,
        "titles": page_title,
    }

    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        data = response.json()
        page = next(iter(data["query"]["pages"].values()))
        if "extract" in page:
            summary = page["extract"]
            return summary[:max_chars] + ("..." if len(summary) > max_chars else "")
    return None  # Return None if no summary found or if the request failed


def categorize_adv_type(question: str, answer: str):
    input_text = f"Answer: {answer}. Question about {answer}: {question}"
    return query_model(
        input_text, adv_type_categorization_system_prompt, return_json=True
    )


# %%
