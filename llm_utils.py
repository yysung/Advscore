import litellm
from litellm.caching import Cache
from typing import Optional

import os
import json_repair

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


