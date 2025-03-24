from openai_batch_utils import OpenAIChat

import pandas as pd  # type: ignore

import pydantic  # type: ignore

import pytest  # type: ignore


@pytest.mark.parametrize("batch_size", [1000, 3])
def test_openai_chat_basic(batch_size: int):
    prompt = [
        "what is the capital of netherlands",
        "what is the capital of france",
        "what is the capital of germany",
        "what is the capital of italy",
        "what is the capital of spain",
    ]

    chat = OpenAIChat()
    result = chat.openai_chat(prompt=prompt, batch_size=batch_size, model="gpt-4o-mini-2024-07-18", sleep_interval=5)

    assert len(result) == len(prompt)
    assert "amsterdam" in result[0].lower()
    assert "paris" in result[1].lower()
    assert "berlin" in result[2].lower()
    assert "rome" in result[3].lower()
    assert "madrid" in result[4].lower()


@pytest.mark.parametrize("batch_size", [1000, 3])
def test_openai_chat_json_mode(batch_size: int):
    prompt = [
        "what is the capital of netherlands",
        "what is the capital of france",
        "what is the capital of germany",
        "what is the capital of italy",
        "what is the capital of spain",
    ]

    system = """Reply in JSON format. Obey the following schema:
    {'city': `Capital`. 'country': `Country`}"""

    chat = OpenAIChat()
    result = chat.openai_chat(
        prompt=prompt, system_prompt=system, batch_size=batch_size, model="gpt-4o-mini-2024-07-18", response_format={"type": "json_object"}, sleep_interval=5
    )

    assert len(result) == len(prompt)
    assert isinstance(result[0], dict)
    df = pd.DataFrame(result)
    assert df.shape[0] == len(prompt)
    assert df.shape[1] == 2
    assert df.columns.tolist() == ["city", "country"]
    assert df["city"].tolist() == ["Amsterdam", "Paris", "Berlin", "Rome", "Madrid"]
    assert df["country"].tolist() == ["Netherlands", "France", "Germany", "Italy", "Spain"]


def test_system_prompt_not_string():
    chat = OpenAIChat()
    prompt = "This is some prompt"
    system = ["This is not a string but a list"]

    with pytest.raises(ValueError):
        chat.openai_chat(prompt=prompt, system_prompt=system, batch_size=1, model="gpt-4o-mini-2024-07-18", sleep_interval=5)


def test_openai_chat_return_parsed_false():
    chat = OpenAIChat()
    prompt = ["What is the capital of France?", "What is the capital of Germany?"]
    result = chat.openai_chat(prompt=prompt, return_parsed=False, batch_size=1, model="gpt-4o-mini-2024-07-18", sleep_interval=5)

    assert isinstance(result, list)
    assert hasattr(result[0], "choices")
    assert hasattr(result[0].choices[0], "message")
    assert hasattr(result[0].choices[0].message, "content")
    assert hasattr(result[0].choices[0].message, "tool_calls")

    assert isinstance(result, list)
    assert hasattr(result[1], "choices")
    assert hasattr(result[1].choices[0], "message")
    assert hasattr(result[1].choices[0].message, "content")
    assert hasattr(result[1].choices[0].message, "tool_calls")


@pytest.mark.parametrize("batch_size", [1000, 3])
def test_openai_chat_pydantic_structured_output(batch_size: int):
    class Response(pydantic.BaseModel):
        city: str
        country: str

    prompt = [
        "what is the capital of netherlands",
        "what is the capital of france",
        "what is the capital of germany",
        "what is the capital of italy",
        "what is the capital of spain",
    ]

    system = """Reply only in JSON format, using the provided schema."""

    chat = OpenAIChat()
    result = chat.openai_chat(
        system_prompt=system,
        prompt=prompt,
        batch_size=batch_size,
        model="gpt-4o-mini-2024-07-18",
        sleep_interval=5,
        response_format=Response,
    )

    assert len(result) == len(prompt)
    assert isinstance(result[0], Response)
    assert result[0].city == "Amsterdam"
    assert result[0].country == "Netherlands"
    assert result[1].city == "Paris"
    assert result[1].country == "France"
    assert result[2].city == "Berlin"
    assert result[2].country == "Germany"
    assert result[3].city == "Rome"
    assert result[3].country == "Italy"
    assert result[4].city == "Madrid"
    assert result[4].country == "Spain"
