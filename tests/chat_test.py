import pandas as pd
import pytest

from openai_batch_utils import OpenAIChat

@pytest.mark.parametrize("batch_size", [1000, 3])
def test_openai_chat_basic(batch_size: int):
    prompt = ['what is the capital of netherlands',
              'what is the capital of france',
              'what is the capital of germany',
              'what is the capital of italy',
              'what is the capital of spain']

    chat = OpenAIChat()
    result = chat.openai_chat(prompt=prompt,
                              batch_size=batch_size,
                              model='gpt-3.5-turbo',
                              sleep_interval=5)

    assert len(result) == len(prompt)
    assert 'amsterdam' in result[0].lower()
    assert 'paris' in result[1].lower()
    assert 'berlin' in result[2].lower()
    assert 'rome' in result[3].lower()
    assert 'madrid' in result[4].lower()


@pytest.mark.parametrize("batch_size", [1000, 3])
def test_openai_chat_json_mode(batch_size: int):

    prompt = ['what is the capital of netherlands',
              'what is the capital of france',
              'what is the capital of germany',
              'what is the capital of italy',
              'what is the capital of spain']

    system = """Reply in JSON format. Obey the following schema:
    {'city': `Capital`. 'country': `Country`}"""

    chat = OpenAIChat()
    result = chat.openai_chat(prompt=prompt,
                              system_prompt=system,
                              batch_size=batch_size,
                              model='gpt-3.5-turbo',
                              response_format={'type': 'json_object'},
                              sleep_interval=5)

    assert len(result) == len(prompt)
    assert type(result[0]) == dict
    df = pd.DataFrame(result)
    assert df.shape[0] == len(prompt)
    assert df.shape[1] == 2
    assert df.columns.tolist() == ['city', 'country']
    assert df['city'].tolist() == ['Amsterdam', 'Paris', 'Berlin', 'Rome', 'Madrid']
    assert df['country'].tolist() == ['Netherlands', 'France', 'Germany', 'Italy', 'Spain']
