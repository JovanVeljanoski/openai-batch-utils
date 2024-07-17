import pytest

from openai_batch_utils import OpenAIEmbed

@pytest.mark.parametrize("batch_size", [1000, 3])
def test_openai_embedding_basic(batch_size: int):
    prompt = ['what is the capital of netherlands',
              'what is the capital of france',
              'what is the capital of germany',
              'what is the capital of italy',
              'what is the capital of spain']

    embed = OpenAIEmbed()
    result = embed.openai_embed(input=prompt,
                                model='text-embedding-3-small',
                                batch_size=batch_size)

    assert len(result) == len(prompt)
    for i in result:
        assert len(i) == 1536
