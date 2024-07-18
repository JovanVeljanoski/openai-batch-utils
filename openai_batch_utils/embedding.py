import asyncio

from .base import OpenAIBase
from .utils import async_to_sync


class OpenAIEmbed(OpenAIBase):
    def __init__(self, api_key: str | None = None, **kwargs):
        """A class for generating (batch) embeddings using the OpenAI API.

        Args:
            api_key (str): The API key used for authentication.
            **kwargs: Additional keyword arguments to be passed to the `AsyncOpenAI` client.

        Returns:
            None
        """
        super().__init__(api_key=api_key, **kwargs)

    @async_to_sync
    async def openai_embed(
        self,
        input: str | list[str],
        model: list,
        batch_size: int = 1000,
    ) -> list[float] | list[list[float]]:

        if isinstance(input, str):
            input = [input]

        tasks = []

        for batch in self._get_list_slices(input, batch_size):
            task = self.client.embeddings.create(
                input=batch,
                model=model,
            )
            tasks.append(task)

        res = await asyncio.gather(*tasks)
        res = [item.embedding for sublist in res for item in sublist.data]
        return res
