import openai


class OpenAIBase:
    def __init__(self, api_key: str | None, **kwargs):
        """A base class to be used OpenAI API clients.
        It contains common methods to be used accross different OpenAI API clients.

        Args:
            api_key (str): The API key used for authentication.
            **kwargs: Additional keyword arguments to be passed to the `AsyncOpenAI` client.

        Returns:
            None
        """
        self.api_key = api_key
        self.client = openai.AsyncOpenAI(api_key=self.api_key, **kwargs)

    def _get_list_slices(self, l: list, batch_size: int):
        """
        Slices a list into batches of a specified size.

        Args:
            l (list): The input list to be sliced.
            batch_size (int): The size of each batch.

        Yields:
            list: A batch of elements from the input list.
        """
        for i in range(0, len(l), batch_size):
            yield l[i : i + batch_size]
