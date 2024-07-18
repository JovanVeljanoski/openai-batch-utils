import asyncio
import json

from tenacity import retry, stop_after_attempt, wait_random_exponential

from .base import OpenAIBase
from .utils import async_to_sync


class OpenAIChat(OpenAIBase):
    def __init__(self, api_key: str = None, **kwargs):
        """A class for generating (batch) chat responses using the OpenAI GPT model.

        Args:
            api_key (str): The API key used for authentication.
            **kwargs: Additional keyword arguments to be passed to the `AsyncOpenAI` client.

        Returns:
            None
        """
        super().__init__(api_key=api_key, **kwargs)

    @async_to_sync
    async def openai_chat(
        self,
        prompt: str | list,
        system_prompt: str = "",
        model: str = "gpt-4o-2024-05-13",
        max_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        response_format: dict = None,
        batch_size: int = 1000,
        sleep_interval: int = 60,
        verbose: bool = False,
    ) -> list[str]:
        """
        Generates chat responses using the OpenAI GPT model.

        Args:
            prompt (str | list): The input prompt(s) for the chat conversation.
            system_prompt (str): The system message prompt for the chat conversation.
            model (str, optional): The model to use for generating responses. Defaults to "gpt-4o-2024-05-13".
            max_tokens (int, optional): The maximum number of tokens in the generated response. Defaults to 256.
            temperature (float, optional): The temperature parameter for controlling the randomness of the generated response. Defaults to 1.0.
            top_p (float, optional): The top-p parameter for controlling the diversity of the generated response. Defaults to 1.0.
            frequency_penalty (float, optional): The frequency penalty parameter for penalizing frequently generated tokens. Defaults to 0.0.
            presence_penalty (float, optional): The presence penalty parameter for penalizing tokens that are not present in the input. Defaults to 0.0.
            response_format (dict, optional): The desired format of the generated response. Defaults to None.
            batch_size (int, optional): The batch size for making API requests. Defaults to 1000.
            sleep_interval (int, optional): The sleep interval between batches to relieve the API. Defaults to 60.
            verbose (bool, optional): Whether to print verbose output. Defaults to False.

        Returns:
            list[str]: A list of generated chat responses.
        """

        if isinstance(prompt, str):
            prompt = [prompt]
        if not isinstance(system_prompt, str):
            raise ValueError("system_prompt must be a string. Only one system prompt is supported.")

        res: list = []
        tasks: list = []

        for batch in self._get_list_slices(prompt, batch_size):
            for one_prompt in batch:
                task = self.create_gpt_call_task(
                    prompt=one_prompt,
                    system_prompt=system_prompt,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    response_format=response_format,
                )
                tasks.append(task)
            if verbose:
                print(f"Dispatching {len(tasks)} requests to OpenAI. Awaiting their response...")
            x = await asyncio.gather(*tasks)
            res.extend(x)
            if len(tasks) == batch_size:
                if verbose:
                    print(f"Sleeping for {sleep_interval} seconds to relieve the API a bit ...")
                await asyncio.sleep(sleep_interval)
            tasks = []

        return res

    @retry(wait=wait_random_exponential(min=10, max=80), stop=stop_after_attempt(11))
    async def create_gpt_call_task(
        self,
        prompt: str,
        system_prompt: str,
        model: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        frequency_penalty: float,
        presence_penalty: float,
        response_format: dict = None,
    ) -> str | dict:
        """
        Creates a GPT call task using OpenAI's chat completions API.

        Args:
            prompt (str): The user's prompt for the conversation.
            system_prompt (str): The system's prompt for the conversation.
            model (str): The model to use for generating the response.
            max_tokens (int): The maximum number of tokens in the generated response.
            temperature (float): Controls the randomness of the generated response. Higher values make the output more random.
            top_p (float): Controls the diversity of the generated response. Lower values make the output more focused.
            frequency_penalty (float): Controls the penalty for using frequent tokens in the generated response.
            presence_penalty (float): Controls the penalty for using tokens that are not present in the input.
            response_format (dict, optional): The format of the response. Defaults to None.

        Returns:
            str or dict: The generated response in the specified format.

        Raises:
            ValueError: If the response_format type is not recognized or supported.

        """
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]

        response = await self.client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            response_format=response_format,
        )

        raw_json = response.choices[0].message.content

        if response_format is None:
            return raw_json
        elif response_format.get("type") == "json_object":
            return json.loads(raw_json)
        else:
            raise ValueError("response_format type not recognized or not supported. Use 'json_object' for a JSON object or None for a raw string.")
