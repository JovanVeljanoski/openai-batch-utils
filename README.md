# `openai-batch-utils`
Simple, helpful tools for batch async calls to OpenAI APIs.

# Why does this exist?
Sometimes I have a need to make 100s of 1000s of independent calls to OpenAI APIs, mostly to the chat and embeddings APIs. The [python openai](https://github.com/openai/openai-python) library contains all the tools to make this possible, but setting things up for each project, test or experimentation can be cumbersome and may involve a good amount of boilerplate code.

Also, I sometimes feel the need to use a Jupyter notebook, and other times I prefer to use a Python script. There are some small differences in both scenarios when making async calls, due to the existence of an event loop in Jupyter notebooks. I prefer to have a single way to make such calls, no matter in which environment I am working, without having to worry about these details.

Thus I make this library, mostly for myself and to help me in my work, projects and expeirments. Perhaps othes might find it useful too. Feel free to open issues, propose changes, or contribute to the codebase.

# Installation
For now I am keeping this simple and will not push a package to PyPI, since it is a small "library" and essentially a wrapper around the OpenAI Python library. You can install it directly from GitHub:

```zsh
pip install git+https://github.com/JovanVeljanoski/openai-batch-utils
```
Alternatively, and for potential contributors, you can always clone the repository and install it locally:

First fork the repository to your GitHub account, then:

```zsh
git clone https://github.com/<your-username>/openai-batch-utils.git
cd openai-batch-utils
pip install -e ".[dev]"
```

# Usage

# Contributing
Be kind, respectful, helpful, and we will get along just fine! Feel free to open issues, propose changes, or contribute to the codebase. Open to ideas, suggestions, and improvements.

# License
MIT License
