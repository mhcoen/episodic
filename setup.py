from setuptools import setup, find_packages

setup(
    name="episodic",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["openai", "networkx", "prompt_toolkit", "pygments", "plotly", "flask", "numpy", "pywebview", "litellm", "typer"],
    entry_points={
        "console_scripts": [
            "episodic=episodic.__main__:main",
        ],
    },
    author="Michael H. Coen",
    author_email="mhcoen@gmail.com",
    description="A conversational memory system that creates persistent, navigable conversations with LLMs",
    keywords="episodic, llm, conversation, memory, rag, ai, chatbot",
    url="https://github.com/mhcoen/episodic",
    license="Apache-2.0",
    python_requires=">=3.6",
)
