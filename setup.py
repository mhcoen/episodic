from setuptools import setup, find_packages

setup(
    name="episodic",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[],
    entry_points={
        "console_scripts": [
            "episodic=episodic.__main__:main",
        ],
    },
    author="Example Author",
    author_email="example@example.com",
    description="A simple episodic database application",
    keywords="episodic, database",
    python_requires=">=3.6",
)