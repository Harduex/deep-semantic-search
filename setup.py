import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Main dependencies required for the package
dependencies = [
    "sentence-transformers>=2.7.0",
    "torch>=2.0.0",
    "transformers>=4.30.0",
    "faiss-cpu>=1.8.0",
    "pandas>=2.0.0",
    "numpy>=1.22.0",
    "matplotlib>=3.5.0",
    "tqdm>=4.60.0",
    "beautifulsoup4>=4.9.0",
    "kmeans-pytorch>=0.3",
    "langchain>=0.1.0",
    "langchain-community>=0.0.1",
    "pillow>=10.0.0",
]

setuptools.setup(
    name="deep-semantic-search",
    version="0.1.1",
    author="Harduex",
    author_email="simeon.simka@gmail.com",
    description="A library for embedding, indexing, and applying semantic search for text and image data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Harduex/deep-semantic-search",
    project_urls={
        "Bug Tracker": "https://github.com/Harduex/deep-semantic-search/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Indexing",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=dependencies,
)
