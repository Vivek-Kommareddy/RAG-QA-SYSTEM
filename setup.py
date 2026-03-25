from setuptools import setup, find_packages

setup(
    name="rag-qa-system",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "fastapi",
        "uvicorn[standard]",
        "pydantic-settings",
        "jinja2",
        "PyMuPDF",
        "python-docx",
        "chromadb",
        "sentence-transformers",
        "numpy",
        "scikit-learn",
        "openai",
        "anthropic",
        "streamlit",
        "httpx",
    ],
    python_requires=">=3.11",
)