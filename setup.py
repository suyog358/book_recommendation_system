from setuptools import setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()


REPO_NAME = "Books-Recommendation-system-using-Machine-Learning"
AUTHER_USER_NAME = "entbappy"
SRC_REPO = "src"
LIST_OF_REQUIREMENTS = ["streamlit", "numpy"]

setup(
    name=SRC_REPO,
    version="0.0.1",
    author=AUTHER_USER_NAME,
    description="A small package for Books Recommendation system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author_email="ssblackgamerr@gamil.com",
    packages=[SRC_REPO],
    python_requires=">=3.12.0",
    install_reuries=LIST_OF_REQUIREMENTS
)
# -*- coding: utf-8 -*-
