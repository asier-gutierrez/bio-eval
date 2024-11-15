
import setuptools


setuptools.setup(
    name="bioeval",
    version='0.0.1',
    description="Biomedical LM evaluation framework.",
    author="Asier Gutiérrez-Fandiño",
    author_email="asierguti96@gmail.com",
    packages=setuptools.find_packages(
        exclude=["*.test", "*.test.*", "test.*", "test", "tests", "*.ipynb"]
    ),
    python_requires='>=3.11',
)