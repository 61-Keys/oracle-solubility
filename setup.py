from setuptools import setup, find_packages

setup(
    name="oracle-solubility",
    version="1.0.0",
    author="Asutosh Rath",
    description="Predict protein solubility in E. coli using deep learning",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/61-Keys/oracle-solubility",
    packages=find_packages(),
    include_package_data=True,
    package_data={'oracle': ['data/*.pt', 'data/*.pkl']},
    install_requires=["torch>=2.0.0", "numpy>=1.21.0", "scikit-learn>=1.0.0", "matplotlib>=3.5.0"],
    entry_points={'console_scripts': ['oracle=oracle.cli:main']},
    python_requires=">=3.8",
)
