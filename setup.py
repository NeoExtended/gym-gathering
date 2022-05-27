from setuptools import setup, find_packages

__version__ = "1.0.4"

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="gym_gathering",
    packages=[
        package for package in find_packages() if package.startswith("gym_gathering")
    ],
    version=__version__,
    author="Matthias Konitzny",
    author_email="konitzny@ibr.cs.tu-bs.de",
    url="https://github.com/NeoExtended/gym-gathering",
    description="A gym-compatible reinforcement learning environment which simulates various particle gathering "
    "problems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords="reinforcement-learning machine-learning gym environment python "
    "data-science particle-gathering gathering particles algorithms",
    install_requires=["gym", "numpy", "scipy", "opencv-python"],
    extras_require={"keyboard": ["keyboard"]},
    python_requires=">=3.7",
    #    include_package_data=True,
    package_data={"gym_gathering": ["mapdata/*.csv"]},
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
