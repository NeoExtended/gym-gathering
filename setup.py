from setuptools import setup, find_packages

setup(
    name="gym_gathering",
    packages=[package for package in find_packages() if package.startswith("gym_gathering")],
    version="1.0.0",
    author="Matthias Konitzny",
    author_email="konitzny@ibr.cs.tu-bs.de",
    description="A gym-compatible reinforcement learning environment which simulates various particle gathering "
                "problems",
    keywords="reinforcement-learning machine-learning gym environment python "
             "data-science particle-gathering gathering particles algorithms",
    install_requires=["gym", "numpy", "scipy", "opencv-python"],
    extras_require={"keyboard": ["keyboard"]},
#    include_package_data=True,
    package_data={"gym_gathering": ["mapdata/*.csv"]},
)
