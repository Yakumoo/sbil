
from setuptools import find_packages, setup


setup(
    name="sbil",
    packages=[package for package in find_packages() if package.startswith("sbil")],
    #package_data={"stable_baselines3": ["py.typed", "version.txt"]},
    install_requires=[
        "stable_baselines3>=1.1",

    ],
    extras_require={
        "tests": [
            # Run tests and coverage
            "pytest",
            "pytest-cov",
            "pytest-env",
            "pytest-xdist",
            # Type check
            "pytype",
            # Lint code
            "flake8>=3.8",
            # Find likely bugs
            "flake8-bugbear",
            # Sort imports
            "isort>=5.0",
            # Reformat
            "black",
        ],
        "docs": [
            "sphinx",
            "sphinx-autobuild",
            "sphinx-rtd-theme",
            # For spelling
            "sphinxcontrib.spelling",
            # Type hints support
            "sphinx-autodoc-typehints",
        ],
        "extra": [
            # For render
            "opencv-python",
            # For atari games,
            "atari_py~=0.2.0",
            "pillow",
            # Tensorboard support
            "tensorboard>=2.2.0",
            # Checking memory taken by replay buffer
            "psutil",
        ],
    },
    description="Imitation learning algorithms based on the Stable Baselines framework",
    author="",
    url="https://github.com/Yakumoo/sbil",
    author_email="",
    keywords="reinforcement-learning-algorithms reinforcement-learning machine-learning imitation-learning"
    "gym openai stable baselines toolbox python data-science",
    license="MIT",
    long_description="",
    long_description_content_type="text/markdown",
    version="",
)
