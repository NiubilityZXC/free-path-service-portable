from setuptools import find_namespace_packages, setup


setup(
    name="cli-anything-flash",
    version="0.1.0",
    description="Agent-friendly CLI harness for FLASH simulations",
    python_requires=">=3.10",
    packages=find_namespace_packages(include=["cli_anything.*"]),
    install_requires=[
        "click>=8.0",
        "numpy>=1.20",
        "matplotlib>=3.5",
        "h5py>=3.0",
    ],
    package_data={
        "cli_anything.flash": ["skills/*.md"],
    },
    entry_points={
        "console_scripts": [
            "cli-anything-flash=cli_anything.flash.flash_cli:main",
        ],
    },
)
