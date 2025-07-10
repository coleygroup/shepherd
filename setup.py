"""
Minimal setup.py for backward compatibility.
For modern package configuration, see pyproject.toml.
"""
import setuptools

REQUIREMENTS = [
    'open3d>=0.18'
]

if __name__ == "__main__":
    setuptools.setup(
        name="shepherd",
        package_dir={"": "src"},
        packages=setuptools.find_packages(where="src"),
        install_requires=REQUIREMENTS,
        version="0.2.1",
        author="Keir Adams",
        author_email="keir@keiradams.com",
        description="ShEPhERD: Diffusing Shape, Electrostatics, and Pharmacophores for Drug Design.",
        url="https://github.com/coleygroup/shepherd",
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
        ],
        python_requires='>=3.8,<3.12',
    )
