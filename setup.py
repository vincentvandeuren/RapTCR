from setuptools import setup, find_packages
import versioneer

# conda package requirements
requirements = [
    "python==3.10",
    "pip",
    "numpy",
    "pandas",
    "matplotlib",
    "seaborn",
    "colorcet",
    "faiss-cpu",
]

setup(
    name="RapTCR",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="Rapid TCR repertoire visualization and annotation.",
    license="Proprietary",
    author="Vincent Van Deuren",
    author_email="vincent.vandeuren@uantwerpen.be",
    url="https://github.com/vincentvandeuren/RapTCR",
    packages=find_packages(),
    #package_data={"RapTCR": ["data/bg_position_matrices.pkl","modules/olga/*",]}, !TODO not forget
    #include_package_data=True,
    install_requires=requirements,
    keywords="",
    classifiers=[
        "Programming Language :: Python :: 3.10",
    ],
)
