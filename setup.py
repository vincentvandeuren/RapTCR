from setuptools import setup, find_packages
import versioneer

# conda package requirements
requirements = [
    "python",
    "pip",
    "numpy",
    "pandas",
    "matplotlib",
    "seaborn",
    "colorcet",
    "mmh3",
    "faerun",
    "faiss-cpu",
    "tqdm",
]

setup(
    name="TCRexplore",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="cluster-based visualization of T-cell receptor repertoires",
    license="Proprietary",
    author="Vincent Van Deuren",
    author_email="vincent.vandeuren@uantwerpen.be",
    url="https://github.com/vincentvandeuren/tcrexplore",
    packages=find_packages(),
    package_data={
        "tcrexplore": [
            "data/bg_position_matrices.pkl",
            "modules/olga/*",
        ]
    },
    include_package_data=True,
    install_requires=requirements,
    keywords="",
    classifiers=[
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
)
