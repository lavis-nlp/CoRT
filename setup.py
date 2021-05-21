from setuptools import setup

setup(
    name="CoRT",
    version="0.1.0",
    url="https://github.com/lavis-nlp/CoRT",
    license="",
    author="Marco Wrzalik",
    author_email="marco.wrzalik@hs-rm.de",
    description="CoRT - [Co]mplementary [R]ankings from [T]ransformers",
    packages= ["cort", "cort.tools"],
    install_requires=[
        "transformers",
        "torch",
        "pytorch-lightning==0.9.0",
        "msgpack==0.6.1",
        "numpy",
        "more_itertools",
        "tqdm",
    ],
    entry_points={
        "console_scripts": ["cort = cort.__main__:main"],
    },
)
