from setuptools import setup, find_packages


setup(
    name="deepwaveform",
    version="0.1",
    packages=find_packages(),

    author="Hans Harder",
    author_email="hans.harder@mailbox.tu-dresden.de",

    install_requires=[
        "matplotlib",
        "torch",
        "numpy"
    ]
)
