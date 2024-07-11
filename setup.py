from setuptools import setup

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="q-galore-torch",
    version="1.0",
    description="Q-GaLore: Quantized GaLore with INT4 Projection and Layer-Adaptive Low-Rank Gradients",
    url="https://github.com/VITA-Group/Q-GaLore",
    author="Zhenyu Zhang",
    author_email="zhenyu.zhang@utexas.edu",
    license="Apache 2.0",
    packages=["q_galore_torch"],
    install_requires=required,
)