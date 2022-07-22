import setuptools


requirements = []
with open("requirements.txt", "r") as f:
    for line in f:
        requirements.append(line.strip())


setuptools.setup(
    name="meta-learn-tpe",
    version="0.0.1",
    author="nabenabe0928",
    author_email="shuhei.watanabe.utokyo@gmail.com",
    url="https://github.com/nabenabe0928/meta-learn-tpe",
    packages=["tpe", "tpe.optimizer", "tpe.utils"],
    python_requires=">=3.8",
    platforms=["Linux"],
    install_requires=requirements,
    include_package_data=True,
)
