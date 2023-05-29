import setuptools

install_requires = [
    'numpy',
    'pandas',
    'jax',
    'jaxlib',
    'jaxtyping'
]


setuptools.setup(
    name='chromax',
    version='0.0.2a',
    description='Breeding simulator based on JAX',
    url='https://github.com/younik/chromax',
    author='Omar Younis',
    author_email='omar.younis98@gmail.com',
    license='bsd-3-clause',
    keywords=['Breeding', 'simulator', 'JAX', 'chromosome', 'genetics', 'bioinformatics'],
    packages=setuptools.find_packages(),
    python_requires=">=3.8",
    include_package_data=True,
    install_requires=install_requires,
    project_urls={
        "Code": "https://github.com/kora-labs/chromax",
        "Documentation": "https://chromax.readthedocs.io/",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
