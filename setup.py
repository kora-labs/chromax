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
    version='0.0.1a',
    description='Breeding simulator based on JAX',
    url='https://github.com/younik/chromax',
    author='Omar Younis',
    author_email='omar.younis98@gmail.com',
    keywords='Breeding, simulator, JAX, chromosome, genetic, bioinformatic',
    packages=setuptools.find_packages(),
    python_requires=">=3.8",
    include_package_data=True,
    install_requires=install_requires,
)
