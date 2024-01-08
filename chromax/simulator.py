"""Module containing simulator class."""
import logging
import random
from pathlib import Path
from typing import Callable, List, Optional, Union

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax._src.lib import xla_client as xc
from jaxtyping import Array, Float, Int

from . import functional
from .index_functions import conventional_index
from .trait_model import TraitModel
from .typing import Parents, Population


class Simulator:
    """Breeding simulator class. It can perform the most common operation of a breeding program.

    :param genetic_map: the path, or dataframe, containing the genetic map.
        It needs to have all the columns specified in trait_names, `CHR.PHYS`
        (with the name of the marker chromosome), and one between `cM` or `RecombRate`.
    :type genetic_map: Path or DataFrame
    :param trait_names: column names in the genetic_map.
        The values of the columns are the marker effects on the trait for each marker.
        The default value is `Yield`.
    :type trait_names: List of strings
    :param chr_column: name of the column containing the chromosome identifier.
        The default value is `CHR.PHYS`.
    :type chr_column: str
    :param position_column: name of the column containing the position in cM of the marker.
        The default value is `cM`.
    :type position_column: str
    :param recombination_column: name of the column containing the probability that a
        recombination happens before the current marker and after the previous one.
        The default value is `RecombRate`.
    :type recombination_column: str
    :param h2: narrow-sense heritability value for each trait.
        The default value is 0.5 for each trait.
    :type h2: array of float
    :param seed: the random seed for reproducibility.
    :type seed: int
    :param device: the device for computing simulations. It will be automatically selected if not
        specified; by default to the first available GPU or TPU, or the CPU if neither is present.
    :type device: XLA Device
    :param backend: the backend of the device.
        Common choices are `gpu`, `cpu` or `tpu`.
    :type backend: str or XLA client

    :Example:
        >>> from chromax import Simulator, sample_data
        >>> simulator = Simulator(genetic_map=sample_data.genetic_map)
        >>> f1 = simulator.load_population(sample_data.genome)
        >>> f2, _ = simulator.random_crosses(f1, n_crosses=10, n_offspring=20)
        >>> f2.shape
        (10, 20, 9839, 2)
    """

    def __init__(
        self,
        genetic_map: Union[Path, pd.DataFrame],
        trait_names: Optional[List[str]] = None,
        chr_column: str = "CHR.PHYS",
        position_column: str = "cM",
        recombination_column: str = "RecombRate",
        h2: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
        device: xc.Device = None,
        backend: Union[str, xc._xla.Client] = None,
    ):
        """Initialization method. See class docstring for information about parameters."""
        self.random_key = None
        if seed is None:
            seed = random.randint(0, 2**32)
        self.set_seed(seed)

        if device is None:  # use the right backend
            device = jax.local_devices(backend=backend)[0]
        elif isinstance(device, int):
            local_devices = jax.local_devices(backend=backend)
            matched_devices = filter(lambda d: d.id == device, local_devices)
            matched_devices = list(matched_devices)
            assert len(matched_devices) <= 1
            if len(matched_devices) == 0:
                print(jax.devices(), flush=True)
                logging.warning("No device with id: %i. Using the default one.", device)
                device = jax.local_devices(backend=backend)[0]
            else:
                device = matched_devices[0]
        self.device = device

        if not isinstance(genetic_map, pd.DataFrame):
            genetic_map = pd.read_table(genetic_map, sep="\t")
        if trait_names is None:
            other_col = {chr_column, position_column, recombination_column}
            trait_names = genetic_map.columns.drop(other_col, errors="ignore")
        self.trait_names = trait_names

        self.n_markers = len(genetic_map)
        chr_map = genetic_map[chr_column]
        self.chr_lens = chr_map.groupby(chr_map).count().values

        mrk_effects = genetic_map[self.trait_names]
        self.GEBV_model = TraitModel(
            marker_effects=mrk_effects.to_numpy(dtype=np.float32), device=self.device
        )

        if h2 is None:
            h2 = np.full((len(self.trait_names),), 0.5)
        h2 = np.asarray(h2)
        self.random_key, split_key = jax.random.split(self.random_key)
        env_effects = jax.random.normal(
            split_key, shape=(self.n_markers, len(self.trait_names))
        )
        target_vars = (1 - h2) / h2 * self.GEBV_model.var
        env_effects *= np.sqrt(2 * target_vars / self.n_markers)
        self.GxE_model = TraitModel(
            marker_effects=env_effects, offset=1, device=self.device
        )

        if recombination_column in genetic_map.columns:
            recombination_vec = genetic_map[recombination_column].to_numpy(
                dtype=np.float32
            )
            # change semantic to "recombine now" instead of "recombine after"
            recombination_vec[1:] = recombination_vec[:-1]
            self.cM = np.zeros(self.n_markers, dtype=np.float32)
            start_idx = 0
            for chr_len in self.chr_lens:
                end_idx = start_idx + chr_len
                self.cM[start_idx + 1 : end_idx] = (
                    recombination_vec[start_idx + 1 : end_idx].cumsum() * 100
                )
                start_idx = end_idx

        elif position_column in genetic_map.columns:
            self.cM = genetic_map[position_column].to_numpy(dtype=np.float32)
            recombination_vec = np.zeros(self.n_markers, dtype=np.float32)
            start_idx = 0
            for chr_len in self.chr_lens:
                end_idx = start_idx + chr_len
                recombination_vec[start_idx + 1 : end_idx] = (
                    self.cM[start_idx + 1 : end_idx] - self.cM[start_idx : end_idx - 1]
                )
                recombination_vec /= 100
        else:
            raise ValueError(
                f"One between {recombination_column} and {position_column} must be specified"
            )

        first_mrk_map = np.zeros(len(chr_map), dtype="bool")
        first_mrk_map[1:] = chr_map.iloc[1:].values != chr_map.iloc[:-1].values
        first_mrk_map[0] = True
        recombination_vec[first_mrk_map] = 0.5  # first equally likely
        self.recombination_vec = jax.device_put(recombination_vec, device=self.device)

    def set_seed(self, seed: int):
        """Set random seed for reproducibility.

        :param seed: random seed.
        :type seed: int
        """
        self.random_key = jax.random.PRNGKey(seed)

    def load_population(self, file_name: Union[Path, str]) -> Population["n"]:
        """Load a population from file.

        :param file_name: path of the file with the population genome.
        :type file_name: path

        :return: loaded population of shape (n, m, d), where
            n is the number of individual, m is the total number of marker,
            and d is the diploidy of the population.
        :rtype: ndarray

        :Example:
            >>> from chromax import Simulator, sample_data
            >>> simulator = Simulator(genetic_map=sample_data.genetic_map)
            >>> f1 = simulator.load_population(sample_data.genome)
            >>> f1.shape
            (371, 9839, 2)
        """
        population = np.load(file_name)
        return jax.device_put(population, device=self.device)

    def save_population(self, population: Population["n"], file_name: Union[Path, str]):
        """Save a population to file.

        :param population: population to save.
        :type population: ndarray
        :file_name: file path to save the population.
        :type file_name: path

        :Example:
            >>> from chromax import Simulator, sample_data
            >>> simulator = Simulator(genetic_map=sample_data.genetic_map)
            >>> f1 = simulator.load_population(sample_data.genome)
            >>> f2, _ = simulator.random_crosses(f1, n_crosses=10, n_offspring=20)
            >>> simulator.save_population(f2, "pop_file")
        """
        np.save(file_name, population, allow_pickle=False)

    def cross(self, parents: Parents["n"]) -> Population["n"]:
        """Main function that computes crosses from a list of parents.

        :param parents: parents to compute the cross. The shape of
            the parents is (n, 2, m, d), where n is the number of parents,
            m is the number of markers, and d is the ploidy.
        :type parents: ndarray


        :return: offspring population of shape (n, m, d).
        :rtype: ndarray

        :Example:
            >>> from chromax import Simulator, sample_data
            >>> import numpy as np
            >>> simulator = Simulator(genetic_map=sample_data.genetic_map)
            >>> f1 = simulator.load_population(sample_data.genome)
            >>> parents_indices = np.array([
                [1, 5],
                [4, 7],
                [5, 6]
            ])
            >>> parents = f1[parents_indices]
            >>> f2 = simulator.cross(parents)
            >>> f2.shape
            (3, 9839, 2)
        """
        self.random_key, split_key = jax.random.split(self.random_key)
        return functional.cross(parents, self.recombination_vec, split_key)

    @property
    def differentiable_cross_func(self) -> Callable:
        """Experimental features that return a differentiable version of the cross function.

        The differentiable crossing function takes as input:
         - population (array): starting population from which performing the crosses.
            The shape of the population is (n, m, d).
         - cross_weights (array): Array of shape (l, n, d). It is used to compute
            l crosses, starting from a weighted average of the n possible parents.
            When the n-axis has all zeros except of a single element equals to one,
            this function is equivalent to the cross function.
         - random_key (JAX random key): random key used for recombination sampling.

        And returns a population of shape (l, m, d).

        :Example:
            >>> from chromax import Simulator, sample_data
            >>> import numpy as np
            >>> import jax
            >>> simulator = Simulator(genetic_map=sample_data.genetic_map)
            >>> diff_cross = simulator.differentiable_cross_func
            >>> def mean_gebv(pop, weights, random_key):
                    new_pop = diff_cross(pop, weights, random_key)
                    return simulator.GEBV(new_pop, raw_array=True).mean()
            >>> grad_f = jax.grad(mean_gebv, argnums=1)
            >>> f1 = simulator.load_population(sample_data.genome)
            >>> weights = np.random.uniform(size=(10, len(f1), 2))
            >>> weights /= weights.sum(axis=1, keepdims=True)
            >>> random_key = jax.random.PRNGKey(42)
            >>> grad_value = grad_f(f1, weights, random_key)
            >>> grad_value.shape
            (10, 371, 2)
        """
        cross_haplo = jax.vmap(functional._meiosis, in_axes=(None, None, 0), out_axes=1)
        cross_individual = jax.vmap(cross_haplo, in_axes=(0, None, 0))
        cross_pop = jax.vmap(cross_individual, in_axes=(None, None, 0))

        @jax.jit
        def diff_cross_f(
            population: Population["n"],
            cross_weights: Float[Array, "m n 2"],
            random_key: jax.random.PRNGKeyArray,
        ) -> Population["m"]:
            population = population.reshape(*population.shape[:-1], -1, 2)
            keys_shape = len(cross_weights), len(population), 2, population.shape[-2]
            keys = jax.random.split(random_key, num=np.prod(keys_shape))
            keys = keys.reshape(*keys_shape, 2)
            outer_res = cross_pop(population, self.recombination_vec, keys)
            outer_res = outer_res.reshape(*outer_res.shape[:-2], -1)
            return (cross_weights[:, :, None, :] * outer_res).sum(axis=1)

        return diff_cross_f

    def double_haploid(
        self, population: Population["n"], n_offspring: int = 1
    ) -> Population["n n_offspring"]:
        """Computes the double haploid of the input population.

        :param population: input population of shape (n, m, 2).
        :type population: ndarray
        :param n_offspring: number of offspring per plant.
            The default value is 1.
        :type n_offspring: int

        :return: output population of shape (n, n_offspring, m, 2).
            This population will be homozygote.
        :rtype: ndarray

        :Example:
            >>> from chromax import Simulator, sample_data
            >>> simulator = Simulator(genetic_map=sample_data.genetic_map)
            >>> f1 = simulator.load_population(sample_data.genome)
            >>> dh = simulator.double_haploid(f1, n_offspring=10)
            >>> dh.shape
            (371, 10, 9839, 2)
        """
        self.random_key, split_key = jax.random.split(self.random_key)
        dh = functional.double_haploid(
            population, n_offspring, self.recombination_vec, split_key
        )

        if n_offspring == 1:
            dh = dh.squeeze(1)
        return dh

    def diallel(
        self, population: Population["n"], n_offspring: int = 1
    ) -> Population["n*(n-1)/2 n_offspring"]:
        """Diallel crossing function (crossing between every possible couple) except self-crossing.

        :param population: input population of shape (n, m, d).
        :type population: ndarray
        :param n_offspring: number of offspring per cross.
            The default value is 1.
        :type n_offspring: int

        :return: output population of shape (l, n_offspring, m, d),
            where l is the number of possible pair, i.e `n * (n-1) / 2`.
        :rtype: ndarray

        :Example:
            >>> from chromax import Simulator, sample_data
            >>> simulator = Simulator(genetic_map=sample_data.genetic_map)
            >>> f1 = simulator.load_population(sample_data.genome)[:10]
            >>> f2 = simulator.diallel(f1, n_offspring=10)
            >>> f2.shape
            (45, 10, 9839, 2)
        """
        if n_offspring < 1:
            raise ValueError("n_offspring must be higher or equal to 1")

        all_indices = jnp.arange(len(population))
        diallel_indices = self._diallel_indices(all_indices)
        cross_indices = jnp.repeat(diallel_indices, n_offspring, axis=0)
        out = self.cross(population[cross_indices])
        out = out.reshape(len(diallel_indices), n_offspring, *out.shape[1:])
        if n_offspring == 1:
            out = out.squeeze(1)
        return out

    def _diallel_indices(self, indices: Int[Array, "n"]) -> Int[Array, "n*(n-1)/2"]:
        triu_indices = jnp.triu_indices(len(indices), k=1)
        mesh1 = indices[triu_indices[0]]
        mesh2 = indices[triu_indices[1]]
        return jnp.stack([mesh1, mesh2], axis=1)

    def random_crosses(
        self, population: Population["n"], n_crosses: int, n_offspring: int = 1
    ) -> Population["n_crosses n_offspring"]:
        """Computes random crosses on a population.

        :param population: input population of shape (n, m, d).
        :type population: ndarray
        :param n_crosses: number of random crosses to perform.
        :type n_crosses: int
        :param n_offspring: number of offspring per cross.
            The default value is 1.
        :type n_offspring: int

        :return: output population of shape (n_crosses, n_offspring, m, d)
            and parent indices of shape (n_crosses, 2) of performed crosses.
        :rtype: tuple of two ndarrays

        :Example:
            >>> from chromax import Simulator, sample_data
            >>> simulator = Simulator(genetic_map=sample_data.genetic_map)
            >>> f1 = simulator.load_population(sample_data.genome)
            >>> f2, parent_ids = simulator.random_crosses(f1, 100, n_offspring=10)
            >>> f2.shape
            (100, 10, 9839, 2)
            >>> parent_ids.shape
            (100, 2)
        """
        all_indices = np.arange(len(population))
        diallel_indices = self._diallel_indices(all_indices)
        if n_crosses > len(diallel_indices):
            raise ValueError("n_crosses can be at most the diallel length")

        self.random_key, split_key = jax.random.split(self.random_key)
        random_select_idx = jax.random.choice(
            split_key, len(diallel_indices), shape=(n_crosses,), replace=False
        )
        parent_indices = diallel_indices[random_select_idx]

        cross_indices = np.repeat(parent_indices, n_offspring, axis=0)
        out = self.cross(population[cross_indices])
        out = out.reshape(n_crosses, n_offspring, *out.shape[1:])
        if n_offspring == 1:
            out = out.squeeze(1)
        return out, parent_indices

    def select(
        self,
        population: Population["_g n"],
        k: int,
        f_index: Optional[Callable[[Population["n"]], Float[Array, "n"]]] = None,
    ) -> Population["_g k"]:
        """Function to select individuals based on their score (index).

        :param population: input population of shape (n, m, d),
            or shape (g, n, m, d), to select k individual from each group population group g.
        :type population: ndarray
        :param k: number of individual to select.
        :type k: int
        :param f_index: function that computes a score from each individual.
            The function accepts as input the population, i.e. and array of shape
            (n, m, d) and returns a n float numbers. The default f_index is the conventional index,
            i.e. the sum of the marker effects masked with the SNPs from the genetic_map.
        :type f_index: Callable

        :return: output population of shape (k, m, d) or (g, k, m, d),
            depending on the input population.
        :rtype: ndarray

        :Example:
            >>> from chromax import Simulator, sample_data
            >>> simulator = Simulator(genetic_map=sample_data.genetic_map, trait_names=["Yield"])
            >>> f1 = simulator.load_population(sample_data.genome)
            >>> len(f1), simulator.GEBV(f1).mean().values
            (371, [8.223844])
            >>> f2 = simulator.select(f1, k=20)
            >>> len(f2), simulator.GEBV(f2).mean().values
            (20, [14.595136])
        """
        if f_index is None:
            f_index = conventional_index(self.GEBV_model)

        if len(population.shape) == 3:
            select_f = functional.select
        elif len(population.shape) == 4:
            select_f = jax.vmap(functional.select, in_axes=(0, None, None))
        else:
            raise ValueError(
                f"Unexpected shape {population.shape} for input population"
            )

        return select_f(population, k, f_index)

    def GEBV(
        self, population: Population["n"], *, raw_array: bool = False
    ) -> Union[pd.DataFrame, np.ndarray]:
        """Computes the Genomic Estimated Breeding Values using the data from the genetic_map.

        :param population: input population of shape (n, m, d).
        :type population: ndarray
        :param raw_array: whether to return a raw array or a DataFrame.
            Default value is False.
        :type raw_array: bool

        :return: a DataFrame (or array) with n rows and a column for each trait.
            It contains the GEBV of each trait for each individual.
        :rtype: DataFrame or ndarray

        :Example:
            >>> from chromax import Simulator, sample_data
            >>> simulator = Simulator(genetic_map=sample_data.genetic_map)
            >>> f1 = simulator.load_population(sample_data.genome)
            >>> simulator.GEBV(f1).mean()
            Heading Date              0.196119
            Protein Content          -0.228718
            Plant Height             -5.888406
            Thousand Kernel Weight   -1.029418
            Yield                     8.223843
            Fusarium Head Blight      5.318052
            Spike Emergence Period   -0.933169
            dtype: float32
        """
        GEBV = self.GEBV_model(population)
        if not raw_array:
            GEBV = pd.DataFrame(GEBV, columns=self.trait_names)
        return GEBV

    def create_environments(
        self, num_environments: int
    ) -> Float[Array, "num_environments"]:
        """Create environments to phenotype the population.

        In practice, it generates random numbers from a normal distribution.

        :param num_environments: number of environments to create.
        :type num_environments: int

        :return: array of floating point numbers.
            This output can be used for the function `phenotype`.
        :rtype: ndarray
        """
        self.random_key, split_key = jax.random.split(self.random_key)
        return jax.random.normal(
            key=split_key,
            shape=(num_environments,),
        )

    def phenotype(
        self,
        population: Population["n"],
        *,
        num_environments: Optional[int] = None,
        environments: Optional[np.ndarray] = None,
        raw_array: bool = False,
    ) -> Union[pd.DataFrame, np.ndarray]:
        """Simulates the phenotype of a population.

        This uses the Genotype-by-Environment model described in `AlphaSimR
        <https://cran.r-project.org/web/packages/AlphaSimR/vignettes/traits.pdf>`_.

        :param population: input population of shape (n, m, d)
        :type population: ndarray
        :param num_environments: number of environments to test the population.
            Default value is 1.
        :type num_environments: int
        :param environments: environments to test the population. Each environment
            must be represented by a floating number in the range (-1, 1).
            When drawing new environments use normal distribution to maintain
            heretability semantics.
        :type environments: ndarray
        :param raw_array: whether to return a raw array or a DataFrame.
            Default value is False.
        :type raw_array: bool

        :return: a DataFrame (or array) with n rows and a column for each trait.
            It contains the simulated phenotype for each individual.
        :rtype: DataFrame or ndarray

        :Example:
            >>> from chromax import Simulator, sample_data
            >>> simulator = Simulator(genetic_map=sample_data.genetic_map, seed=42)
            >>> f1 = simulator.load_population(sample_data.genome)
            >>> envs = simulator.create_environments(4)
            >>> simulator.phenotype(f1, environments=envs).mean()
            Heading Date              0.105397
            Protein Content          -0.172026
            Plant Height             -5.813669
            Thousand Kernel Weight   -1.372738
            Yield                     8.306302
            Fusarium Head Blight      4.286477
            Spike Emergence Period   -0.575061
            dtype: float32
        """
        if num_environments is not None and environments is not None:
            raise ValueError(
                "You cannot specify both num_environments and environments"
            )
        if environments is None:
            num_environments = num_environments if num_environments is not None else 1
            environments = self.create_environments(num_environments)

        w = jnp.mean(environments)
        GEBV = self.GEBV_model(population)
        GxE = self.GxE_model(population)
        phenotype = GEBV + w * GxE
        if not raw_array:
            phenotype = pd.DataFrame(phenotype, columns=self.trait_names)
        return phenotype

    def corrcoef(self, population: Population["n"]) -> Float[Array, "n"]:
        """Computes the correlation coefficient of the population against its centroid.

        It can be used as an indicator of variance in the population.

        :param population: input population of shape (n, m, d)
        :type population: ndarray

        :return: vector of length n, containing the correlation coefficient
            of each individual against the average of the population.
        :rtype: ndarray

        :Example:
            >>> from chromax import Simulator, sample_data
            >>> simulator = Simulator(genetic_map=sample_data.genetic_map, seed=42)
            >>> f1 = simulator.load_population(sample_data.genome)
            >>> corrcoef = simulator.corrcoef(f1)
            >>> corrcoef.shape
            (371,)
        """
        monoploid_enc = population.reshape(population.shape[0], -1)
        mean_pop = jnp.mean(monoploid_enc, axis=0)
        pop_with_centroid = jnp.vstack([mean_pop, monoploid_enc])
        corrcoef = jnp.corrcoef(pop_with_centroid)
        return corrcoef[0, 1:]
