from pathlib import Path
from typing import Callable, List, Optional, Union
import pandas as pd
from . import functional
from .trait_model import TraitModel
from .typing import Parents, Population
from .index_functions import conventional_index
import numpy as np
import jax
import jax.numpy as jnp
from jax._src.lib import xla_client as xc
from jaxtyping import Array, Float, Int
import random
import logging


class Simulator:
    """Breeding simulator class. It can perform the most common operation of a breeding program.

    Args:
     - genetic_map (Path or DataFrame): the path, or dataframe, containing the genetic map.
        It needs to have all the columns specified in trait_names, `CHR.PHYS` 
        (with the name of the marker chromosome), and one between `cM` or `RecombRate`.
     - trait_names (List of strings): Column names in the genetic_map. 
        The values of the columns are the marker effects on the trait for each marker.
        The default value is `Yield`.
     - seed (int, optional): The random seed for reproducibility.
     - device (XLA Device, optional): the device on which to run the simulations. 
        If not specified, the default device will be chosen.
     - backend: (str or XLA client): the backend of the device. 
        Common choices are `gpu`, `cpu` or `tpu`.

    Example:
    >>> from chromax import Simulator
    >>> simulator = Simulator(genetic_map='path_to_genetic_map.csv')
    >>> f1 = simulator.load_population('path_to_genome.txt')
    >>> f2 = simulator.random_crosses(f1, n_crosses=10, n_offspring=20)
    >>> len(f2)
    200
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
        backend: Union[str, xc._xla.Client] = None
    ):
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
                logging.warning(
                    "No device with id: %i. Using the default one.", device
                )
                device = jax.local_devices(backend=backend)[0]
            else:
                device = matched_devices[0]
        self.device = device

        if not isinstance(genetic_map, pd.DataFrame):
            genetic_map = pd.read_table(genetic_map, sep="\t")
        if trait_names is None:
            other_col = {chr_column, position_column, recombination_column}
            trait_names = genetic_map.columns.drop(other_col, errors='ignore')
        self.trait_names = trait_names

        self.n_markers = len(genetic_map)
        chr_map = genetic_map[chr_column]
        self.chr_lens = chr_map.groupby(chr_map).count().values

        mrk_effects = genetic_map[self.trait_names]
        self.GEBV_model = TraitModel(
            marker_effects=mrk_effects.to_numpy(dtype=np.float32),
            device=self.device
        )

        if h2 is None:
            h2 = np.full((len(self.trait_names),), 0.5)
        self.random_key, split_key = jax.random.split(self.random_key)
        env_effects = jax.random.normal(split_key, shape=(self.n_markers, len(self.trait_names)))
        target_vars = (1 - h2) / h2 * self.GEBV_model.var
        env_effects *= target_vars * 2 / self.n_markers
        self.GxE_model = TraitModel(
            marker_effects=env_effects,
            mean=1,
            device=self.device
        )

        if recombination_column in genetic_map.columns:
            recombination_vec = genetic_map[recombination_column].to_numpy(dtype=np.float32)
            # change semantic to "recombine now" instead of "recombine after"
            recombination_vec[1:] = recombination_vec[:-1]
            self.cM = np.zeros(self.n_markers, dtype=np.float32)
            start_idx = 0
            for chr_len in self.chr_lens:
                end_idx = start_idx + chr_len
                self.cM[start_idx + 1:end_idx] = recombination_vec[start_idx + 1:end_idx].cumsum() * 100
                start_idx = end_idx

        elif position_column in genetic_map.columns:
            self.cM = genetic_map[position_column].to_numpy(dtype=np.float32)
            recombination_vec = np.zeros(self.n_markers, dtype=np.float32)
            start_idx = 0
            for chr_len in self.chr_lens:
                end_idx = start_idx + chr_len
                recombination_vec[start_idx + 1:end_idx] = self.cM[start_idx + 1:end_idx] - self.cM[start_idx:end_idx - 1]
                recombination_vec /= 100
        else:
            raise ValueError(
                f"One between {recombination_column} and {position_column} must be specified"
            )

        first_mrk_map = np.zeros(len(chr_map), dtype='bool')
        first_mrk_map[1:] = chr_map.iloc[1:].values != chr_map.iloc[:-1].values
        first_mrk_map[0] = True
        recombination_vec[first_mrk_map] = 0.5  # first equally likely
        self.recombination_vec = jax.device_put(
            recombination_vec,
            device=self.device
        )

    def set_seed(self, seed: int):
        """
        Set random seed for reproducibility.

        Args:
         - seed (int): random seed.
        """
        self.random_key = jax.random.PRNGKey(seed)

    def load_population(self, file_name: Union[Path, str]) -> Population["n"]:
        """Load a population from file.

        Args:
         - file_name (path): path of the file with the population genome.

        Returns:
         - population (array): loaded population of shape (n, m, d), where
            n is the number of individual, m is the total number of marker,
            and d is the diploidy of the population.
        """
        population = np.load(file_name)
        return jax.device_put(population, device=self.device)

    def save_population(self, population: Population["n"], file_name: Union[Path, str]):
        """Save a population to file.

        Args:
         - population (array): population to save.
         - file_name (path): file path to save the population.
        """
        np.save(file_name, population, allow_pickle=False)

    def cross(self, parents: Parents["n"]) -> Population["n"]:
        """Main function that computes crosses from a list of parents.

        Args:
         - parents (array): parents to compute the cross. The shape of
          the parents is (n, 2, m, 2), where n is the number of parents
          and m is the number of markers.

        Returns:
         - population (array): offspring population of shape (n, m, 2).
        """
        keys = jax.random.split(self.random_key, num=len(parents) * 2 + 1)
        self.random_key = keys[0]
        split_keys = keys[1:].reshape(len(parents), 2, 2)
        return functional.cross(
            parents,
            self.recombination_vec,
            split_keys
        )

    @property
    def differentiable_cross_func(self) -> Callable:
        """Experimental features that return a differentiable version
        of the cross function.

        The differentiable crossing function takes as input:
         - population (array): starting population from which performing the crosses.
            The shape of the population is (n, m, 2).
         - cross_weights (array): Array of shape (l, n, 2). It is used to compute
            l crosses, starting from a weighted average of the n possible parents. 
            When the n-axis has all zeros except of a single element equals to one,
            this function is equivalent to the cross function.
         - random_key (JAX random key): random key used for recombination sampling.

        And returns a population of shape (l, m, 2).
        """

        cross_haplo = jax.vmap(
            functional._cross_individual,
            in_axes=(None, None, 0),
            out_axes=1
        )
        cross_individual = jax.vmap(cross_haplo, in_axes=(0, None, 0))
        cross_pop = jax.vmap(cross_individual, in_axes=(None, None, 0))

        @jax.jit
        def diff_cross_f(
            population: Population["n"],
            cross_weights: Float[Array, "m n 2"],
            random_key: jax.random.PRNGKeyArray
        ) -> Population["m"]:
            num_keys = len(cross_weights) * len(population) * 2
            keys = jax.random.split(random_key, num=num_keys)
            keys = keys.reshape(len(cross_weights), len(population), 2, 2)
            outer_res = cross_pop(population, self.recombination_vec, keys)
            return (cross_weights[:, :, None, :] * outer_res).sum(axis=1)

        return diff_cross_f

    def double_haploid(
        self,
        population: Population["n"],
        n_offspring: int = 1
    ) -> Population["n n_offspring"]:
        """Computes the double haploid of the input population.

        Args:
         - population (array): input population of shape (n, m, 2).
         - n_offspring (int): number of offspring per plant.
            The default value is 1.

        Returns:
         - population (array): output population of shape (n, n_offspring, m, 2).
            This population will be homozygote.
        """
        self.random_key, split_key = jax.random.split(self.random_key)
        dh = functional.double_haploid(
            population,
            n_offspring,
            self.recombination_vec,
            split_key
        )

        if n_offspring == 1:
            dh = dh.squeeze(1)
        return dh

    def diallel(
        self,
        population: Population["n"],
        n_offspring: int = 1
    ) -> Population["n*(n-1)/2 n_offspring"]:
        """Diallel crossing function, i.e. crossing between every possible
        couple, except self-crossing.

        Args:
         - population (array): input population of shape (n, m, 2).
         - n_offspring (int): number of offspring per cross.
            The default value is 1.

        Returns:
         - population (array): output population of shape (l, n_offspring, m, 2),
            where l is the number of possible pair, i.e `n * (n-1) / 2`.
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

    def _diallel_indices(
        self,
        indices: Int[Array, "n"]
    ) -> Int[Array, "n*(n-1)/2"]:
        triu_indices = jnp.triu_indices(len(indices), k=1)
        mesh1 = indices[triu_indices[0]]
        mesh2 = indices[triu_indices[1]]
        return jnp.stack([mesh1, mesh2], axis=1)

    def random_crosses(
        self,
        population: Population["n"],
        n_crosses: int,
        n_offspring: int = 1
    ) -> Population["n_crosses n_offspring"]:
        """Computes random crosses on a population.

        Args:
         - population (array): input population of shape (n, m, 2).
         - n_crosses (int): number of random crosses to perform.
         - n_offspring (int): number of offspring per cross. 
            The default value is 1.

        Returns:
         - population (array): output population of shape (n_crosses, n_offspring, m, 2).
        """

        if n_crosses < 1:
            raise ValueError("n_crosses must be higher or equal to 1")
        if n_offspring < 1:
            raise ValueError("n_offspring must be higher or equal to 1")

        all_indices = np.arange(len(population))
        diallel_indices = self._diallel_indices(all_indices)
        if n_crosses > len(diallel_indices):
            raise ValueError("n_crosses can be at most the diallel length")

        self.random_key, split_key = jax.random.split(self.random_key)
        random_select_idx = jax.random.choice(
            split_key,
            len(diallel_indices),
            shape=(n_crosses,),
            replace=False
        )
        cross_indices = diallel_indices[random_select_idx]

        cross_indices = np.repeat(cross_indices, n_offspring, axis=0)
        out = self.cross(population[cross_indices])
        out = out.reshape(n_crosses, n_offspring, *out.shape[1:])
        if n_offspring == 1:
            out = out.squeeze(1)
        return out

    def select(
        self,
        population: Population["_g n"],
        k: int,
        f_index: Optional[Callable[[Population["n"]], Float[Array, "n"]]] = None
    ) -> Population["_g k"]:
        """Function to select individuals based on their score (index).

        Args:
         - population (array): input population of shape (n, m, 2), 
            or shape (g, n, m, 2), to select k individual from each group population group g.
         - k (int): number of individual to select.
         - f_index (function): function that computes a score from each individual.
          The function accepts as input the population, i.e. and array of shape
          (n, m, 2) and returns a n float numbers. The default f_index is the conventional index, 
          i.e. the sum of the marker effects masked with the SNPs from the genetic_map.

        Returns:
         - population (array): output population of shape (k, m, 2) or (g, k, m, 2), 
            depending on the input population.
        """
        if f_index is None:
            f_index = conventional_index(self.GEBV_model)

        if len(population.shape) == 3:
            select_f = functional.select
        elif len(population.shape) == 4:
            select_f = jax.vmap(functional.select, in_axes=(0, None, None))
        else:
            raise ValueError(f"Unexpected shape {population.shape} for input population")

        return select_f(population, k, f_index)

    def GEBV(
        self,
        population: Population["n"]
    ) -> pd.DataFrame:
        """Computes the Genomic Estimated Breeding Values using the
        marker effects from the genetic_map.

        Args:
         - population (array): input population of shape (n, m, 2).

        Returns:
         - gebv (DataFrame): a DataFrame with n rows and a column for each trait.
            It contains the GEBV of each trait for each individual.
        """
        GEBV = self.GEBV_model(population)
        return pd.DataFrame(GEBV, columns=self.trait_names)

    def create_environments(
        self,
        num_environments: int
    ) -> Float[Array, "num_environments"]:
        """Create environments to phenotype the population.
        In practice, it generates random numbers from a normal distribution.

        Args:
         - num_environments (int): number of environments to create.

        Returns:
         - environments (array): array of floating point numbers.
            This output can be used for the function `phenotype`.
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
        environments: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """Simulates the phenotype of a population.
        This uses the Genotype-by-Environment model described in the following:
        https://cran.r-project.org/web/packages/AlphaSimR/vignettes/traits.pdf

        Args:
         - population (array): input population of shape (n, m, 2)
         - num_environments (int): number of environments to test the population.
            Default value is 1.
         - environments (array): environments to test the population. Each environment
            must be represented by a floating number in the range (-1, 1).
            When drawing new environments use normal distribution to mantain
            heretability semantics.

        Returns:
         - phenotype (DataFrame): a DataFrame with n rows and a column for each trait.
            It contains the simulated phenotype for each individual.
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
        return pd.DataFrame(phenotype, columns=self.trait_names)

    def corrcoef(
        self,
        population: Population["n"]
    ) -> Float[Array, "n"]:
        """Computes the correlation coefficient of the population against its centroid.
        It can be used as an indicator of variance in the population.

        Args:
         - population (array): input population of shape (n, m, 2)

        Returns:
         - corrcoefs (array): vector of length n, containing the correlation coefficient
            of each individual againts the average of the population.
        """
        monoploid_enc = population.reshape(population.shape[0], -1)
        mean_pop = jnp.mean(monoploid_enc, axis=0)
        pop_with_centroid = jnp.vstack([mean_pop, monoploid_enc])
        corrcoef = jnp.corrcoef(pop_with_centroid)
        return corrcoef[0, 1:]
