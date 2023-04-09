from pathlib import Path
from typing import Callable, List, Optional, Union
import pandas as pd
from .gebv_model import GEBVModel
from .typing import N_MARKERS, Haploid, Individual, Parents, Population
from .index_functions import conventional_index
import numpy as np
import jax
import jax.numpy as jnp
from jax._src.lib import xla_client as xc
from jaxtyping import Array, Float, Int
from functools import partial
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
        trait_names: List["str"] = ["Yield"],
        seed: Optional[int] = None,
        device: xc.Device = None,
        backend: Union[str, xc._xla.Client] = None
    ):
        self.trait_names = trait_names

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
                    f"No device with id:{device}. Using the default one."
                )
                device = jax.local_devices(backend=backend)[0]
            else:
                device = matched_devices[0]
        self.device = device

        if not isinstance(genetic_map, pd.DataFrame):
            types = {name: 'float32' for name in trait_names}
            types['cM'] = 'float32'
            types['RecombRate'] = 'float32'
            genetic_map = pd.read_table(genetic_map, sep="\t", dtype=types)

        self.n_markers = len(genetic_map)
        chr_map = genetic_map['CHR.PHYS']
        self.chr_lens = chr_map.groupby(chr_map).count().values

        mrk_effects = genetic_map[trait_names]
        self.GEBV_model = GEBVModel(
            marker_effects=mrk_effects.to_numpy(),
            device=self.device
        )

        if "RecombRate" in genetic_map.columns:
            recombination_vec = genetic_map["RecombRate"].to_numpy()
            # change semantic to "recombine now" instead of "recombine after"
            recombination_vec[1:] = recombination_vec[:-1]

            self.cM = np.zeros(self.n_markers, dtype=np.float32)
            start_idx = 0
            for chr_len in self.chr_lens:
                end_idx = start_idx + chr_len
                self.cM[start_idx + 1:end_idx] = recombination_vec[start_idx + 1:end_idx].cumsum() * 100
                start_idx = end_idx

        elif "cM" in genetic_map.columns:
            self.cM = genetic_map["cM"].to_numpy()

            recombination_vec = np.zeros(self.n_markers, dtype=np.float32)
            recombination_vec[start_idx + 1:end_idx] = self.cM[start_idx + 1:end_idx] - self.cM[start_idx:end_idx - 1]
            recombination_vec /= 100
        else:
            raise ValueError("One between RecombRate and cM must be specified")

        first_mrk_map = np.zeros(len(chr_map), dtype='bool')
        first_mrk_map[1:] = chr_map.iloc[1:].values != chr_map.iloc[:-1].values
        first_mrk_map[0] = True
        recombination_vec[first_mrk_map] = 0.5  # first equally likely
        self.recombination_vec = jax.device_put(
            recombination_vec,
            device=self.device
        )

        self.random_key = None
        if seed is None:
            seed = random.randint(0, 2**32)
        self.set_seed(seed)

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
         - population (array): loaded population of shape (n, m, 2), where
            n is the number of individual and m is the total number of marker.
        """
        population = np.loadtxt(file_name, dtype='bool')
        population = population.reshape(population.shape[0], self.n_markers, 2)
        return jax.device_put(population, device=self.device)

    def save_population(self, population: Population["n"], file_name: Union[Path, str]):
        """Save a population to file.

        Args:
         - population (array): population to save.
         - file_name (path): file path to save the population.
        """
        flatten_pop = population.reshape(population.shape[0], -1)
        np.savetxt(file_name, flatten_pop, fmt="%i")

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
        return Simulator._cross(
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
            Simulator._cross_individual,
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

    @staticmethod
    @jax.jit
    @partial(jax.vmap, in_axes=(0, None, 0))  # parallelize across individuals
    @partial(jax.vmap, in_axes=(0, None, 0), out_axes=1)  # parallelize parents
    def _cross(
        parent: Individual,
        recombination_vec: Float[Array, N_MARKERS],
        random_key: jax.random.PRNGKeyArray
    ) -> Haploid:
        return Simulator._cross_individual(
            parent,
            recombination_vec,
            random_key
        )

    @staticmethod
    @jax.jit
    def _cross_individual(
        parent: Individual,
        recombination_vec: Float[Array, N_MARKERS],
        random_key: jax.random.PRNGKeyArray
    ) -> Haploid:
        samples = jax.random.uniform(random_key, shape=recombination_vec.shape)
        rec_sites = samples < recombination_vec
        crossover_mask = jax.lax.associative_scan(jnp.logical_xor, rec_sites)

        crossover_mask = crossover_mask.astype(jnp.int8)
        haploid = jnp.take_along_axis(
            parent,
            crossover_mask[:, None],
            axis=-1
        )

        return haploid.squeeze()

    def double_haploid(self, population: Population["n"]) -> Population["n"]:
        """Computes the double haploid of the input population.

        Args:
         - population (array): input population of shape (n, m, 2).
        
        Returns:
         - population (array): Output population of shape (n, m, 2).
            This population will be homozygote.
        """
        keys = jax.random.split(self.random_key, num=len(population) + 1)
        self.random_key = keys[0]
        split_keys = keys[1:]
        return Simulator._vmap_dh(
            population,
            self.recombination_vec,
            split_keys
        )

    @staticmethod
    @jax.jit
    @partial(jax.vmap, in_axes=(0, None, 0))  # parallelize across individuals
    def _vmap_dh(
        individual: Individual,
        recombination_vec: Float[Array, N_MARKERS],
        random_key: jax.random.PRNGKeyArray
    ) -> Individual:
        haploid = Simulator._cross_individual(
            individual,
            recombination_vec,
            random_key
        )
        return jnp.broadcast_to(haploid[:, None], shape=(*haploid.shape, 2))

    def diallel(
        self,
        population: Population["n"],
        n_offspring: int = 1
    ) -> Population["n*(n-1)/2*n_offspring"]:
        """Diallel crossing function, i.e. crossing between every possible
        couple, except self-crossing.
        
        Args:
         - population (array): input population of shape (n, m, 2).
         - n_offspring (int): number of offspring per cross.
            The default value is 1.
        
        Returns:
         - population (array): output population of shape (l * n_offspring, m, 2),
            where l is the number of possible pair, i.e `n * (n-1) / 2`.
        """

        if n_offspring < 1:
            raise ValueError("n_offspring must be higher or equal to 1")

        all_indices = np.arange(len(population))
        diallel_indices = self._diallel_indices(all_indices)
        cross_indices = np.repeat(diallel_indices, n_offspring, axis=0)
        return self.cross(population[cross_indices])

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
    ) -> Population["n_crosses*n_offspring"]:
        """Computes random crosses on a population.

        Args:
         - population (array): input population of shape (n, m, 2).
         - n_crosses (int): number of random crosses to perform.
         - n_offspring (int): number of offspring per cross. 
            The default value is 1.
        
        Returns:
         - population (array): output population of shape (l, m, 2), 
            where l is `n_crosses * n_offspring`
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
        return self.cross(population[cross_indices])

    def select(
        self,
        population: Population["n"],
        k: int,
        f_index: Optional[Callable[[Population["n"]], Float[Array, "n"]]] = None
    ) -> Population["k"]:
        """Function to select individuals based on their score (index).

        Args:
         - population (array): input population of shape (n, m, 2).
         - k (int): number of individual to select.
         - f_index (function): function that computes a score from an individual.
          The function accepts as input the individual, i.e. and array of shape
          (m, 2) and returns a float number. The default f_index is the conventional index, 
          i.e. the sum of the marker effects masked with the SNPs from the genetic_map.
        
        Returns:
         - population (array): output population of shape (k, m, 2).
        """
        if f_index is None:
            f_index = conventional_index(self.GEBV_model)

        indices = f_index(population)
        _, best_pop = jax.lax.top_k(indices, k)
        return population[best_pop, :, :]

    def GEBV(
        self,
        population: Population["n traits"]
    ) -> pd.DataFrame:
        """ Computes the Genomic Estimated Breeding Values using the
        marker effects from the genetic_map.

        Args:
         - population (array): input population of shape (n, m, 2).
        
        Returns:
         - gebv (DataFrame): a DataFrame with n rows and a column for each trait.
            It contains the GEBV of each trait for each individual.
        """
        GEBV = self.GEBV_model(population)
        return pd.DataFrame(GEBV, columns=self.trait_names)

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

    @property
    def max_gebv(self) -> float:
        return self.GEBV_model.max

    @property
    def min_gebv(self) -> float:
        return self.GEBV_model.min

    @property
    def mean_gebv(self) -> float:
        return self.GEBV_model.mean

    @property
    def var_gebv(self) -> float:
        return self.GEBV_model.var
