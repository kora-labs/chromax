# This script demonstrates and tests the usage of the `mutation_mask_index` parameter
# in the Chromax Simulator, specifically using `chromax.functional.cross` for fine control.
# The `mutation_mask_index` allows specifying which markers (loci) are subject to mutation.

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

import chromax.functional
from chromax import Simulator


# --- 1. Define parameters ---
N_INDIVIDUALS = 10
N_MARKERS = 100
MUTATION_PROBABILITY = 1.0
INITIAL_SEED = 42
PLOIDY = 2

# --- 2. Create a mutation_mask_index ---
# - True: Mutations can occur here.
# - False: MASKED (mutations cannot occur).
mutation_mask_index = np.zeros(N_MARKERS, dtype=bool)
mutation_mask_index[: N_MARKERS // 2] = False
mutation_mask_index[N_MARKERS // 2 :] = True

# --- 3. Create a genetic_map ---
genetic_map_data = {
    "CHR.PHYS": np.ones(N_MARKERS, dtype=int),
    "cM": np.arange(N_MARKERS, dtype=float) * 1.0,  # Dummy cM values
    "Yield": np.random.default_rng(INITIAL_SEED).normal(size=N_MARKERS),
}
genetic_map = pd.DataFrame(genetic_map_data)

# --- 4. Initialize Simulators (primarily for recombination_vec) ---
# We use functional.cross for direct control over keys and parameters.
# Simulators are used here to conveniently get a valid recombination_vec.
# Both simulators share the same base seed for their own internal state,
simulator_base = Simulator(
    genetic_map=genetic_map,
    seed=INITIAL_SEED,
    # mutation params don't matter here as we won't use its cross method directly for this test
)

# --- 5. Generate an initial population ---
key = jax.random.PRNGKey(INITIAL_SEED)
key, pop_subkey = jax.random.split(key)
# For functional.cross, we need the shape (N_INDIVIDUALS, N_MARKERS, ploidy)
# The ploidy must be divisible by 2 for the rearrange operation in functional.cross
initial_population = jax.random.randint(
    pop_subkey, shape=(N_INDIVIDUALS, N_MARKERS, PLOIDY), minval=0, maxval=2
).astype(jnp.int8)

print(initial_population.shape)

# --- 6. Select a single pair of parents ---
parent1_idx = 0
parent2_idx = 1
# Shape for functional.cross: (n_crosses, 2_parents, n_markers, ploidy)
# For this function, the parents array shape must be (n_crosses, 2, n_markers, ploidy)
# where ploidy is the number of alleles per individual
# Use JAX-compatible indexing with jnp.array
parent_indices = jnp.array([parent1_idx, parent2_idx])
parents = jnp.take(initial_population, parent_indices, axis=0)
# Reshape to (1, 2, N_MARKERS, PLOIDY) for one cross with two parents
parents_for_cross = parents.reshape(1, 2, N_MARKERS, PLOIDY)

print(
    f"Selected Parent 1 (idx {parent1_idx}) genome shape: {initial_population[parent1_idx].shape}"
)
print(
    f"Selected Parent 2 (idx {parent2_idx}) genome shape: {initial_population[parent2_idx].shape}"
)
print(f"Parents array for cross shape: {parents_for_cross.shape}\n")

# --- 8. Perform cross with NO MUTATION ---
print("\nPerforming cross with NO MUTATION...")
simulator_base.random_key, key_for_cross = jax.random.split(simulator_base.random_key)
print(f"Using key_for_cross: {key_for_cross}")

offspring_no_mutation = chromax.functional.cross(
    parents_for_cross,
    simulator_base.recombination_vec,
    key_for_cross,
    mutation_probability=0.0,
)
# offspring_no_mutation has shape (1, N_MARKERS, PLOIDY) because 1 cross was made.
offspring1_no_mut_genome = offspring_no_mutation[
    0
]  # Genome of the first (and only) offspring

# --- 9. Perform cross WITH MUTATION and MASK ---
print("\nPerforming cross WITH MUTATION and MASK...")
offspring_with_mutation = chromax.functional.cross(
    parents_for_cross,
    simulator_base.recombination_vec,
    key_for_cross,
    mutation_probability=MUTATION_PROBABILITY,
    mutation_index_mask=mutation_mask_index,
)
offspring1_with_mut_genome = offspring_with_mutation[0]  # Genome of the first offspring

c1 = jax.random.uniform(key_for_cross, shape=simulator_base.recombination_vec.shape)
c2 = jax.random.uniform(key_for_cross, shape=simulator_base.recombination_vec.shape)
print("Checking array uniform equality")
print(jnp.array_equal(c1, c2))


# --- 10. Compare offspring genomes locus by locus ---
print("\n--- Comparing Offspring Genomes ---")
print(f"Mutation Mask: \n{mutation_mask_index}")
print(f"Offspring (No Mutation) first 10 loci: \n{offspring1_no_mut_genome[:10, :]}")
print(
    f"Offspring (With Mutation & Mask) first 10 loci: \n{offspring1_with_mut_genome[:10, :]}\n"
)

mismatches_in_protected = 0
mutations_as_expected = 0
unexpected_behavior = 0

for locus in range(N_MARKERS):
    genome_no_mut_locus = offspring1_no_mut_genome[locus, :]
    genome_with_mut_locus = offspring1_with_mut_genome[locus, :]

    if not mutation_mask_index[locus]:
        if not jnp.array_equal(genome_with_mut_locus, genome_no_mut_locus):
            print(f"ERROR at protected locus {locus} (mutation_mask_index=False): ")
            print(f"  Expected (no mutation): {genome_no_mut_locus}")
            print(f"  Got (with mutation):    {genome_with_mut_locus}")
            mismatches_in_protected += 1
    else:  # This locus can have mutations (mutation_mask_index=True)
        # For loci with MUTATION_PROBABILITY = 1.0, each allele in the gametes
        # contributing to offspring_no_mut_locus should flip due to mutation.
        # Therefore, offspring_with_mut_locus should be element-wise (1 - offspring_no_mut_locus).
        expected_genome_after_mutation = 1 - genome_no_mut_locus

        if jnp.array_equal(genome_with_mut_locus, expected_genome_after_mutation):
            mutations_as_expected += 1
        else:
            print(f"ERROR at mutable locus {locus}: ")
            print(f"  Offspring (no mutation):    {genome_no_mut_locus}")
            print(f"  Expected (with mutation):   {expected_genome_after_mutation}")
            print(f"  Got (with mutation):        {genome_with_mut_locus}")
            unexpected_behavior += 1

print("\nSummary of Test Results: ")
print(f"Number of loci: {N_MARKERS}")
print(f"Number of mutable loci: {np.sum(mutation_mask_index)}")
print(f"Number of protected loci: {N_MARKERS - np.sum(mutation_mask_index)}")

print("\nChecks for protected regions (mutation_mask_index=False): ")
if mismatches_in_protected == 0:
    print("  OK: No mutations detected in protected regions.")
else:
    print(f"  ERROR: {mismatches_in_protected} protected loci showed mutations.")

print(f"\nChecks for mutable regions (MUTATION_PROBABILITY = {MUTATION_PROBABILITY}): ")
print(
    f"  # of loci where expected element-wise flipped alleles: {mutations_as_expected}"
)
print(
    f"  Number of loci with unexpected allele states in mutable region: {unexpected_behavior}"
)

# Final Assertions
assert mismatches_in_protected == 0, "FAIL: Mutations occurred in protected regions."
# For mutable regions with MUTATION_PROBABILITY = 1.0, all loci should show the flipped behavior.
assert (
    unexpected_behavior == 0
), f"FAIL: Unexpected behavior in mutable regions for {unexpected_behavior} loci."
assert mutations_as_expected == np.sum(
    mutation_mask_index
), "FAIL: Not all mutable loci showed the expected flipped allele behavior."

print("\nRobust `mutation_mask_index` test completed.")
print(
    "If all assertions passed, the mask is working as expected with functional.cross."
)
