# This script demonstrates and tests the usage of the `mutation_mask_index` parameter
# in the Chromax Simulator, specifically using `chromax.functional.cross` for fine control.
# The `mutation_mask_index` allows specifying which markers (loci) are subject to mutation.

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from chromax import Simulator
import chromax.functional # For direct cross function call

# --- 1. Define parameters ---
N_INDIVIDUALS = 10
N_MARKERS = 100 # Keep it manageable for printing, e.g., 20-100
MUTATION_PROBABILITY = 1.0 # Ensure mutations are highly likely in unmasked regions
INITIAL_SEED = 42 # Seed for initial population and base simulator states

# --- 2. Create a mutation_mask_index ---
# - True: MASKED (NO mutations allowed).
# - False: UNMASKED (mutations CAN occur).
mutation_mask_index = np.zeros(N_MARKERS, dtype=bool)
# Mask the first half of the markers.
mutation_mask_index[: N_MARKERS // 2] = True
# Unmask the second half of the markers.
mutation_mask_index[N_MARKERS // 2 :] = False

# --- 3. Create a genetic_map ---
genetic_map_data = {
    "CHR.PHYS": np.ones(N_MARKERS, dtype=int),
    "cM": np.arange(N_MARKERS, dtype=float) * 1.0, # Dummy cM values
    "Yield": np.random.default_rng(INITIAL_SEED).normal(size=N_MARKERS), # Example trait
}
genetic_map = pd.DataFrame(genetic_map_data)

# --- 4. Initialize Simulators (primarily for recombination_vec) ---
# We use functional.cross for direct control over keys and parameters.
# Simulators are used here to conveniently get a valid recombination_vec.
# Both simulators share the same base seed for their own internal state,
# but the key for the cross operation itself will be controlled separately.

simulator_base = Simulator(
    genetic_map=genetic_map,
    seed=INITIAL_SEED 
    # mutation params don't matter here as we won't use its cross method directly for this test
)
recombination_vec = simulator_base.recombination_vec # Get a valid recombination vector

# --- 5. Generate an initial population ---
key = jax.random.PRNGKey(INITIAL_SEED)
key, pop_subkey = jax.random.split(key)
initial_population = jax.random.randint(
    pop_subkey, shape=(N_INDIVIDUALS, N_MARKERS, 2), minval=0, maxval=2
).astype(jnp.int8)

# --- 6. Select a single pair of parents ---
parent1_idx = 0
parent2_idx = 1
# Shape for functional.cross: (n_crosses, 2_parents, n_markers, 2_ploidy_alleles)
parents_for_cross = initial_population[jnp.array([[parent1_idx, parent2_idx]])]
# parents_for_cross now has shape (1, 2, N_MARKERS, 2)

print(f"Selected Parent 1 (idx {parent1_idx}) genome shape: {initial_population[parent1_idx].shape}")
print(f"Selected Parent 2 (idx {parent2_idx}) genome shape: {initial_population[parent2_idx].shape}")
print(f"Parents array for cross shape: {parents_for_cross.shape}\n")

# --- 7. Generate a new JAX random key for the cross operation ---
# This single key will be used for both cross operations to ensure identical recombination.
key, key_for_cross = jax.random.split(key)
print(f"Using key_for_cross: {key_for_cross}")

# --- 8. Perform cross with NO MUTATION ---
print("\nPerforming cross with NO MUTATION...")
offspring_no_mutation = chromax.functional.cross(
    parents=parents_for_cross,
    recombination_vec=recombination_vec,
    random_key=key_for_cross, # Use the dedicated key
    mutation_probability=0.0, # NO mutations
    mutation_index_mask=None # Not relevant when mutation_probability is 0
)
# offspring_no_mutation has shape (1, N_MARKERS, 2) because 1 cross was made.
offspring1_no_mut_genome = offspring_no_mutation[0] # Genome of the first (and only) offspring

# --- 9. Perform cross WITH MUTATION and MASK ---
print("\nPerforming cross WITH MUTATION and MASK...")
offspring_with_mutation = chromax.functional.cross(
    parents=parents_for_cross,
    recombination_vec=recombination_vec, # Same recombination vector
    random_key=key_for_cross, # CRITICAL: Same key for identical recombination events
    mutation_probability=MUTATION_PROBABILITY, # Mutations active
    mutation_index_mask=mutation_mask_index # Apply the defined mask
)
offspring1_with_mut_genome = offspring_with_mutation[0] # Genome of the first offspring

# --- 10. Compare offspring genomes locus by locus ---
print("\n--- Comparing Offspring Genomes ---")
print(f"Mutation Mask (True = MASKED, no mutation expected): \n{mutation_mask_index}")
print(f"Offspring (No Mutation) first 10 loci: \n{offspring1_no_mut_genome[:10, :]}")
print(f"Offspring (With Mutation & Mask) first 10 loci: \n{offspring1_with_mut_genome[:10, :]}\n")

mismatches_in_masked = 0
mutations_as_expected_in_unmasked = 0
unexpected_behavior_in_unmasked = 0

for locus in range(N_MARKERS):
    genome_no_mut_locus = offspring1_no_mut_genome[locus, :]
    genome_with_mut_locus = offspring1_with_mut_genome[locus, :]

    if mutation_mask_index[locus]:  # This locus is MASKED
        if not jnp.array_equal(genome_with_mut_locus, genome_no_mut_locus):
            print(f"ERROR at MASKED locus {locus}:")
            print(f"  Expected (no mutation): {genome_no_mut_locus}")
            print(f"  Got (with mutation):    {genome_with_mut_locus}")
            mismatches_in_masked += 1
    else:  # This locus is UNMASKED
        # For unmasked loci with MUTATION_PROBABILITY = 1.0, each allele in the gametes
        # contributing to offspring_no_mut_locus should flip due to mutation.
        # Therefore, offspring_with_mut_locus should be element-wise (1 - offspring_no_mut_locus).
        expected_genome_after_mutation = 1 - genome_no_mut_locus # Element-wise flip

        if jnp.array_equal(genome_with_mut_locus, expected_genome_after_mutation):
            mutations_as_expected_in_unmasked += 1
        else:
            print(f"ERROR at UNMASKED locus {locus}:")
            print(f"  Offspring (no mutation):    {genome_no_mut_locus}")
            print(f"  Expected (with mutation):   {expected_genome_after_mutation} (element-wise 1 - no_mutation)")
            print(f"  Got (with mutation):        {genome_with_mut_locus}")
            unexpected_behavior_in_unmasked +=1

print(f"\nSummary of Test Results:")
print(f"Number of loci: {N_MARKERS}")
print(f"Number of MASKED loci: {np.sum(mutation_mask_index)}")
print(f"Number of UNMASKED loci: {N_MARKERS - np.sum(mutation_mask_index)}")

print(f"\nChecks for MASKED region:")
if mismatches_in_masked == 0:
    print("  OK: No mutations detected in MASKED regions.")
else:
    print(f"  ERROR: {mismatches_in_masked} loci showed differences in MASKED regions.")

print(f"\nChecks for UNMASKED region (MUTATION_PROBABILITY = {MUTATION_PROBABILITY}):")
print(f"  Number of loci where mutation resulted in expected element-wise flipped alleles: {mutations_as_expected_in_unmasked}")
print(f"  Number of loci with unexpected allele states in UNMASKED region: {unexpected_behavior_in_unmasked}")

# Final Assertions
assert mismatches_in_masked == 0, "FAIL: Mutations occurred in MASKED regions."
# For unmasked regions with MUTATION_PROBABILITY = 1.0, all loci should show the flipped behavior.
assert unexpected_behavior_in_unmasked == 0, \
    f"FAIL: Unexpected behavior in UNMASKED regions for {unexpected_behavior_in_unmasked} loci. All should have flipped alleles."
assert mutations_as_expected_in_unmasked == (N_MARKERS - np.sum(mutation_mask_index)), \
    "FAIL: Not all unmasked loci showed the expected flipped allele behavior."

print("\nRobust `mutation_mask_index` test completed.")
print("If all assertions passed, the mask is working as expected with functional.cross.")

# --- Remove old verification logic (already done by overwriting) ---
