from chromax import Simulator, sample_data
simulator = Simulator(genetic_map=sample_data.genetic_map)
f1 = simulator.load_population(sample_data.genome)
f2, parent_ids = simulator.random_crosses(f1, 100, n_offspring=10)

print(parent_ids.shape)