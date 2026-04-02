import sys
import os
import neat
import pickle

sys.path.insert(0, os.path.dirname(__file__))
import visualize

GENOME_PATH = os.path.join(os.path.dirname(__file__), 'best_genome.pkl')
CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'neat_config.txt')

with open(GENOME_PATH, 'rb') as f:
    genome = pickle.load(f)

config = neat.Config(
    neat.DefaultGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    CONFIG_PATH
)

visualize.draw_net(config, genome, view=True, filename='ia/reseau')