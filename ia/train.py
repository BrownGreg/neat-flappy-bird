import sys
import os
import neat
import pickle
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'game'))

from game_engine import FlappyBirdEnv

CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'neat_config.txt')
N_GENERATIONS = 100


def evaluate_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    env = FlappyBirdEnv()
    state = env.reset()
    done = False

    while not done:
        output = net.activate(state)
        action = 1 if output[0] > 0.5 else 0
        state, reward, done = env.step(action)

    return env.frames + 500 * env.score


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = evaluate_genome(genome, config)


def plot_stats(stats, output_path):
    generations = range(len(stats.most_fit_genomes))
    best_fitness = [g.fitness for g in stats.most_fit_genomes]
    avg_fitness = stats.get_fitness_mean()

    plt.figure(figsize=(10, 5))
    plt.plot(generations, best_fitness, label='Fitness maximale')
    plt.plot(generations, avg_fitness, label='Fitness moyenne')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Evolution de la fitness par generation')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Courbe sauvegardee dans {output_path}")


def run():
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        CONFIG_PATH
    )

    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    os.makedirs(os.path.join(os.path.dirname(__file__), 'checkpoints'), exist_ok=True)
    checkpointer = neat.Checkpointer(
        generation_interval=10,
        filename_prefix=os.path.join(os.path.dirname(__file__), 'checkpoints', 'checkpoint-')
    )
    population.add_reporter(checkpointer)

    best = population.run(eval_genomes, N_GENERATIONS)

    genome_path = os.path.join(os.path.dirname(__file__), 'best_genome.pkl')
    with open(genome_path, 'wb') as f:
        pickle.dump(best, f)

    plot_stats(stats, os.path.join(os.path.dirname(__file__), 'fitness_courbe.png'))

    print(f"\nMeilleur genome sauvegarde dans {genome_path}")
    print(f"Fitness du meilleur genome : {best.fitness:.1f}")


if __name__ == '__main__':
    run()