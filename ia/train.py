import sys
import os
import random
import neat
import pickle
import functools
import multiprocessing
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'game'))
from game_engine import FlappyBirdEnv

CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'neat_config.txt')
GENOME_PATH = os.path.join(os.path.dirname(__file__), 'best_genome.pkl')
N_GENERATIONS = 500
N_RUNS = 7


def evaluate_genome(genome, config, seeds):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    total_fitness = 0.0

    for seed in seeds:
        random.seed(seed)
        env = FlappyBirdEnv()
        state = env.reset()
        done = False
        run_fitness = 0.0

        while not done:
            output = net.activate(state)
            action = 1 if output[0] > 0.5 else 0
            state, reward, done = env.step(action)

            if not done:
                run_fitness += 1.0
                if state[3] > 0 and state[4] > 0:
                    run_fitness += 2.0

        run_fitness += env.score * 1000
        total_fitness += run_fitness

    return total_fitness / N_RUNS


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
    population.add_reporter(neat.Checkpointer(
        generation_interval=10,
        filename_prefix=os.path.join(os.path.dirname(__file__), 'checkpoints', 'checkpoint-')
    ))

    best_ever = None

    def eval_genomes(genomes, config):
        nonlocal best_ever
        seeds = [random.randint(0, 10_000) for _ in range(N_RUNS)]
        eval_fn = functools.partial(evaluate_genome, config=config, seeds=seeds)

        with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
            fitnesses = pool.map(eval_fn, [genome for _, genome in genomes])

        for (_, genome), fitness in zip(genomes, fitnesses):
            genome.fitness = fitness

        gen_best = max(genomes, key=lambda x: x[1].fitness)[1]
        if best_ever is None or gen_best.fitness > best_ever.fitness:
            best_ever = gen_best
            with open(GENOME_PATH, 'wb') as f:
                pickle.dump(best_ever, f)
            print(f"  >> Nouveau meilleur genome sauvegarde (fitness={best_ever.fitness:.1f})")

    best = population.run(eval_genomes, N_GENERATIONS)

    plot_stats(stats, os.path.join(os.path.dirname(__file__), 'fitness_courbe.png'))
    print(f"\nMeilleur genome : fitness={best_ever.fitness:.1f}")
    print(f"Sauvegarde dans {GENOME_PATH}")


if __name__ == '__main__':
    run()
