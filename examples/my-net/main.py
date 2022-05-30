import os
import neat
import pickle
# import visualize
from classes import World
from simulation import run_simulation


def eval_genomes(genomes, config):
    world = World(genomes, config, 50, 50)
    for _ in range(200):
        world.tick()


def save_winner(winner, filename="winner.dat"):
    with open(filename, "wb") as f:
        pickle.dump(winner, f)


def load_winner(filename="winner.dat"):
    try:
        with open("winner.dat", "rb") as f:
            w = pickle.load(f)
            return w
    except FileNotFoundError:
        print(f"Запрашиваемый файл {filename } не найден")
        return None


def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    # p.add_reporter(neat.Checkpointer(1))

    # Run for up to 300 generations.
    winner = p.run(eval_genomes, 100)
    save_winner(winner)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # node_names = {-1: 'A', -2: 'B', 0: 'A XOR B'}
    # visualize.draw_net(config, winner, True, node_names=node_names)
    # visualize.draw_net(config, winner, True, node_names=node_names, prune_unused=True)
    # visualize.plot_stats(stats, ylog=False, view=True)
    # visualize.plot_species(stats, view=True)

    # p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-9')
    # p.run(eval_genomes, 1)
    genomes = [[0, winner]] * 1
    run_simulation(World(genomes, config, 50, 50))


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    run(config_path)
