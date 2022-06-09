import os
import neat
import pickle
# import visualize
from classes import World
from simulation import run_simulation


def eval_genomes(genomes, config):
    world = World(genomes, config, 1200, 800)
    # run_simulation(world)
    for _ in range(100):
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
    #
    # # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    # stats = neat.StatisticsReporter()
    # p.add_reporter(stats)
    # # p.add_reporter(neat.Checkpointer(1))
    #
    # # Run for up to 200 generations.
    winner = p.run(eval_genomes, 50)
    save_winner(winner)
    # # winner = load_winner()

    # Display the winning genome.
    # print('\nBest genome:\n{!s}'.format(winner))

    genomes = [[0, winner]] * 50
    run_simulation(World(genomes, config, 1200, 800))


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    run(config_path)
