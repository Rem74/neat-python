import math
import random
import neat
import numpy as np

DIRECTION = [(-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1)]
MOVEMENT = [(-1, 0), (0, -1), (1, 0), (0, 1)]


class World:
    def __init__(self, genomes, config, width, height):
        self.pop_size = len(genomes)
        self.width = width
        self.height = height
        self.generation = 1
        self.config = config
        self.fruits = []
        self.creatures = []
        for genome_id, genome in genomes:
            pos = self.random_pos()
            self.creatures.append(SmartCreature(self, genome, pos[0], pos[1]))
        for _ in range(50):
            self.add_fruit()

    def draw(self):
        for c in self.creatures:
            c.draw()

    def check_position(self, x, y):
        for c in self.fruits:
            if c.x == x and c.y == y:
                return c
        for c in self.creatures:
            if c.x == x and c.y == y:
                return c
        return None

    def random_pos(self):
        while True:
            x = random.randint(1, self.width)
            y = random.randint(1, self.height)
            if self.check_position(x, y) is None:
                break
        return x, y

    def add_fruit(self):
        pos = self.random_pos()
        self.fruits.append(Fruit(pos[0], pos[1]))

    def next_generation(self):
        pass

    def tick(self):
        for c in self.creatures:
            c.act()

    def kill(self, obj, reborn=False):
        if type(obj) == Fruit:
            self.fruits.remove(obj)
            if reborn:
                self.add_fruit()


class Fruit:
    def __init__(self, x, y, health=1):
        self.x = x
        self.y = y
        self.health = health
        self.color = (0, 255, 255)


class Creature:
    def __init__(self, world, x, y):
        self.age = 0
        self.generation = world.generation
        self.health = 10
        self.world = world
        self.x = x
        self.y = y
        self.color = (255, 255, 255)
        self.brain = None

    def draw(self):
        print(self, " X=", self.x, " Y=", self.y, " HEALTH=", self.health)

    @property
    def fitness(self):
        return self.health

    def act(self):
        if self.health > 0:
            self.age += 1
            sensor = []
            s = self.look()
            sensor.extend(s[0])
            sensor.append(s[1])
            action = self.brain.think(sensor)
            self.move(MOVEMENT[action])

    def look(self):
        direction = [0, 0, 0, 0]
        distance = 99999.99
        nearest = (0, 0)
        for fruit in self.world.fruits:
            delta_x, delta_y = fruit.x - self.x, fruit.y - self.y
            cur_distance = (delta_x ** 2 + delta_y ** 2) ** 0.5
            if distance >= cur_distance:
                distance = cur_distance
                nearest = (delta_x, delta_y)
        if nearest[0] > 0:
            direction[0], direction[2] = nearest[0], 0
        else:
            direction[2], direction[0] = -nearest[0], 0
        if nearest[1] > 0:
            direction[1], direction[3] = nearest[1], 0
        else:
            direction[3], direction[1] = -nearest[1], 0
        return direction, distance

    def move(self, d):
        pos_x, pos_y = self.x + d[0], self.y + d[1]
        if pos_x > self.world.width:
            pos_x = 1
        elif pos_x < 1:
            pos_x = self.world.width
        if pos_y > self.world.height:
            pos_y = 1
        elif pos_y < 1:
            pos_y = self.world.height

        creature_in_position = self.world.check_position(pos_x, pos_y)
        if type(creature_in_position) == Fruit and creature_in_position.health > 0:
            self.attack(creature_in_position)
        else:
            self.x, self.y = pos_x, pos_y

    def get_strength(self, mode=1):
        return 0

    def attack(self, victim):
        pass


class PredatorCreature(Creature):
    def __init__(self, world, x, y):
        super().__init__(world, x, y)
        self.color = (255, 0, 0)

    def get_strength(self, mode=1):
        return math.ceil(self.health / 5 / mode)

    def attack(self, victim):
        my_strength = self.get_strength()
        my_health = self.health + min(my_strength, victim.health)
        victim.health = max(0, victim.health - my_strength)
        if victim.health == 0:
            self.world.kill(victim, reborn=True)
        self.health = my_health


class VegetarianCreature(Creature):
    def __init__(self, world, x, y):
        super().__init__(world, x, y)
        self.color = (0, 255, 0)

    @property
    def fitness(self):
        if self.health > 0:
            return self.health + (self.world.generation - self.generation)
        else:
            return 0


class SmartCreature(PredatorCreature):
    def __init__(self, world, genome, x, y):
        super().__init__(world, x, y)
        self.genome = genome
        self.genome.fitness = self.fitness
        net = neat.nn.FeedForwardNetwork.create(genome, world.config)
        self.brain = SmartBrain(net)

    def attack(self, victim):
        super().attack(victim)
        self.genome.fitness = self.fitness


class Brain:
    def __init__(self):
        pass

    def think(self, inputs):
        return 0


class SmartBrain(Brain):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def think(self, inputs):
        result = self.net.activate(inputs)
        return np.argmax(result)
