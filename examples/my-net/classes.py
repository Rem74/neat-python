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
        self.creatures = []
        for genome_id, genome in genomes:
            while True:
                x = random.randint(1, width)
                y = random.randint(1, height)
                if self.check_position(x, y) is None:
                    break
            self.creatures.append(SmartCreature(self, genome, x, y))

    def draw(self):
        for c in self.creatures:
            c.draw()

    def check_position(self, x, y):
        for c in self.creatures:
            if c.x == x and c.y == y:
                return c
        return None

    def next_generation(self):
        pass

    def tick(self):
        for c in self.creatures:
            c.act()


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
            for d in DIRECTION:
                s = self.look_in_direction(d)
                sensor.append(s[0])
                sensor.append(s[1])
                sensor.append(s[2])
            sensor.append(self.health)
            action = self.brain.think(sensor)
            self.move(MOVEMENT[action])

    def look_in_direction(self, d):
        pos_x, pos_y = self.x, self.y
        distance = 0
        while 0 < pos_x < self.world.width and self.world.height > pos_y > 0:
            distance = distance + 1
            pos_x = pos_x + d[0]
            pos_y = pos_y + d[1]
            obj = self.world.check_position(pos_x, pos_y)
            if obj is not None:
                return distance, obj.health, obj.get_strength()
        return max(self.world.width, self.world.height), 0, 0

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
        if creature_in_position is None:
            self.x, self.y = pos_x, pos_y
        else:
            if creature_in_position.health > 0:
                self.bite(creature_in_position)

    def get_strength(self, mode=1):
        return 0

    def bite(self, victim):
        pass


class PredatorCreature(Creature):
    def __init__(self, world, x, y):
        super().__init__(world, x, y)
        self.color = (255, 0, 0)

    def get_strength(self, mode=1):
        return math.ceil(self.health / 5 / mode)

    def bite(self, victim):
        my_strength = self.get_strength()
        victim_strength = victim.get_strength(2)
        my_health = self.health + min(my_strength, victim.health)
        victim.health = max(0, victim.health - my_strength)
        if victim.health > 0:
            victim.health += min(victim_strength, self.health)
            self.health = max(0, my_health - victim_strength)
        else:
            self.health = my_health
        self.health = min(99, self.health)
        victim.health = min(99, victim.health)


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

    def bite(self, victim):
        super().bite(victim)
        self.genome.fitness = self.fitness
        victim.genome.fitness = victim.fitness


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
