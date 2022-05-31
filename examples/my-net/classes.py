from __future__ import annotations
import math
import random
from abc import ABC, abstractmethod
from typing import Tuple, List, Iterable
import neat
import numpy as np
from dataclasses import dataclass
from neat import DefaultGenome
from neat.nn import FeedForwardNetwork

Color = Tuple[int, int, int]


@dataclass
class Position:
    x: int
    y: int


Vector = Position

MOVEMENT = (Vector(-1, 0), Vector(0, -1), Vector(1, 0), Vector(0, 1))


@dataclass
class RelativePosition:
    delta: Vector
    distance: float
    strength: int


@dataclass
class Nature(ABC):
    world: World
    pos: Position
    color: Color
    health: int
    reborn: bool = False

    def draw(self):
        print(self, " X=", self.pos.x, " Y=", self.pos.y, " HEALTH=", self.health)

    @abstractmethod
    def act(self) -> None:
        pass

    @abstractmethod
    def react(self, source: Nature) -> int:
        pass

    @property
    def can_be_attacked(self) -> bool:
        return False

    @property
    def strength(self) -> int:
        return 0


class Fruit(Nature):
    def __init__(self, world: World, pos: Position, health: int = 1):
        super().__init__(world, pos, (255, 255, 0), health, reborn=True)

    @property
    def can_be_attacked(self) -> bool:
        return self.health > 0

    def act(self):
        pass

    def react(self, source: Nature) -> int:
        result = self.health - self.strength
        self.health = max(0, self.health - source.strength)
        if self.health <= 0:
            self.world.kill(self)
        return min(source.strength, result)


class PoisonedFruit(Fruit):
    def __init__(self, world: World, pos: Position, health: int = 1):
        super().__init__(world, pos, health)
        self.color = (0, 255, 255)

    @property
    def strength(self) -> int:
        return 10


class Creature(Nature):
    def __init__(self, world: World, pos: Position, color: Color, health: int = 10):
        super().__init__(world, pos, color, health)
        self.age = 0
        self.generation = world.generation
        self.brain = None

    @property
    def fitness(self) -> int:
        return self.health

    def act(self):
        self.age += 3

    def react(self, source: Nature) -> int:
        return 0


class PredatorCreature(Creature):
    def __init__(self, world: World, pos: Position, health: int = 10):
        super().__init__(world, pos, (255, 0, 0), health)

    @property
    def strength(self) -> int:
        return math.ceil(self.health / 5)

    def act(self) -> None:
        if self.health > 0:
            self.age += 1
            sensor = self.look()
            action = self.brain.think(sensor)
            self.move(MOVEMENT[action])

    # def look(self) -> List[...]:
    #     inputs = [0] * 12
    #     dist_by_type = {Fruit: RelativePosition(Vector(0, 0), 99999.99, 0),
    #                     PoisonedFruit: RelativePosition(Vector(0, 0), 99999.9, 0)}
    #     victims = list(filter(lambda x: x.can_be_attacked, self.world.creatures))
    #     for victim in victims:
    #         delta = Vector(victim.pos.x - self.pos.x, victim.pos.y - self.pos.y)
    #         cur_distance = (delta.x ** 2 + delta.y ** 2) ** 0.5
    #         if dist_by_type[type(victim)].distance > cur_distance:
    #             dist_by_type[type(victim)].distance = cur_distance
    #             dist_by_type[type(victim)].delta = delta
    #             dist_by_type[type(victim)].strength = victim.strength
    #     offset = 0
    #     for d in dist_by_type:
    #         if dist_by_type[d].delta.x > 0:
    #             inputs[0+offset], inputs[2+offset] = dist_by_type[d].delta.x, 0
    #         else:
    #             inputs[2+offset], inputs[0+offset] = -dist_by_type[d].delta.x, 0
    #         if dist_by_type[d].delta.y > 0:
    #             inputs[1+offset], inputs[3+offset] = dist_by_type[d].delta.y, 0
    #         else:
    #             inputs[3+offset], inputs[1+offset] = -dist_by_type[d].delta.y, 0
    #         inputs[4+offset] = dist_by_type[d].distance
    #         inputs[5+offset] = dist_by_type[d].strength*100
    #         offset += 6
    #     return inputs
    def look(self) -> List[...]:
        direction_dist = [0, 0, 0, 0]
        distance = 99999.99
        nearest = (0, 0)
        strength = 0
        fruits = list(filter(lambda x: x.can_be_attacked, self.world.creatures))
        for fruit in fruits:
            delta_x, delta_y = fruit.pos.x - self.pos.x, fruit.pos.y - self.pos.y
            cur_distance = (delta_x ** 2 + delta_y ** 2) ** 0.5
            if distance >= cur_distance:
                distance = cur_distance
                nearest = (delta_x, delta_y)
                strength = fruit.strength
        if nearest[0] > 0:
            direction_dist[0], direction_dist[2] = nearest[0], 0
        else:
            direction_dist[2], direction_dist[0] = -nearest[0], 0
        if nearest[1] > 0:
            direction_dist[1], direction_dist[3] = nearest[1], 0
        else:
            direction_dist[3], direction_dist[1] = -nearest[1], 0
        direction_dist.append(distance)
        direction_dist.append(strength*10)
        return direction_dist

    def move(self, delta: Vector) -> None:
        pos = self.world.calculate_position(self.pos, delta)
        creature = self.world.check_position(pos)
        if creature is None or not creature.can_be_attacked:
            self.pos = pos
        else:
            self.attack(creature)

    def attack(self, victim: Nature) -> None:
        self.health += victim.react(self)
        if self.health <= 0:
            self.world.kill(self)


class VegetarianCreature(Creature):
    def __init__(self, world: World, pos: Position, health=10):
        super().__init__(world, pos, (0, 255, 0), health)

    @property
    def fitness(self):
        if self.health > 0:
            return self.health + (self.world.generation - self.generation)
        else:
            return 0


class SmartCreature(PredatorCreature):
    def __init__(self, world: World, genome: DefaultGenome, pos: Position):
        super().__init__(world, pos)
        self.genome = genome
        self.genome.fitness = self.fitness
        net = neat.nn.FeedForwardNetwork.create(genome, world.config)
        self.brain = SmartBrain(net)

    def attack(self, victim: Nature) -> None:
        super().attack(victim)
        self.genome.fitness = self.fitness


class Brain:
    def __init__(self):
        pass

    def think(self, inputs: Iterable[float]) -> float:
        return 0


class SmartBrain(Brain):
    def __init__(self, net: FeedForwardNetwork):
        super().__init__()
        self.net = net

    def think(self, inputs: Iterable[float]) -> int:
        result = self.net.activate(inputs)
        return np.argmax(result)


class World:
    def __init__(self, genomes, config, width: int, height: int):
        self.pop_size = len(genomes)
        self.width = width
        self.height = height
        self.generation = 1
        self.config = config
        self.creatures = []
        for _, genome in genomes:
            self.add_creature(genome)
        for _ in range(50):
            self.add_fruit(Fruit)
        # for _ in range(20):
        #     self.add_fruit(PoisonedFruit)

    def draw(self) -> None:
        for c in self.creatures:
            c.draw()

    def calculate_position(self, cur_pos: Position, delta: Vector) -> Position:
        pos = Position(cur_pos.x + delta.x, cur_pos.y + delta.y)
        if pos.x > self.width:
            pos.x = 1
        elif pos.x < 1:
            pos.x = self.width
        if pos.y > self.height:
            pos.y = 1
        elif pos.y < 1:
            pos.y = self.height
        return pos

    def check_position(self, pos: Position) -> Nature | None:
        for c in self.creatures:
            if c.pos.x == pos.x and c.pos.y == pos.y:
                return c
        return None

    def random_pos(self) -> Position:
        while True:
            pos = Position(x=random.randint(1, self.width), y=random.randint(1, self.height))
            if self.check_position(pos) is None:
                break
        return pos

    def add_creature(self, genome) -> None:
        pos = self.random_pos()
        self.creatures.append(SmartCreature(self, genome, pos))

    def add_fruit(self, fruit_type) -> None:
        pos = self.random_pos()
        fruit = fruit_type(self, pos)
        self.creatures.append(fruit)

    def next_generation(self) -> None:
        pass

    def tick(self) -> None:
        for c in self.creatures:
            c.act()

    def kill(self, creature: Nature) -> None:
        self.creatures.remove(creature)
        if creature.reborn:
            self.add_fruit(type(creature))
