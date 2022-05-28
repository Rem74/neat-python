from __future__ import annotations
import math
import random
from abc import ABC, abstractmethod
from typing import Tuple, List, Iterable
import neat
import numpy as np
from dataclasses import dataclass

from neat import DefaultGenome
from neat.genome import DefaultGenomeConfig
from neat.nn import FeedForwardNetwork


Color = Tuple[int, int, int]


@dataclass
class Position:
    x: int
    y: int


Vector = Position

MOVEMENT = (Vector(-1, 0), Vector(0, -1), Vector(1, 0), Vector(0, 1))


@dataclass
class Nature(ABC):
    world: world.World
    pos: Position
    color: Color
    health: int

    def draw(self):
        print(self, " X=", self.pos.x, " Y=", self.pos.y, " HEALTH=", self.health)

    @abstractmethod
    def act(self) -> None:
        pass

    @abstractmethod
    def react(self, source: Nature) -> None:
        pass


class Fruit(Nature):
    def __init__(self, world: World, pos: Position, health: int = 1):
        super().__init__(world, pos, (0, 255, 255), health)

    def act(self):
        pass

    def react(self, source: Nature):
        pass


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
        pass

    def react(self, source: Nature):
        pass


class PredatorCreature(Creature):
    def __init__(self, world: World, pos: Position, health: int = 10):
        super().__init__(world, pos, (255, 0, 0), health)

    def get_strength(self, mode: int = 1) -> int:
        return math.ceil(self.health / 5 / mode)

    def act(self) -> None:
        if self.health > 0:
            self.age += 1
            sensor = []
            s = self._look()
            sensor.extend(s[0])
            sensor.append(s[1])
            action = self.brain.think(sensor)
            self._move(MOVEMENT[action])

    def _look(self) -> Tuple[List[int], float]:
        direction_dist = [0, 0, 0, 0]
        distance = 99999.99
        nearest = (0, 0)
        for fruit in self.world.fruits:
            delta_x, delta_y = fruit.pos.x - self.pos.x, fruit.pos.y - self.pos.y
            cur_distance = (delta_x ** 2 + delta_y ** 2) ** 0.5
            if distance >= cur_distance:
                distance = cur_distance
                nearest = (delta_x, delta_y)
        if nearest[0] > 0:
            direction_dist[0], direction_dist[2] = nearest[0], 0
        else:
            direction_dist[2], direction_dist[0] = -nearest[0], 0
        if nearest[1] > 0:
            direction_dist[1], direction_dist[3] = nearest[1], 0
        else:
            direction_dist[3], direction_dist[1] = -nearest[1], 0
        return direction_dist, distance

    def _move(self, delta: Vector) -> None:
        pos = self.world.calculate_position(self.pos, delta)
        creature_in_position = self.world.check_position(pos)
        if type(creature_in_position) == Fruit and creature_in_position.health > 0:
            self._attack(creature_in_position)
        else:
            self.pos = pos

    def _attack(self, victim: Nature) -> None:
        my_strength = self.get_strength()
        my_health = self.health + min(my_strength, victim.health)
        victim.health = max(0, victim.health - my_strength)
        if victim.health == 0:
            self.world.kill(victim, reborn=True)
        self.health = my_health


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

    def _attack(self, victim: Nature) -> None:
        super()._attack(victim)
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
        self.fruits = []
        self.creatures = []
        for genome_id, genome in genomes:
            pos = self.random_pos()
            self.creatures.append(SmartCreature(self, genome, pos))
        for _ in range(50):
            self.add_fruit()

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

    def check_position(self, pos: Position) -> Nature:
        for c in self.fruits:
            if c.pos.x == pos.x and c.pos.y == pos.y:
                return c
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

    def add_fruit(self) -> None:
        pos = self.random_pos()
        self.fruits.append(Fruit(self, pos))

    def next_generation(self) -> None:
        pass

    def tick(self) -> None:
        for c in self.creatures:
            c.act()

    def kill(self, obj: Nature, reborn=False) -> None:
        if type(obj) == Fruit:
            self.fruits.remove(obj)
            if reborn:
                self.add_fruit()
