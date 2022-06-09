from __future__ import annotations
import math
import random
import pygame as pg
from abc import abstractmethod
from typing import List, Iterable
import neat
import numpy as np
from dataclasses import dataclass
from neat import DefaultGenome
from neat.nn import FeedForwardNetwork
from geometry import *


@dataclass
class Position:
    x: int
    y: int


Vector = Position


class Nature(pg.sprite.Sprite):
    def __init__(self, world: World, pos: Position, health: int, reborn: bool):
        pg.sprite.Sprite.__init__(self)
        self.world = world
        self.pos = pos
        self.health = health
        self.reborn = reborn

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
        super().__init__(world, pos, health, reborn=True)
        self.image = pg.Surface((10, 10))
        self.image.fill((255, 255, 0))
        self.rect = self.image.get_rect()
        self.rect.x = self.pos.x
        self.rect.y = self.pos.y

    @property
    def can_be_attacked(self) -> bool:
        return self.health > 0

    def act(self) -> None:
        pass

    def react(self, source: Nature) -> int:
        result = self.health - self.strength
        self.health = max(0, self.health - source.strength)
        if self.health <= 0:
            self.world.kill(self)
        return min(source.strength, result)

    def update(self) -> None:
        if self.health <= 0:
            self.kill()


class PoisonedFruit(Fruit):
    def __init__(self, world: World, pos: Position, health: int = 1):
        super().__init__(world, pos, health)

    @property
    def strength(self) -> int:
        return 10


class Creature(Nature):
    orientation: float
    speed: float = 5.00
    age: int = 0
    generation: int
    brain: Brain = None

    def __init__(self, world: World, pos: Position, image: pg.Surface, health: int = 10):
        super().__init__(world, pos, health, False)
        self.generation = world.generation
        self.orientation = random.random() * math.pi
        self.image = image
        self.initial_image = image
        self.rect = self.image.get_rect()

    @property
    def fitness(self) -> int:
        return self.health

    def act(self):
        self.age += 1

    def react(self, source: Nature) -> int:
        return 0

    def update(self) -> None:
        self.rotate(self.orientation)
        self.rect.x = self.pos.x
        self.rect.y = self.pos.y

    def rotate(self, radians: float):
        self.image = pg.transform.rotate(self.initial_image, -math.degrees(radians) - 180)


class PredatorCreature(Creature):
    sensors_qty: int = 9
    sensors_view_angle: int = 90
    sensor_len: int = 200

    def __init__(self, world: World, pos: Position, health: int = 10):
        super().__init__(world, pos, pg.image.load("creature.png"), health)

    @property
    def strength(self) -> int:
        return math.ceil(self.health / 5)

    def act(self) -> None:
        if self.health > 0:
            self.age += 1
            sensor = self.look()
            if self.brain is not None:
                action = self.brain.think(sensor)
                self.orientation += math.radians(10) * (action - 1)
            self.move()

    def look(self) -> List[...]:
        def get_sensor(i):
            sensors_view_angle_rad = math.radians(self.sensors_view_angle)
            angle = math.radians(self.sensors_view_angle/self.sensors_qty)
            sensor_direction = self.orientation - sensors_view_angle_rad/2 + i * angle
            start_point = self.rect.center
            end_point = (start_point[0] + round(self.sensor_len * math.cos(sensor_direction)),
                         start_point[1] + round(self.sensor_len * math.sin(sensor_direction)))
            return start_point, end_point

        def get_nearest(cur_sensor):
            fruits = list(filter(lambda x: x.can_be_attacked, self.world.creatures))
            min_distance = 9999
            for f in fruits:
                distance_to_obj = distance_to_rect(cur_sensor, Rectangle(f.rect.left, f.rect.top, f.rect.width, f.rect.height))
                if distance_to_obj is None:
                    distance_to_obj = 9999
                min_distance = min(min_distance, distance_to_obj)
            return min_distance

        sensor_distance = []
        for n in range(self.sensors_qty):
            sensor = get_sensor(n)
            sensor_distance.append(get_nearest(sensor))
        return sensor_distance

    def move(self) -> None:
        self.pos = self.world.calculate_position(self.pos, self.orientation, self.speed)
        self.update()
        self.speed = max(0.00, self.speed * (1 - self.world.friction))
        for c in self.world.creatures:
            if c is not self and self.rect.colliderect(c.rect) and c.can_be_attacked:
                self.attack(c)

    def attack(self, victim: Nature) -> None:
        self.health += victim.react(self)
        if self.health <= 0:
            self.world.kill(self)


class VegetarianCreature(Creature):
    def __init__(self, world: World, pos: Position, health=10):
        super().__init__(world, pos, pg.image.load("creature.png"), health)

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
        self.friction = 0.00
        for _, genome in genomes:
            self.add_creature(genome)
        for _ in range(50):
            self.add_fruit(Fruit)
        # for _ in range(20):
        #     self.add_fruit(PoisonedFruit)

    def draw(self) -> None:
        for c in self.creatures:
            c.draw()

    def calculate_position(self, cur_pos: Position, orientation: float, speed: float) -> Position:
        pos = Position(cur_pos.x + round(speed * math.cos(orientation)),
                       cur_pos.y + round(speed * math.sin(orientation)))
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
