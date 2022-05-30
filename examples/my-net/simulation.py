from typing import Tuple

import pygame
from classes import Nature, World, Fruit, SmartCreature


class MySprite(pygame.sprite.Sprite):
    def __init__(self, creature: Nature, size: Tuple[int, int], font=None):
        pygame.sprite.Sprite.__init__(self)
        self.size = size
        self.font = font
        self.creature = creature
        self.image = pygame.Surface(size)
        self.color = creature.color
        self.image.fill(self.color)
        self.rect = self.image.get_rect()
        self.rect.x = (self.creature.pos.x - 1) * self.size[0]
        self.rect.y = (self.creature.pos.y - 1) * self.size[1]


class CreatureSprite(MySprite):
    def update(self) -> None:
        self.rect.x = (self.creature.pos.x - 1) * self.size[0]
        self.rect.y = (self.creature.pos.y - 1) * self.size[1]
        if self.creature.health == 0 and self.color != (0, 0, 0):
            self.color = (0, 0, 0)
            self.image.fill(self.color)
        s = str(self.creature.health).rjust(2, ' ')
        text = self.font.render(s, True, (0, 0, 0), self.color)
        w = text.get_width()
        h = text.get_height()
        self.image.blit(text, [self.size[0] / 2 - w / 2, self.size[1] / 2 - h / 2])


class FruitSprite(MySprite):
    def update(self) -> None:
        if self.creature.health <= 0:
            self.kill()


def run_simulation(world: World) -> None:
    SPRITE = {Fruit: FruitSprite, SmartCreature: CreatureSprite}
    pygame.init()
    WORLD_WIDTH = 50
    WORLD_HEIGHT = 50
    SCREEN_WIDTH = 500
    SCREEN_HEIGHT = 500
    SCREEN_FPS = 20
    SCREEN_COLOR = (128, 128, 128)
    CRT_FONT = pygame.font.SysFont("Arial", 10)

    # pygame.mixer.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Creatures")
    clock = pygame.time.Clock()
    all_sprites = pygame.sprite.Group()
    for c in world.creatures:
        all_sprites.add(SPRITE[type(c)](c, (int(SCREEN_WIDTH / WORLD_WIDTH), int(SCREEN_HEIGHT / WORLD_HEIGHT)), CRT_FONT))

    running = True
    while running:
        clock.tick(SCREEN_FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        world.tick()
        sprite_list = list(map(lambda x: x.creature, all_sprites.sprites()))
        for c in world.creatures:
            if c not in sprite_list:
                all_sprites.add(SPRITE[type(c)](c, (int(SCREEN_WIDTH / WORLD_WIDTH), int(SCREEN_HEIGHT / WORLD_HEIGHT)), CRT_FONT))
        all_sprites.update()
        screen.fill(SCREEN_COLOR)
        all_sprites.draw(screen)
        pygame.display.flip()

    pygame.quit()
