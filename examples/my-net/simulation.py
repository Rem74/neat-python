from typing import Any

import pygame
from classes import World


class MySprite(pygame.sprite.Sprite):
    def __init__(self, creature, size, font=None):
        pygame.sprite.Sprite.__init__(self)
        self.size = size
        self.font = font
        self.creature = creature
        self.image = pygame.Surface(size)
        self.color = creature.color
        self.image.fill(self.color)
        self.rect = self.image.get_rect()
        self.rect.x = (self.creature.x - 1) * self.size[0]
        self.rect.y = (self.creature.y - 1) * self.size[1]


class CreatureSprite(MySprite):
    def update(self):
        self.rect.x = (self.creature.x - 1) * self.size[0]
        self.rect.y = (self.creature.y - 1) * self.size[1]
        if self.creature.health == 0 and self.color != (0, 0, 0):
            self.color = (0, 0, 0)
            self.image.fill(self.color)
        s = str(self.creature.health).rjust(2, ' ')
        text = self.font.render(s, True, (0, 0, 0), self.color)
        w = text.get_width()
        h = text.get_height()
        self.image.blit(text, [self.size[0] / 2 - w / 2, self.size[1] / 2 - h / 2])


class FruitSprite(MySprite):
    def update(self):
        if self.creature.health <= 0:
            self.kill()


def run_simulation(world):
    pygame.init()
    WORLD_WIDTH = 50
    WORLD_HEIGHT = 50
    WORLD_CAPACITY = 50
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
    fruit_sprites = pygame.sprite.Group()
    for c in world.creatures:
        all_sprites.add(
            CreatureSprite(c, (int(SCREEN_WIDTH / WORLD_WIDTH), int(SCREEN_HEIGHT / WORLD_HEIGHT)), CRT_FONT))
    for c in world.fruits:
        fruit_sprites.add(FruitSprite(c, (int(SCREEN_WIDTH / WORLD_WIDTH), int(SCREEN_HEIGHT / WORLD_HEIGHT))))

    running = True
    while running:
        clock.tick(SCREEN_FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        world.tick()
        all_sprites.update()
        fruit_list = list(map(lambda x: x.creature, fruit_sprites.sprites()))
        for f in world.fruits:
            if f not in fruit_list:
                fruit_sprites.add(FruitSprite(f, (int(SCREEN_WIDTH / WORLD_WIDTH), int(SCREEN_HEIGHT / WORLD_HEIGHT))))
        fruit_sprites.update()
        screen.fill(SCREEN_COLOR)
        all_sprites.draw(screen)
        fruit_sprites.draw(screen)
        pygame.display.flip()

    pygame.quit()
