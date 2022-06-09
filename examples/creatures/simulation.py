import pygame
from classes import World


def run_simulation(world: World) -> None:
    pygame.init()
    SCREEN_WIDTH = world.width
    SCREEN_HEIGHT = world.height
    SCREEN_FPS = 10
    SCREEN_COLOR = (112, 112, 112)

    # pygame.mixer.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Creatures")
    clock = pygame.time.Clock()
    all_sprites = pygame.sprite.Group()
    for c in world.creatures:
        all_sprites.add(c)

    running = True
    while running:
        clock.tick(SCREEN_FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        world.tick()
        sprite_list = all_sprites.sprites()
        for c in world.creatures:
            if c not in sprite_list:
                all_sprites.add(c)
        all_sprites.update()
        screen.fill(SCREEN_COLOR)
        all_sprites.draw(screen)
        pygame.display.flip()

    pygame.quit()
