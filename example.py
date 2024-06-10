import pygame
import numpy as np

# Initialize Pygame
pygame.init()

# Define dimensions and colors
array = np.random.rand(5, 5)  # Replace with your 2D array
cell_size = 100
margin = 5
width = len(array[0]) * (cell_size + margin) + margin
height = len(array) * (cell_size + margin) + margin
background_color = (30, 30, 30)
border_color = (200, 200, 200)
title = "Machine 1"

# Create Pygame screen
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption(title)

def get_color(value):
    # Map the value to a color (green to orange)
    green_value = int(255 * value)
    red_value = int(255 * (1 - value))
    return (red_value, green_value, 0)

def draw_array(screen, array):
    for y, row in enumerate(array):
        for x, value in enumerate(row):
            color = get_color(value)
            rect = pygame.Rect(x * (cell_size + margin) + margin,
                               y * (cell_size + margin) + margin,
                               cell_size, cell_size)
            pygame.draw.rect(screen, border_color, rect)
            inner_rect = rect.inflate(-margin, -margin)
            pygame.draw.rect(screen, color, inner_rect)

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Draw the array
    screen.fill(background_color)
    draw_array(screen, array)
    pygame.display.flip()

pygame.quit()
