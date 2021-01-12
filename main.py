# CUDA Mandelbrot Set renderer made by Kacprate (https://github.com/Kacprate)
# Version: 4.0
#
# Changes:
# 4.0
# - code refactor
# - position saving feature
# - argument parser
# 3.2
# - optimized the algorithm, now it's 6x more efficient in comparison to v3.0

import math
import sys
import time
import argparse

import keyboard
import numpy as np
import pygame
from numba import cuda

from modules.renderer import Renderer
from modules.utils import HSVtoRGB, lerp, sign

# parse arguments
parser = argparse.ArgumentParser(description='CUDA Mandelbrot argument parser')
parser.add_argument('--config', dest='config', default="./config.json", help='Path to the configuration file')
parser.add_argument('--saves', dest='saves', default="./saves", help='Path to the saves folder')
args = parser.parse_args()

DEBUG = False # debug features
showInfo = True # toggle information display on/off

pygame.init()
pygame.display.set_caption('CUDA Mandelbrot Set renderer by Kacprate')

# pygame loop variables
running = True
t = 0
lastFrameTicks = 1
doRender = True
update = False

renderer = Renderer(args.config)

renderer.step(0, 0, update)
while running:
    lastFrameTicks = t
    t = pygame.time.get_ticks()
    deltaTime = (t - lastFrameTicks) / 1000.0
    moveDelta = deltaTime
    if moveDelta < 1/200:
        moveDelta = 1/100
    movex, movey = 0, 0

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
                continue
            if event.key == pygame.K_t:
                renderer.maxIterations += 50
                doRender = True
                update = False
            elif event.key == pygame.K_g:
                renderer.maxIterations -= 50
                if renderer.maxIterations < 50:
                    renderer.maxIterations = 50
                doRender = True
                update = False
            elif event.key == pygame.K_i:
                showInfo = not showInfo
            elif event.key == pygame.K_p: # save state
                state = renderer.get_state()
            elif event.key == pygame.K_l: # load state
                renderer.load_state(state)
                doRender = True
                update = False

    if keyboard.is_pressed('a'):
        movex = math.floor(renderer.cursorSpeed * moveDelta)
        doRender = True
    elif keyboard.is_pressed('d'):
        movex = -math.floor(renderer.cursorSpeed * moveDelta)
        doRender = True
    if keyboard.is_pressed('w'):
        movey = math.floor(renderer.cursorSpeed * moveDelta)
        doRender = True
    elif keyboard.is_pressed('s'):
        movey = -math.floor(renderer.cursorSpeed * moveDelta)
        doRender = True
    if keyboard.is_pressed('r'):
        renderer.cursorSpeed += 1
    elif keyboard.is_pressed('f'):
        renderer.cursorSpeed -= 1
        if renderer.cursorSpeed < 150:
            renderer.cursorSpeed = 150
    if keyboard.is_pressed('q'):
        renderer.i -= 1
        renderer.scale = math.floor(renderer.scale / renderer.zoomCoeff)
        doRender = True
        update = False
    elif keyboard.is_pressed('e'):
        renderer.i += 1
        renderer.scale = math.floor(renderer.scale * renderer.zoomCoeff)
        doRender = True
        update = False

    if doRender:
        doRender = False
        if movex != 0 or movey != 0:
            leng = abs(movex) + abs(movey)
            movex = int(movex ** 2 / leng) * sign(movex)
            movey = int(movey ** 2 / leng) * sign(movey)
            renderer.center['x'] += movex / renderer.scale
            renderer.center['y'] += movey / renderer.scale
        renderer.step(movex, movey, update)

    renderer.display.blit(renderer.surf, (0, 0))
    fps = 60
    if deltaTime != 0:
        fps = 1/deltaTime
    if renderer.AUTO_LOWER_CURSORSPEED and fps < 15 and renderer.cursorSpeed > 300:
        renderer.cursorSpeed -= 50
        if renderer.cursorSpeed < 300:
            renderer.cursorSpeed = 300
    if showInfo:
        renderer.show_info(fps)
    if not update:
        update = True
    pygame.display.update()

pygame.quit()
