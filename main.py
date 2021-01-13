# CUDA Mandelbrot Set renderer made by Kacprate (https://github.com/Kacprate)
# Version: 4.0
#
# Changes:
# 4.0
# - code refactor
# - position saving feature
# - argument parser
# - save multiple locations and settings
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
from modules.save_manager import Save_Manager
from modules.state_machine import State_Machine

# parse arguments
parser = argparse.ArgumentParser(description='CUDA Mandelbrot argument parser')
parser.add_argument('--config', dest='config', default="./config.json", help='Path to the configuration file')
parser.add_argument('--saves', dest='saves', default="./saves/saves.json", help='Path to the saves file')
args = parser.parse_args()

DEBUG = False # debug features
showInfo = True # toggle information display on/off
doRender = True
update = False

pygame.init()
pygame.display.set_caption('CUDA Mandelbrot Set renderer by Kacprate')

save_manager = Save_Manager(args.saves)
save_manager.load()

state_machine = State_Machine()



renderer = Renderer(args.config)
renderer.step(0, 0, False)

def run():
    global DEBUG, showInfo, doRender, update

    t0 = 0
    running = True
    while running:
        lastFrameTicks = t0
        t0 = pygame.time.get_ticks()
        deltaTime = (t0 - lastFrameTicks) / 1000.0
        moveDelta = deltaTime
        if moveDelta < 1/200:
            moveDelta = 1/100
        movex, movey = 0, 0

        pygame_events = pygame.event.get()

        for event in pygame_events:
            if event.type == pygame.QUIT:
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return

        if state_machine.get_state().name == "choosing_save_to_load":
            for event in pygame_events:
                if event.type == pygame.KEYDOWN:
                    slot = event.key - ord("1")
                    if 0 <= slot <= save_manager.slot_number:
                        save = save_manager.get_state(slot)
                        if save:
                            renderer.load_state(save)
                            state_machine.change_state("rendering")
                            doRender = True
                            update = False
                            break

        if state_machine.get_state().name == "choosing_save_to_save":
            for event in pygame_events:
                if event.type == pygame.KEYDOWN:
                    slot = event.key - ord("1")
                    if 0 <= slot <= save_manager.slot_number:
                        state = renderer.get_state()
                        save_manager.set_state(slot, state)
                        state_machine.change_state("rendering")
                        break

        if state_machine.get_state().name == "rendering":
            for event in pygame_events:
                if event.type == pygame.KEYDOWN:
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
                        state_machine.change_state("choosing_save_to_save")
                        continue
                    elif event.key == pygame.K_l: # load state
                        state_machine.change_state("choosing_save_to_load")
                        continue

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

run()
pygame.quit()
save_manager.save()