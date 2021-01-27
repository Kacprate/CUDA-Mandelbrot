# CUDA Mandelbrot Set renderer made by Kacprate (https://github.com/Kacprate)
# Version: 4.0
#
# Changes:
# 4.0
# - code refactor
# - position saving feature
# - save multiple locations and settings
# - argument parser
# - tests
# - logging
# 3.2
# - optimized the algorithm, now it's 6x more efficient in comparison to v3.0

import argparse
import logging
import math
import sys
import time

import keyboard
import numpy as np
import pygame
from numba import cuda

from modules.renderer import Renderer
from modules.save_manager import Save_Manager
from modules.state_machine import State, State_Machine
from modules.utils import HSVtoRGB, lerp, sign

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

# parse arguments
parser = argparse.ArgumentParser(description='CUDA Mandelbrot argument parser')
parser.add_argument('--config', dest='config', default="./config.json", help='Path to the configuration file')
parser.add_argument('--saves', dest='saves', default="./saves/saves.json", help='Path to the saves file')
args = parser.parse_args()

# Renderer variables
DEBUG = False # debug features
showInfo = True # toggle information display on/off
doRender = True # render scane
update = False # True - evaluate only new pixels, False - evaluate all pixels

pygame.init()
pygame.display.set_caption('CUDA Mandelbrot Set renderer by Kacprate')

save_manager = Save_Manager(args.saves)
save_manager.load()

state_machine = State_Machine(states={1: State("rendering", [2, 3, 4]), 
                                      2: State("choosing_save_to_load", [1]),
                                      3: State("choosing_save_to_save", [1]),
                                      4: State("choosing_save_to_remove", [1])})

renderer = Renderer(args.config)
renderer.step(0, 0, False)

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

    if state_machine.get_state().name == "choosing_save_to_load":
        for event in pygame_events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    state_machine.change_state("rendering")
                    break
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
                if event.key == pygame.K_ESCAPE:
                    state_machine.change_state("rendering")
                    break
                slot = event.key - ord("1")
                if 0 <= slot <= save_manager.slot_number:
                    state = renderer.get_state()
                    save_manager.set_state(slot, state)
                    state_machine.change_state("rendering")
                    break
    
    if state_machine.get_state().name == "choosing_save_to_remove":
        for event in pygame_events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    state_machine.change_state("rendering")
                    break
                slot = event.key - ord("1")
                if 0 <= slot <= save_manager.slot_number:
                    save_manager.set_state(slot, None)
                    state_machine.change_state("rendering")
                    break

    if state_machine.get_state().name == "rendering":
        for event in pygame_events:
            if event.type == pygame.QUIT:
                running = False
                break
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_t:
                    renderer.change_iterations(50)
                    doRender = True
                    update = False
                elif event.key == pygame.K_g:
                    renderer.change_iterations(-50)
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
                elif event.key == pygame.K_m: # remove state
                    state_machine.change_state("choosing_save_to_remove")
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
            renderer.change_cursor_speed(1)
        elif keyboard.is_pressed('f'):
            renderer.change_cursor_speed(-1)
        if keyboard.is_pressed('q'):
            renderer.zoom_out()
            doRender = True
            update = False
        elif keyboard.is_pressed('e'):
            renderer.zoom_in()
            doRender = True
            update = False

        if doRender:
            doRender = False
            if movex != 0 or movey != 0:
                leng = abs(movex) + abs(movey)
                movex = int(movex ** 2 / leng) * sign(movex)
                movey = int(movey ** 2 / leng) * sign(movey)
                renderer.move_window(movex, movey)
            renderer.step(movex, movey, update)

    renderer.display.blit(renderer.surf, (0, 0))
    if deltaTime != 0:
        fps = 1 / deltaTime
    else:
        fps = 60

    if renderer.AUTO_LOWER_CURSORSPEED and fps < 15 and renderer.cursorSpeed > 300:
        renderer.change_cursor_speed(-50)
    
    if state_machine.get_state().name.startswith("choosing_save"):
        renderer.show_saves(save_manager.saves)
    elif showInfo:
        renderer.show_info(fps)

    update = True

    pygame.display.update()

pygame.quit()
save_manager.save()
