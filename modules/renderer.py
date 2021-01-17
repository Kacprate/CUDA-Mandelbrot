import json
import math
import time

import numpy as np
import pygame
from numba import cuda

from modules.utils import HSVtoRGB, step_kernel

DEBUG = False

class Renderer:
    def __init__(self, config):
        with open(config) as json_file:
            data = json.load(json_file)
        if not data:
            raise Exception("Failed to open config file")

        startMaxIterations = data['startMaxIterations']
        finalZoomSize = data['finalZoomSize']
        zooms = data['zooms']
        initial_scale = data['initial_scale']

        # renderer variables
        self.maxIterations = startMaxIterations
        self.flags = pygame.DOUBLEBUF #| pygame.FULLSCREEN
        self.screen = {'x' : data['resolution']['x'], 'y': data['resolution']['y']} # screen size
        self.display = pygame.display.set_mode((self.screen['x'], self.screen['y']), self.flags)
        self.display.set_alpha(None)
        self.image = np.zeros((int(self.screen['x']), int(self.screen['y']), 3), dtype = np.float)
        self.d_image = cuda.to_device(self.image)
        self.step_index = 0
        self.surf = self.display
        self.finalZoomSize = finalZoomSize
        self.zooms = zooms
        self.scale = initial_scale
        if self.zooms != 0:
            self.zoomCoeff = (self.finalZoomSize / self.scale) ** (1 / self.zooms)
        else:
            self.zoomCoeff = 1

        self.center = {'x': data['default_center']['x'], 'y': data['default_center']['y']}

        self.AUTO_LOWER_CURSORSPEED = data['AUTO_LOWER_CURSORSPEED'] # automatically lowers the cursor speed if the performance drops
        self.cursorSpeed = data['cursorSpeed'] # initial cursor movement speed, 300 is optimal for good performance, too low might cause no movement at all, too high might lower the framerate significantly
        self.fontSize = data['fontSize'] # information display font size
        self.saves_font = pygame.font.SysFont(None, self.fontSize)

        # CUDA SETTINGS 
        self.griddim = data['cuda']['griddim'] # dimensions of the grid
        self.blockdim = data['cuda']['blockdim'] # dimensions of the block

    def change_iterations(self, delta):
        self.maxIterations += delta
        if self.maxIterations < 50:
            self.maxIterations = 50

    def change_cursor_speed(self, delta):
        self.cursorSpeed += delta
        if self.cursorSpeed < 150:
            self.cursorSpeed = 150

    def zoom_in(self):
        self.step_index += 1
        self.scale = math.floor(self.scale * self.zoomCoeff)
        if self.step_index >= self.zooms:
            self.step_index = self.zooms
            self.scale = self.finalZoomSize

    def zoom_out(self):
        self.step_index -= 1
        if self.step_index < 0:
            self.step_index = 0
        else:
            self.scale = math.floor(self.scale / self.zoomCoeff)

    def move_window(self, dx, dy):
        self.center['x'] += dx / self.scale
        self.center['y'] += dy / self.scale

    def get_state(self):
        state = dict()
        state["center"] = self.center.copy()
        state["step_index"] = self.step_index
        state["cursorSpeed"] = self.cursorSpeed
        state["scale"] = self.scale
        state["maxIterations"] = self.maxIterations
        state["timestamp"] = time.time()
        return state

    def load_state(self, state):
        try:
            self.center = state["center"].copy()
            self.step_index = state["step_index"]
            self.cursorSpeed = state["cursorSpeed"]
            self.scale = state["scale"]
            self.maxIterations = state["maxIterations"]
        except:
            raise Exception("Error while loading the state")

        if DEBUG:
            print("State loaded successfully")
    
    def get_render_data(self, fps):
        return [f"Step: {self.step_index}", 
            f" - zooming progress: {math.floor(self.step_index / self.zooms * 100)}%", 
            f" - scaling per step: ~x{math.floor(self.zoomCoeff * 1000) / 1000}", 
            f"Zoom: x{self.scale}", f"Target zoom: x{self.finalZoomSize}", 
            f"Maximum function iterations per pixel: {self.maxIterations}", 
            "Coordinates:", f" Re = {-self.center['x']}", f" Im = {self.center['y']}", 
            f"Resolution: {self.screen['x']}x{self.screen['y']}", 
            f"Cursor speed: {self.cursorSpeed}", 
            f"FPS: {int(fps)}"]

    def step(self, dx, dy, update):
        if DEBUG:
            print("Coordinates:", f" Re = {-self.center['x']} Im = {self.center['y']}")

        if update == True:
            if dx == 0 and dy == 0:
                return

        x1, x2, y1, y2 = 0, int(self.screen['x']), 0, int(self.screen['y'])
        if update:
            if dx < 0:
                x1 = self.screen['x'] + dx
            elif dx > 0:
                x2 = dx
            if dy < 0:
                y2 = -dy
            elif dy > 0:
                y1 = self.screen['y'] - dy

        t1 = time.time()
        step_kernel[self.griddim, self.blockdim](self.d_image, int(self.screen['x']), int(self.screen['y']), self.center['x'], self.center['y'], self.maxIterations, self.scale, update, x1, x2, y1, y2, dx, dy)
        self.d_image.to_host()

        if DEBUG:
            dt = int((time.time() - t1) * 1000) / 1000
            if dt != 0:
                print(f"Step render time: {dt} seconds, FPS: {1 / dt}")

        t1 = time.time()
        if update:
            self.surf.scroll(dx, dy)
            if dx != 0:
                for x in range(x1, x2):
                    for y in range(0, self.screen['y']):
                        y = self.screen['y'] - y - 1
                        self.surf.set_at((x, y), self.image[x][y])
            if dy != 0:
                for x in range(0, self.screen['x']):
                    for y in range(y1, y2):
                        y = self.screen['y'] - y - 1
                        self.surf.set_at((x, y), self.image[x][y])
        else:
            self.surf = pygame.surfarray.make_surface(self.image)

        if DEBUG:
            dt = int((time.time() - t1) * 1000) / 1000
            if dt != 0:
                print(f"Surface render time: {dt} seconds, FPS: {1 / dt}")
            else:
                print(f"Surface render time: {dt} seconds")

    def show_saves(self, saves):
        font = self.saves_font
        text = []
        for i, save in enumerate(saves):
            if save:
                txt = f"{i + 1}: timestamp: {save['timestamp']}, location: {save['center']}"
            else:
                txt = f"{i + 1}: Empty"
            text.append(txt)

        label = []
        for line in text: 
            label.append(font.render(line, True, (0, 0, 0)))
        for line in range(len(label)):
            self.display.blit(label[line], (10 , 10 + (line * self.fontSize) + (5 * line)))

    def show_info(self, fps):
        font = pygame.font.SysFont(None, self.fontSize)
        text = self.get_render_data(fps)
        label = []
        for line in text: 
            label.append(font.render(line, True, (0, 0, 0)))
        for line in range(len(label)):
            self.display.blit(label[line], (10 , 10 + (line * self.fontSize) + (5 * line)))
