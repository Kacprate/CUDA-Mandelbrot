# CUDA Mandelbrot Set renderer made by Kacprate (https://github.com/Kacprate)

import colorsys
import math
import sys
import time

import keyboard
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pygame
from numba import cuda

#---------------ZOOMING AND POSITION SETTINGS---------------
# zooms is the amount of zooms, pressing the zoom-in button makes it zoom, but it can be done at most as many times as defined by zooms variable
# finalZoomSize is the final zoom scale
# scale is the initial zoom scale
# center is the coordinates of the point we are aiming at

zooms = 30
finalZoomSize = 10000000000000000
#finalZoomSize = 1000000000000000
#finalZoomSize = 100000000000

scale = 250
center = {'x' : 19/30 + 12/3000 + 6/20000 - 16/50000000 + 4/50000000000 - 21/1000000000000 + 11/100000000000000 - 8/10000000000000000, 'y' : 15/30 - 24/2000000 - 4/50000000000 - 4/1000000000000 + 1/100000000000000 + 17/10000000000000000}
#center = {'x' : 1.768778833 + 20/15857146760, 'y' : -0.001738996}
#center = {'x' : 0.235125, 'y' : 0.827215}
#-----------------------------------------------------------

#---------------SETTINGS---------------
screen = {'x' : 800, 'y' : 800} # screen size
startMaxIterations = 50 # initial maximum iterations
DEBUG = True # debug features
MARK_AIM = True # marks the aim at the center of the screen
cursorSpeed = 1 # movement speed
fontSize = 20 # information display font size
showInfo = True # toggle information display on/off
#CUDA SETTINGS 
griddim = (32, 16) # dimensions of the grid
blockdim = (32, 8) # dimensions of the block
#--------------------------------------

pygame.init()
pygame.display.set_caption('CUDA Mandelbrot Set renderer by Kacprate')

# renderer variables
maxIterations = startMaxIterations
display = pygame.display.set_mode((screen['x'], screen['y']))
image = np.zeros((int(screen['x']), int(screen['y']), 3), dtype = np.float)
i = 0
surf = display
zoomCoeff = 1
if zooms != 0:
    zoomCoeff = (finalZoomSize / scale) ** (1/zooms)

# pygame loop variables
running = True
t = 0
lastFrameTicks = 1
doRender = True

@cuda.jit(device=True)
def HSVtoRGB(h, s, v):
    h60 = h / 60.0
    h60f = math.floor(h60)
    hi = int(h60f) % 6
    f = h60 - h60f
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    r, g, b = 0, 0, 0
    if hi == 0: r, g, b = v, t, p
    elif hi == 1: r, g, b = q, v, p
    elif hi == 2: r, g, b = p, v, t
    elif hi == 3: r, g, b = p, q, v
    elif hi == 4: r, g, b = t, p, v
    elif hi == 5: r, g, b = v, p, q
    r, g, b = int(r * 255), int(g * 255), int(b * 255)
    return r, g, b

@cuda.jit
def step_kernel(d_image, screenX, screenY, cX, cY, maxIterations, scale):
    startX = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x
    startY = cuda.blockDim.y * cuda.blockIdx.y + cuda.threadIdx.y
    gridX = cuda.gridDim.x * cuda.blockDim.x;
    gridY = cuda.gridDim.y * cuda.blockDim.y;

    for i in range(startX, screenX, gridX):
        x = i - int(screenX / 2)
        x = x / scale - cX
        for j in range(startY, screenY, gridY):
            y = -(j - int(screenY / 2))
            y = y / scale - cY
            c = complex(x, y)
            z = complex(0, 0)
            iteration = 1
            while iteration <= maxIterations:
                z = z**2 + c
                if z.real ** 2 + z.imag ** 2 >= 4:
                    break
                else:
                    iteration = iteration + 1

            h = (1 - iteration / maxIterations) * 360.0
            s = 1.0
            v = 1.0
            if iteration == maxIterations + 1:
                h, s, v = 0, 0, 0

            r, g, b = HSVtoRGB(h, s, v)
            indexA = i
            indexB = screenY - 1 - j
            d_image[indexA][indexB][0] = r
            d_image[indexA][indexB][1] = g
            d_image[indexA][indexB][2] = b

def lerp(a, b, t):
    return a + (b - a) * t

def dataBoard(i, s, fzs, mi, fi, zc, cs):
    return ["Step: " + str(i), " - zooming progress: " + str(math.floor(i / fi * 100)) + "%", " - scaling per step: ~x" + str(math.floor(zc * 1000) / 1000), "Zoom: x" + str(s), "Target zoom: x" + str(fzs), "Maximum function iterations per pixel: " + str(mi), "Coordinates:", " Re = " + str(-center['x']), " Im = " + str(center['y']), "Resolution: " + str(screen['x']) + "x" + str(screen['y']), "Cursor speed: " + str(cs)]

def renderHandler():
    global i, scale, surf
    if i >= zooms:
        i = zooms
        scale = finalZoomSize

    print("Coordinates:", " Re = " + str(-center['x']), " Im = " + str(center['y']))

    t1 = time.time()
    d_image = cuda.to_device(image)
    step_kernel[griddim, blockdim](d_image, int(screen['x']), int(screen['y']), center['x'], center['y'], maxIterations, scale)
    d_image.to_host()

    if DEBUG:
        dt = int((time.time() - t1) * 1000) / 1000
        print("Step render time: %f seconds" % dt)

    surf = pygame.surfarray.make_surface(image)

while running:
    lastFrameTicks = t
    t = pygame.time.get_ticks()
    deltaTime = (t - lastFrameTicks) / 1000.0

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_e:
                i = i + 1
                scale = math.floor(scale * zoomCoeff)
                doRender = True
            elif event.key == pygame.K_q:
                i = i - 1
                scale = math.floor(scale / zoomCoeff)
                doRender = True
            elif event.key == pygame.K_t:
                maxIterations += 50
                doRender = True
            elif event.key == pygame.K_g:
                maxIterations -= 50
                if maxIterations < 50:
                    maxIterations = 50
                doRender = True
            elif event.key == pygame.K_i:
                showInfo = not showInfo
    if keyboard.is_pressed('a'):
        center['x'] += cursorSpeed * deltaTime / scale
        doRender = True
    elif keyboard.is_pressed('d'):
        center['x'] -= cursorSpeed * deltaTime / scale
        doRender = True
    if keyboard.is_pressed('w'):
        center['y'] += cursorSpeed * deltaTime / scale
        doRender = True
    elif keyboard.is_pressed('s'):
        center['y'] -= cursorSpeed * deltaTime / scale
        doRender = True
    if keyboard.is_pressed('r'):
        cursorSpeed += 1
    elif keyboard.is_pressed('f'):
        cursorSpeed -= 1
        if cursorSpeed < 1:
            cursorSpeed = 1

    if doRender:
        doRender = False
        renderHandler()
    display.blit(surf, (0, 0))
    if showInfo:
        font = pygame.font.SysFont(None, fontSize)
        text = dataBoard(i, scale, finalZoomSize, maxIterations, zooms, zoomCoeff, cursorSpeed)
        label = []
        for line in text: 
            label.append(font.render(line, True, (0, 0, 0)))
        for line in range(len(label)):
            display.blit(label[line],(10 , 10 + (line*fontSize)+(5*line)))
    pygame.display.update()
pygame.quit()
