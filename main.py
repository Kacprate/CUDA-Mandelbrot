# CUDA Mandelbrot Set renderer made by Kacprate (https://github.com/Kacprate)
# Version: 3.1
# Changes:
# - optimized the algorithm, now it's 3x more efficient

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
# finalZoomSize = 10000000000000000
finalZoomSize = 1000000000000000
#finalZoomSize = 100000000000

scale = 250
# center = {'x' : 19/30 + 12/3000 + 6/20000 - 16/50000000 + 4/50000000000 - 21/1000000000000 + 11/100000000000000 - 8/10000000000000000, 'y' : 15/30 - 24/2000000 - 4/50000000000 - 4/1000000000000 + 1/100000000000000 + 17/10000000000000000}
center = {'x' : 1.768778833 + 20/15857146760, 'y' : -0.001738996}
#center = {'x' : 0.235125, 'y' : 0.827215}
#-----------------------------------------------------------

#---------------SETTINGS---------------
screen = {'x' : 800, 'y' : 800} # screen size
startMaxIterations = 50 # initial maximum iterations
AUTO_LOWER_CURSORSPEED = True # automatically lowers the cursor speed if the performance drops
DEBUG = False # debug features
cursorSpeed = 300 # initial cursor movement speed, 300 is optimal for good performance, too low might cause no movement at all, too high might lower the framerate significantly
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
flags = pygame.DOUBLEBUF
display = pygame.display.set_mode((screen['x'], screen['y']), flags)
display.set_alpha(None)
image = np.zeros((int(screen['x']), int(screen['y']), 3), dtype = np.float)
surf = pygame.surfarray.make_surface(image)
i = 0
surf = display
zoomCoeff = 1
if zooms != 0:
    zoomCoeff = (finalZoomSize / scale) ** (1/zooms)
update = False

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

@cuda.jit(device=True)
def iterate(d_image, x, i, j, screenY, cY, maxIterations, scale):
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

@cuda.jit
def step_kernel(d_image, screenX, screenY, cX, cY, maxIterations, scale, update, x1, x2, y1, y2, dx, dy):
    startX = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x
    startY = cuda.blockDim.y * cuda.blockIdx.y + cuda.threadIdx.y
    gridX = cuda.gridDim.x * cuda.blockDim.x;
    gridY = cuda.gridDim.y * cuda.blockDim.y;

    x1 += startX
    y1 += startY

    if not update or dx != 0:
        for i in range(x1, x2, gridX):
            x = i - int(screenX / 2)
            x = x / scale - cX
            for j in range(startY, screenY, gridY):
                iterate(d_image, x, i, j, screenY, cY, maxIterations, scale)
    if not update or dy != 0:
        for i in range(startX, screenX, gridX):
            x = i - int(screenX / 2)
            x = x / scale - cX
            for j in range(y1, y2, gridY):
                iterate(d_image, x, i, j, screenY, cY, maxIterations, scale)

def lerp(a, b, t):
    return a + (b - a) * t

def sign(x):
    if x < 0:
        return -1
    elif x > 0:
        return 1
    else:
        return 0

def dataBoard(i, s, fzs, mi, fi, zc, cs, fps):
    return ["Step: " + str(i), " - zooming progress: " + str(math.floor(i / fi * 100)) + "%", " - scaling per step: ~x" + str(math.floor(zc * 1000) / 1000), "Zoom: x" + str(s), "Target zoom: x" + str(fzs), "Maximum function iterations per pixel: " + str(mi), "Coordinates:", " Re = " + str(-center['x']), " Im = " + str(center['y']), "Resolution: " + str(screen['x']) + "x" + str(screen['y']), "Cursor speed: " + str(cs), "FPS: " + str(int(fps))]

def renderHandler(dx, dy, update):
    global i, scale, surf
    if i >= zooms:
        i = zooms
        scale = finalZoomSize

    print("Coordinates:", " Re = " + str(-center['x']), " Im = " + str(center['y']))

    if update == True:
        if dx == 0 and dy == 0:
            return

    x1, x2, y1, y2 = 0, int(screen['x']), 0, int(screen['y'])
    if update:
        if dx < 0:
            x1 = screen['x'] + dx
        elif dx > 0:
            x2 = dx
        if dy < 0:
            y2 = -dy
        elif dy > 0:
            y1 = screen['y'] - dy
    t1 = time.time()
    d_image = cuda.to_device(image)
    step_kernel[griddim, blockdim](d_image, int(screen['x']), int(screen['y']), center['x'], center['y'], maxIterations, scale, update, x1, x2, y1, y2, dx, dy)
    d_image.to_host()

    if DEBUG:
        dt = int((time.time() - t1) * 1000) / 1000
        if dt != 0:
            print("Step render time: {} seconds, FPS: {}".format(dt, 1/dt))

    t1 = time.time()
    if update:
        surf.scroll(dx, dy)
        # for x in range(x1, x2):
        #     for y in range(y1, y2):
        #         y = screen['y'] - y - 1
        #         surf.set_at((x, y), image[x][y])
        if dx != 0:
            for x in range(x1, x2):
                for y in range(0, screen['y']):
                    y = screen['y'] - y - 1
                    surf.set_at((x, y), image[x][y])
        if dy != 0:
            for x in range(0, screen['x']):
                for y in range(y1, y2):
                    y = screen['y'] - y - 1
                    surf.set_at((x, y), image[x][y])
    else:
        surf = pygame.surfarray.make_surface(image)
    if DEBUG:
        dt = int((time.time() - t1) * 1000) / 1000
        if dt != 0:
            print("Surface render time: {} seconds, FPS: {}".format(dt, 1/dt))
        else:
            print("Surface render time: {} seconds".format(dt))

renderHandler(0, 0, update)
while running:
    lastFrameTicks = t
    t = pygame.time.get_ticks()
    deltaTime = (t - lastFrameTicks) / 1000.0
    movex, movey = 0, 0

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_e:
                i = i + 1
                scale = math.floor(scale * zoomCoeff)
                doRender = True
                update = False
            elif event.key == pygame.K_q:
                i = i - 1
                scale = math.floor(scale / zoomCoeff)
                doRender = True
                update = False
            elif event.key == pygame.K_t:
                maxIterations += 50
                doRender = True
                update = False
            elif event.key == pygame.K_g:
                maxIterations -= 50
                if maxIterations < 50:
                    maxIterations = 50
                doRender = True
                update = False
            elif event.key == pygame.K_i:
                showInfo = not showInfo
    if keyboard.is_pressed('a'):
        movex = math.floor(cursorSpeed * deltaTime)
        doRender = True
    elif keyboard.is_pressed('d'):
        movex = -math.floor(cursorSpeed * deltaTime)
        doRender = True
    if keyboard.is_pressed('w'):
        movey = math.floor(cursorSpeed * deltaTime)
        doRender = True
    elif keyboard.is_pressed('s'):
        movey = -math.floor(cursorSpeed * deltaTime)
        doRender = True
    if keyboard.is_pressed('r'):
        cursorSpeed += 1
    elif keyboard.is_pressed('f'):
        cursorSpeed -= 1
        if cursorSpeed < 150:
            cursorSpeed = 150

    if doRender:
        doRender = False
        if movex != 0 or movey != 0:
            leng = abs(movex) + abs(movey)
            movex = int(movex ** 2 / leng) * sign(movex)
            movey = int(movey ** 2 / leng) * sign(movey)
            center['x'] += movex / scale
            center['y'] += movey / scale
        renderHandler(movex, movey, update)
    display.blit(surf, (0, 0))
    fps = 1000
    if deltaTime != 0:
        fps = 1/deltaTime
    if AUTO_LOWER_CURSORSPEED and fps < 15 and cursorSpeed > 300:
        cursorSpeed -= 50
        if cursorSpeed < 300:
            cursorSpeed = 300
    if showInfo:
        font = pygame.font.SysFont(None, fontSize)
        text = dataBoard(i, scale, finalZoomSize, maxIterations, zooms, zoomCoeff, cursorSpeed, fps)
        label = []
        for line in text: 
            label.append(font.render(line, True, (0, 0, 0)))
        for line in range(len(label)):
            display.blit(label[line],(10 , 10 + (line*fontSize)+(5*line)))
    if not update:
        update = True
    pygame.display.update()
pygame.quit()
