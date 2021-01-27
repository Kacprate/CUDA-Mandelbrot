import math

from numba import cuda


def lerp(a, b, t):
    return a + (b - a) * t

def sign(x):
    if x < 0:
        return -1
    elif x > 0:
        return 1
    else:
        return 0

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
