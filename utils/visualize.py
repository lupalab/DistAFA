import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def mosaic(x, m):
    B,H,W,C = x.shape
    if C == 1:
        x = np.repeat(x, 3, axis=-1)
        m = np.repeat(m, 3, axis=-1)
    img = np.ones([B,H,W,3], dtype=np.uint8) * np.array([60,160,180])
    img = img * (1-m) + x * m
    
    return img.astype(np.uint8)

def make_grid(data, padding=2):
    assert data.dtype == np.uint8, f'data type {data.dtype} is not supported'
    assert np.max(data) <= 255, f'data is out of the valid range'
    if data.shape[-1] == 1:
        data = np.repeat(data, 3, axis=-1)
    nmaps = data.shape[0]
    xmaps = ymaps = int(np.sqrt(nmaps))
    height, width = int(data.shape[1] + padding), int(data.shape[2] + padding)
    grid = np.zeros((height * ymaps + padding, width * xmaps + padding, 3), dtype=np.uint8)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            grid[y*height+padding:y*height+height,x*width+padding:x*width+width] = data[k]
            k = k + 1
    return grid
    
def save_image(data, fname):
    grid = make_grid(data)
    fig, axs = plt.subplots()
    axs.imshow(grid)
    axs.axis('off')
    plt.savefig(fname, dpi=200, bbox_inches='tight')
    plt.close(fig=fig)

    