# pylint: disable=no-member,unsupported-assignment-operation
# pylint: disable=invalid-sequence-index
import cv2
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt

from modeling import train_model

# GLOBAL variables
XYZ_FILE = 'data/dtm/xyz/gom_sigsbee.xyz'
MASK_FILE = 'data/mask/interp_bw_6px.jpg'


def readDtm():
    """
    read raw xyz data file
    """
    fname = XYZ_FILE
    df = pd.read_csv(fname, header=None, delimiter=r"\s+")
    df.columns = ['x', 'y', 'z']

    x = np.asarray(df['x'])
    y = np.asarray(df['y'])
    z = np.asarray(df['z'])

    xmax = np.max(x)
    xmin = np.min(x)
    ymax = np.max(y)
    ymin = np.min(y)
    incr = 0.002

    xi = np.arange(xmin, xmax, incr)
    yi = np.arange(ymin, ymax, incr)
    xi, yi = np.meshgrid(xi, yi)

    grid = griddata((x, y), z, (xi, yi), method='linear')
    col_mean = np.nanmean(grid, axis=0)
    inds = np.where(np.isnan(grid))
    grid[inds] = np.take(col_mean, inds[1])

    return grid


def readEscarpmentMask():
    """
    read mask for escarpment interpretation 
    """
    mask = cv2.imread(MASK_FILE, cv2.IMREAD_GRAYSCALE)
    mask = np.float64(mask)
    mask[mask <= 50] = 1
    mask[mask > 50] = 0
    return mask


def getTileExtents(grid):
    """
    get overlapping tile patches
    128x128 with 64 pixel overlap in both rows and columns 
    """
    row_list = []
    nrow = grid.shape[0]
    delta = 64
    for i in range(delta, nrow, delta):
        istart_row = i - delta
        iend_row = i - delta + 127
        if iend_row >= nrow:
            iend_row = nrow - 1
        row_list.append((istart_row, iend_row))

    col_list = []
    ncol = grid.shape[1]
    for i in range(delta, ncol, delta):
        istart_col = i - delta
        iend_col = i - delta + 127
        if iend_col >= ncol:
            iend_col = ncol - 1
        col_list.append((istart_col, iend_col))

    return row_list, col_list


def augmentTiles(tiles, angle_start=50, angle_end=360, incr=75):
    """
    augment data by flipping and rotating
    this should produce a total of 120*3*5 + 120*3 = 2160 from 120 tile input
    """
    tmp_tiles = []
    output_tiles = []

    # add original tiles
    for tile in tiles:
        tmp_tiles.append(tile)

    # add flipped left-right tiles
    for tile in tiles:
        tmp_tiles.append(np.fliplr(tile))

    # add flipped up-down tiles
    for tile in tiles:
        tmp_tiles.append(np.flipud(tile))

    # add original and flipped tiles to output
    for tile in tmp_tiles:
        output_tiles.append(tile)

    # rotate tiles and add to output
    for angle in range(angle_start, angle_end, incr):
        for tile in tmp_tiles:
            output_tiles.append(ndimage.rotate(tile, angle, reshape=False))

    return output_tiles


def augmentTilesBig(tiles, angle_start=35, angle_end=360, incr=35):
    """
    augment data by flipping and rotating
    this should produce a total of 120*4*10 + 120*4 = 5280 from 120 tile input
    """
    tmp_tiles = []
    output_tiles = []

    # add original tiles
    for tile in tiles:
        tmp_tiles.append(tile)

    # add flipped left-right tiles
    for tile in tiles:
        tmp_tiles.append(np.fliplr(tile))

    # add flipped up-down tiles
    for tile in tiles:
        tmp_tiles.append(np.flipud(tile))

    # add flipped up-down + left-right tiles
    for tile in tiles:
        tmp_tiles.append(np.flipud(np.fliplr(tile)))

    # add original and flipped tiles to output
    for tile in tmp_tiles:
        output_tiles.append(tile)

    # rotate tiles and add to output
    for angle in range(angle_start, angle_end, incr):
        for tile in tmp_tiles:
            output_tiles.append(ndimage.rotate(tile, angle, reshape=False))

    return output_tiles


def makePatches(dtm, mask, row_list, col_list):
    """
    create tiles of overlapping 128x128 patches from dtm and mask
    """

    tiles = {"dtm": [], "mask": []}
    for key in tiles:
        if key == "dtm":
            grid = dtm
        elif key == "mask":
            grid = mask

        grid_list = []
        for r in row_list:
            for c in col_list:
                template = np.zeros((128, 128))
                r1 = r[0]
                r2 = r[1]
                c1 = c[0]
                c2 = c[1]
                template[0:r2-r1, 0:c2-c1] = grid[r1:r2, c1:c2]
                grid_list.append(template)

        if key == "dtm":
            tiles["dtm"] = np.asarray(augmentTilesBig(grid_list))
        elif key == "mask":
            tiles["mask"] = np.asarray(augmentTilesBig(grid_list))

    return tiles


def main():
    dtm = readDtm()
    dtm = (dtm - np.mean(dtm)) / np.std(dtm)  # normalize dtm distribution
    dtm = np.flipud(dtm)  # input data set has wrong orientation
    mask = readEscarpmentMask()

    # counts, bins = np.histogram(np.ravel(dtm),bins=30)
    # plt.close('all')
    # plt.hist(bins[:-1], bins, weights=counts)
    # plt.show()

    row_list, col_list = getTileExtents(dtm)
    data = makePatches(dtm, mask, row_list, col_list)

    data['mask'][data['mask'] >= 0.5] = 1
    data['mask'][data['mask'] < 0.5] = 0

    # plt.close('all')
    # plt.imshow(data['mask'][1, :, :])
    # plt.show()
    # plt.imshow(data['dtm'][1, :, :])
    # plt.show()
    # plt.imshow(data['mask'][65, :, :])
    # plt.show()
    # plt.imshow(data['dtm'][65, :, :])
    # plt.show()

    # plt.imshow(data['mask'][((120*4)+65), :, :])
    # plt.show()
    # plt.imshow(data['dtm'][((120*4)+65), :, :])
    # plt.show()

    data['dtm'] = np.expand_dims(data['dtm'], axis=3)
    data['mask'] = np.expand_dims(data['mask'], axis=3)

    train_model(data['dtm'], data['mask'],
                model_fname='model.h5', N=128, channels=1)
