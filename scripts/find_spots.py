"""Module with code for finding spots in 3D."""

import os

import numpy as np

from scipy.ndimage.filters import (
    gaussian_filter,
    sobel,
)

import skimage.measure
import skimage.feature
import skimage.morphology

from jicbioimage.core.image import Image, Image3D
from jicbioimage.core.transform import transformation
from jicbioimage.core.util.array import normalise
from jicbioimage.core.io import AutoName

from jicbioimage.transform import max_intensity_projection
from jicbioimage.illustrate import AnnotatedImage

from utils import PrettyColorImage3D


@transformation
def identity(image):
    return image


@transformation
def abs_threshold_filter(z_stack, threshold):
    filtered = z_stack > threshold
    return filtered.view(Image3D)

@transformation
def smooth_gaussian(z_stack, sigma):
    smoothed_array = np.zeros(z_stack.shape)
    gaussian_filter(z_stack, sigma, output=smoothed_array)
    return smoothed_array.view(Image3D)


@transformation
def sobel_magnitude_nd(z_stack):
    """Find edges."""

    n_dim = len(z_stack.shape)

    directional_filters = [np.zeros(z_stack.shape) for _ in range(n_dim)]

    [sobel(z_stack, axis, directional_filters[axis])
     for axis, single_filter
     in enumerate(directional_filters)]

    magnitude = np.sqrt(sum(single_axis ** 2
                            for single_axis
                            in directional_filters))

    return magnitude.view(Image3D)


@transformation
def match_template(z_stack, template, pad_input=True):
    match_result = skimage.feature.match_template(
        z_stack,
        template,
        pad_input=pad_input)
    return match_result.view(Image3D)


@transformation
def filter_template_matches(z_stack, match_thresh):
    locs = z_stack > match_thresh
    return locs.view(Image3D)


@transformation
def connected_components(image, connectivity=2, background=None):
    # Work around skimage.measure.label behaviour in version 0.12 and higher
    # treats all 0 pixels as background even if "background" argument is set
    # to None.
    if background is None:
        image[np.where(image == 0)] = np.max(image) + 1

    ar = skimage.measure.label(image, connectivity=connectivity,
                               background=background)

    # The :class:`jicbioimage.core.image.SegmentedImage` assumes that zero is
    # background.  So we need to change the identifier of any pixels that are
    # marked as zero if there is no background in the input image.
    if background is None:
        ar[np.where(ar == 0)] = np.max(ar) + 1
    else:
        if np.min(ar) == -1:
            # Work around skimage.measure.label behaviour pre version 0.12.
            # Pre version 0.12 the background in skimage was labeled -1 and the
            # first component was labelled with 0.
            # The jicbioimage.core.image.SegmentedImage assumes that the
            # background is labelled 0.
            ar[np.where(ar == 0)] = np.max(ar) + 1
            ar[np.where(ar == -1)] = 0

    return ar.view(PrettyColorImage3D)


def annotate(zstack, locs):
    intensity_2d = max_intensity_projection(zstack)
    locs_2d = max_intensity_projection(locs).astype(bool)
    grayscale = normalise(intensity_2d) * 255
    canvas = AnnotatedImage.from_grayscale(grayscale)
    canvas.mask_region(locs_2d, color=(255, 0, 255))
    return canvas


def find_spots_org(zstack):
    zstack = smooth_gaussian(zstack, sigma=1)
    edge_array = sobel_magnitude_nd(zstack)

    ball_template = skimage.morphology.ball(3)
    ball_template[3, 3, 3] = 0

    match_result = match_template(edge_array, ball_template, pad_input=True)

    best_score = np.max(match_result)
    px, py, pz = zip(*np.where(match_result == best_score))[0]
    tr = 4
    better_template = edge_array[
        px-tr:px+tr+1,
        py-tr:py+tr+1,
        pz-tr:pz+tr+1
    ]

    identity(better_template[:, :, 4].view(Image))

    stage_2_match = match_template(edge_array, better_template, pad_input=True)

    locs = filter_template_matches(stage_2_match, match_thresh=0.7)
    return locs


def generate_initial_template(radius, hole_offset):
    ball_template = skimage.morphology.ball(radius)
    ball_template[
        radius-hole_offset:radius+hole_offset+1,
        radius-hole_offset:radius+hole_offset+1,
        radius-hole_offset:radius+hole_offset+1
    ] = 0

    fpath = os.path.join(AutoName.directory, "initial_template.png")
    image = normalise(ball_template) * 255
    with open(fpath, "wb") as fh:
        fh.write(image[:, :, radius].view(Image).png())

    return ball_template


def generate_better_template(edge_array, match_result):
    best_score = np.max(match_result)
    px, py, pz = zip(*np.where(match_result == best_score))[0]
    tr = 4
    better_template = edge_array[
        px-tr:px+tr+1,
        py-tr:py+tr+1,
        pz-tr:pz+tr+1
    ]

    fpath = os.path.join(AutoName.directory, "better_template.png")
    image = normalise(better_template) * 255
    with open(fpath, "wb") as fh:
        fh.write(image[:, :, tr].view(Image).png())

    return better_template




def find_spots(zstack):
    ball_template = generate_initial_template(4, 1)

    edge_array = sobel_magnitude_nd(zstack)

    match_result = match_template(edge_array, ball_template, pad_input=True)

    better_template = generate_better_template(edge_array, match_result)

    stage_2_match = match_template(edge_array, better_template, pad_input=True)

    locs = filter_template_matches(stage_2_match, match_thresh=0.7)
    return locs


