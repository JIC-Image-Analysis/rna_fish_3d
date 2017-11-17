import numpy as np
import skimage.morphology
import skimage.feature.peak

from jicbioimage.core.transform import transformation
from jicbioimage.core.util.array import normalise
from jicbioimage.transform import (
    max_intensity_projection,
)
from jicbioimage.illustrate import AnnotatedImage


@transformation
def threshold_abs(image, cutoff):
    return image > cutoff


@transformation
def white_tophat(image, radius):
    selem = skimage.morphology.disk(radius)
    return skimage.morphology.white_tophat(image, selem)


def white_tophat_3d(image, radius):
    selem = skimage.morphology.ball(radius)
    return skimage.morphology.white_tophat(image, selem)


def find_spots(zstack):
    zstack = white_tophat_3d(zstack, 5)
    intensity_2d = max_intensity_projection(zstack)
    intensity_2d = white_tophat(intensity_2d, 20)
    return skimage.feature.peak.peak_local_max(
            intensity_2d,
            threshold_abs=2500
    )



def annotate(zstack, locs):
    intensity_2d = max_intensity_projection(zstack)
    grayscale = normalise(intensity_2d) * 255
    canvas = AnnotatedImage.from_grayscale(grayscale)
    for pos in locs:
        canvas.draw_cross(
            pos,
            color=(255, 0, 255),
            radius=1
        )
    return canvas
