import os

from skimage.morphology import disk

from jicbioimage.core.io import AutoName, AutoWrite
from jicbioimage.core.transform import transformation
from jicbioimage.core.util.array import normalise
from jicbioimage.core.util.color import pretty_color_from_identifier
from jicbioimage.transform import (
    max_intensity_projection,
    remove_small_objects,
    dilate_binary,
    erode_binary,
    invert,
)
from jicbioimage.segment import (
    connected_components,
    watershed_with_seeds,
)
from jicbioimage.illustrate import AnnotatedImage


from flat_analysis import (
    white_tophat,
    threshold_abs,
)


@transformation
def fill_holes(image, size):
    autowrite_on = AutoWrite.on
    AutoWrite.on = False
    image = invert(image)
    image = remove_small_objects(image, size)
    image = invert(image)
    AutoWrite.on = AutoWrite
    return image


def generate_seeds(image):
    seeds = white_tophat(image, 10)

    seeds = threshold_abs(seeds, 1500)
    seeds = remove_small_objects(seeds, 50)

    selem = disk(5)
    seeds = dilate_binary(seeds, selem)

    return connected_components(seeds, background=0)


def generate_mask(image):
    mask = threshold_abs(image, 6500)
    mask = remove_small_objects(mask, 50)
    mask = fill_holes(mask, 5)

    selem = disk(3)
    mask = erode_binary(mask, selem)
    mask = dilate_binary(mask, selem)

    return mask


def annotate_segmentation(image, segmentation):

    grayscale = normalise(image) * 255
    canvas = AnnotatedImage.from_grayscale(grayscale)

    for i in segmentation.identifiers:
        region = segmentation.region_by_identifier(i)
        outline = region.inner.border.dilate()
        color = pretty_color_from_identifier(i)
        canvas.mask_region(outline, color=color)


    fpath = os.path.join(AutoName.directory, "segmentation.png")
    with open(fpath, "wb") as fh:
        fh.write(canvas.png())


def segment(zstack):
    image = max_intensity_projection(zstack)

    seeds = generate_seeds(image)
    mask = generate_mask(image)

    segmentation =  watershed_with_seeds(image, seeds=seeds, mask=mask)
    annotate_segmentation(image, segmentation)

    return segmentation
