"""rna_fish_3d analysis."""

import os
import logging
import argparse
import errno

from dtoolcore import DataSet

from jicbioimage.core.io import AutoName, AutoWrite, DataManager, FileBackend
from jicbioimage.transform import max_intensity_projection

from flat_analysis import find_spots, annotate
#from cell_segmentation import segment

__version__ = "0.1.0"

AutoName.prefix_format = "{:03d}_"


def get_microscopy_collection(input_file):
    """Return microscopy collection from input file."""
    data_dir = "output"
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)
    backend_dir = os.path.join(data_dir, '.backend')
    file_backend = FileBackend(backend_dir)
    data_manager = DataManager(file_backend)
    microscopy_collection = data_manager.load(input_file)
    return microscopy_collection


def safe_mkdir(directory):
    """Create directories if they do not exist."""
    try:
        os.makedirs(directory)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(directory):
            pass
        else:
            raise


def item_output_path(output_directory, rel_path):
    """Return item output path; and create it if it does not already exist."""
    abs_path = os.path.join(output_directory, rel_path)
    safe_mkdir(abs_path)
    return abs_path


def analyse_channel(microscopy_collection, channel_id):
    zstack = microscopy_collection.zstack(c=channel_id)

    cleaned_proj, locs = find_spots(zstack)
    annotation = annotate(cleaned_proj, locs)

    fpath = os.path.join(
        AutoName.directory,
        "enhanced_annotated_channel_{}.png".format(channel_id)
    )
    with open(fpath, "wb") as fh:
        fh.write(annotation.png())


def analyse_file(fpath, output_directory):
    """Analyse a single file."""
    logging.info("Analysing file: {}".format(fpath))

    AutoName.directory = output_directory

    microscopy_collection = get_microscopy_collection(fpath)

    # Write out a max projection of the DAPI channel.
    dapi_zstack = microscopy_collection.zstack(c=2)
    dapi_image = max_intensity_projection(dapi_zstack)
    fpath = os.path.join(
        AutoName.directory,
        "dapi_channel_2.png"
    )
    with open(fpath, "wb") as fh:
        fh.write(dapi_image.png())

#   segmentation = segment(dapi_zstack)

    for channel_id in [0, 1]:
        analyse_channel(microscopy_collection, channel_id)


def analyse_item(dataset_dir, output_dir, identifier):
    dataset = DataSet.from_uri(dataset_dir)
    data_item_abspath = dataset.item_content_abspath(identifier)
    item_info = dataset.item_properties(identifier)
    specific_output_dir = item_output_path(output_dir, item_info["relpath"])
    analyse_file(data_item_abspath, specific_output_dir)


def analyse_dataset(dataset_dir, output_dir):
    """Analyse all the files in the dataset."""
    dataset = DataSet.from_uri(dataset_dir)
    logging.info("Analysing items in dataset: {}".format(dataset.name))

    for i in dataset.identifiers:
        analyse_item(dataset_dir, output_dir, i)


def main():
    # Parse the command line arguments.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_fpath", help="Input fpath")
    parser.add_argument("output_dir", help="Output directory")
    parser.add_argument("--debug", default=False, action="store_true",
                        help="Write out intermediate images")
    args = parser.parse_args()

    # Create the output directory if it does not exist.
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
    AutoName.directory = args.output_dir

    # Only write out intermediate images in debug mode.
    if not args.debug:
        AutoWrite.on = False

    # Setup a logger for the script.
    log_fname = "audit.log"
    log_fpath = os.path.join(args.output_dir, log_fname)
    logging_level = logging.INFO
    if args.debug:
        logging_level = logging.DEBUG
    logging.basicConfig(filename=log_fpath, level=logging_level)

    # Log some basic information about the script that is running.
    logging.info("Script name: {}".format(__file__))
    logging.info("Script version: {}".format(__version__))

    # Run the analysis.
    analyse_file(args.input_fpath, args.output_dir)

if __name__ == "__main__":
    main()
