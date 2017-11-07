"""rna_fish_3d analysis."""

import os
import logging
import argparse
import errno

from dtoolcore import DataSet

from jicbioimage.core.image import Image
from jicbioimage.core.transform import transformation
from jicbioimage.core.io import AutoName, AutoWrite, DataManager, FileBackend

from find_spots import find_spots, annotate

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


@transformation
def identity(image):
    """Return the image as is."""
    return image


def analyse_file(fpath, output_directory):
    """Analyse a single file."""
    logging.info("Analysing file: {}".format(fpath))

    AutoName.directory = output_directory

    microscopy_collection = get_microscopy_collection(fpath)
    zstack = microscopy_collection.zstack()
    zstack = identity(zstack)

    locs = find_spots(zstack)
    annotation = annotate(zstack, locs)

    fpath = os.path.join(AutoName.directory, "annotated.png")
    with open(fpath, "wb") as fh:
        fh.write(annotation.png())




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
    parser.add_argument("input_dataset", help="Input dataset")
    parser.add_argument("output_dir", help="Output directory")
    parser.add_argument("-i", "--identifier", help="identifier")
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
    if os.path.isdir(args.input_dataset):
        if args.identifier:
            analyse_item(
                args.input_dataset,
                args.output_dir,
                args.identifier
            )
        else:
            analyse_dataset(
                args.input_dataset,
                args.output_dir
            )
    else:
        parser.error("{} not a directory".format(args.input_dataset))

if __name__ == "__main__":
    main()
