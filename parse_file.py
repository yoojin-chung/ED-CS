"""Reconstruct pathfinder data from INI format."""

import configparser
import pandas as pd


def parse_ini_config(FN, if_data):
    """Parse metadata from INI format. Parse data if requested."""
    with open(FN) as fp:
        lines = fp.readlines()
        k = lines.index('[DATA]\n') if '[DATA]\n' in lines else len(lines)
        metadata = parse_metadata(''.join(lines[:k]))
        loc = parse_data(lines[k+1:]) if if_data else pd.DataFrame()
    return metadata, loc


def parse_metadata(s):
    """Parse metadata using configparser."""
    metadata = configparser.ConfigParser()
    metadata.read_string(s)
    return metadata


def parse_data(lines):
    """Construct dataframe from xyz location data."""
    if lines:
        data = [line.rstrip().split(',') for line in lines]
        loc = pd.DataFrame(data[1:], columns=data[0])
    else:
        loc = pd.DataFrame()
    return loc
