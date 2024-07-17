#!/usr/bin/python3 python

"""Script to create the WebDataset for the GigaMIDI dataset."""

from __future__ import annotations

from typing import TYPE_CHECKING

from datasets import Dataset
from datasets.features.music import Music
from webdataset import ShardWriter

from utils.GigaMIDI.gigamidi import _SPLITS, _SUBSETS

if TYPE_CHECKING:
    from pathlib import Path


MAX_NUM_ENTRIES_PER_SHARD = 50000


def create_webdataset_gigamidi(main_data_dir_path: Path) -> None:
    """
    Create the WebDataset shard for the GigaMIDI dataset.

    :param main_data_dir_path: path of the directory containing the datasets.
    """
    dataset_path = main_data_dir_path / "GigaMIDI"
    webdataset_path = main_data_dir_path / "GigaMIDI_webdataset"

    # TODO load metadata
    """md5_mbid_match = json.load(
        (main_data_dir_path / "MMD_METADATA" / "MMD_md5_to_mbid.json").open()
    )
    md5_mbid_matching = json.load(
        (main_data_dir_path / "MMD_METADATA" / "midi_audio_matches.json").open()
    )

    md5_genres = {}
    key = "genre"
    with open(
        main_data_dir_path / "MMD_METADATA" / "MMD_scraped_genre.jsonl"
    ) as genres_file:
        for row in genres_file:
            entry = json.loads(row)
            if len(entry[key][0]) == 0:
                continue
            md5_genres[entry["md5"]] = [format_genre(genre) for genre in entry[key][0]]
    """

    num_shards = {}
    for subset in _SUBSETS:
        num_shards[subset] = {}
        for split in _SPLITS:
            files_paths = list((dataset_path / split / subset).glob("**/*.mid"))
            save_path = webdataset_path / subset / split
            save_path.mkdir(parents=True, exist_ok=True)
            with ShardWriter(
                f"{save_path!s}/GigaMIDI-{subset}-{split}-%04d.tar",
                maxcount=MAX_NUM_ENTRIES_PER_SHARD,
            ) as writer:
                for file_path in files_paths:
                    example = {
                        "__key__": file_path.stem,
                        "mid": file_path.open("rb").read(),  # bytes
                    }
                    writer.write(example)


def load_dataset(main_data_dir_path: Path, num_files_limit: int = 100) -> Dataset:
    """
    Load the dataset.

    :param main_data_dir_path: path of the directory containing the datasets.
    :param num_files_limit: maximum number of entries/files to retain.
    :return: dataset.
    """
    dataset_path = main_data_dir_path / "GigaMIDI"
    files_paths = list(dataset_path.glob("**/*.mid"))[:num_files_limit]
    return Dataset.from_dict(
        {"music": [str(path_) for path_ in files_paths]}
    ).cast_column("music", Music())


if __name__ == "__main__":
    from utils.utils import path_main_data_directory

    create_webdataset_gigamidi(path_main_data_directory())
