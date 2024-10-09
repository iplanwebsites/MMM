#!/usr/bin/python3 python

"""Script to create the WebDataset for the GigaMIDI dataset."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from datasets import Dataset
from miditok.constants import SCORE_LOADING_EXCEPTION
from symusic import Score
from symusic.core import TextMetaSecond
from tqdm import tqdm
from webdataset import ShardWriter

from scripts.utils.GigaMIDI.GigaMIDI import _SPLITS

if TYPE_CHECKING:
    from pathlib import Path


MAX_NUM_ENTRIES_PER_SHARD = 50000
SUBSET_PATHS = {
    "all-instruments-with-drums": "drums+music",
    "drums-only": "drums",
    "no-drums": "music",
}


def create_webdataset_gigamidi(main_data_dir_path: Path) -> None:
    """
    Create the WebDataset shard for the GigaMIDI dataset.

    :param main_data_dir_path: path of the directory containing the datasets.
    """
    dataset_path = main_data_dir_path / "GigaMIDI_original"
    webdataset_path = main_data_dir_path / "GigaMIDI"

    # Load metadata
    md5_sid_matches_scores = {}
    with (
        main_data_dir_path / "MMD_METADATA" / "MMD_audio_matches.tsv"
    ).open() as matches_file:
        matches_file.seek(0)
        next(matches_file)  # first line skipped
        for line in tqdm(matches_file, desc="Reading MMD match file"):
            midi_md5, score, audio_sid = line.split()
            if midi_md5 not in md5_sid_matches_scores:
                md5_sid_matches_scores[midi_md5] = []
            md5_sid_matches_scores[midi_md5].append((audio_sid, float(score)))
    sid_to_mbid = json.load(
        (main_data_dir_path / "MMD_METADATA" / "MMD_sid_to_mbid.json").open()
    )

    md5_genres = {}
    with (
        main_data_dir_path / "MMD_METADATA" / "MMD_audio_matched_genre.jsonl"
    ).open() as file:
        for row in tqdm(file, desc="Reading genres MMD metadata"):
            entry = json.loads(row)
            md5 = entry.pop("md5")
            md5_genres[md5] = entry
    md5_genres_scraped = {}
    with (
        main_data_dir_path / "MMD_METADATA" / "MMD_scraped_genre.jsonl"
    ).open() as file:
        for row in tqdm(file, desc="Reading scraped genres MMD metadata"):
            entry = json.loads(row)
            genres = []
            for genre_list in entry["genre"]:
                genres += genre_list
            md5_genres_scraped[entry["md5"]] = genres
    md5_artist_title_scraped = {}
    with (
        main_data_dir_path / "MMD_METADATA" / "MMD_scraped_title_artist.jsonl"
    ).open() as file:
        for row in tqdm(file, desc="Reading scraped titles/artists MMD metadata"):
            entry = json.loads(row)
            md5_artist_title_scraped[entry["md5"]] = entry["title_artist"][0]
    md5_expressive = {}
    with (
        dataset_path / "Expressive_Performance_Detection_NOMML_gigamidi_tismir.csv"
    ).open() as file:
        file.seek(0)
        next(file)  # skipping first row (header)
        for row in tqdm(file, desc="Reading expressiveness metadata"):
            parts = row.split(",")
            md5 = parts[0].split("/")[-1].split(".")[0]
            if md5 not in md5_expressive:
                md5_expressive[md5] = []
            md5_expressive[md5].append(int(parts[5]))
    md5_loop = {}
    with (
        dataset_path
        / "GigaMIDI-combined-non-expressive-loop-data"
        / "GigaMIDI-combined-non-expressive-loop-dataset.csv"
    ).open() as file:
        file.seek(0)
        next(file)  # skipping first row (header)
        for row in tqdm(file, desc="Reading loops metadata"):
            parts = row.split(",")
            md5 = parts[0].split("/")[-1].split(".")[0]
            if md5 not in md5_loop:
                md5_loop[md5] = []
            md5_loop[md5].append((parts[1], *parts[4:6]))

    # Sharding the data into tar archives
    num_shards = {}
    for subset, subset_path in SUBSET_PATHS.items():
        num_shards[subset] = {}
        for split in _SPLITS:
            files_paths = list((dataset_path / split / subset_path).glob("**/*.mid"))
            save_path = webdataset_path / subset / split
            save_path.mkdir(parents=True, exist_ok=True)
            metadata = {}
            with ShardWriter(
                f"{save_path!s}/GigaMIDI_{subset}_{split}_%04d.tar",
                maxcount=MAX_NUM_ENTRIES_PER_SHARD,
            ) as writer:
                for file_path in files_paths:
                    md5 = file_path.stem
                    example = {
                        "__key__": md5,
                        "mid": file_path.open("rb").read(),  # bytes
                    }
                    writer.write(example)

                    # Get metadata if existing
                    metadata_row = {}

                    sid_matches = md5_sid_matches_scores.get(md5)
                    if sid_matches:
                        metadata_row["sid_matches"] = sid_matches
                        metadata_row["mbid_matches"] = []
                        for sid, _ in sid_matches:
                            mbids = sid_to_mbid.get(sid, None)
                            if mbids:
                                metadata_row["mbid_matches"].append([sid, mbids])

                    title_artist = md5_artist_title_scraped.get(md5)
                    if title_artist:
                        (
                            metadata_row["title_scraped"],
                            metadata_row["artist_scraped"],
                        ) = title_artist
                    genres_scraped = md5_genres_scraped.get(md5)
                    if genres_scraped:
                        metadata_row["genres_scraped"] = genres_scraped
                    genres = md5_genres.get(md5)
                    if genres:
                        for key, val in genres.items():
                            metadata_row[f"genres_{key.split('_')[1]}"] = val
                    interpreted_scores = md5_expressive.get(md5)
                    if interpreted_scores:
                        metadata_row["median_metric_depth"] = interpreted_scores
                    loops = md5_loop.get(md5)
                    if loops:
                        try:
                            score = Score(file_path, ttype="second")
                        except SCORE_LOADING_EXCEPTION:
                            continue  # no loops saved for corrupted files
                        score.markers = []  # delete all existing ones for simplicity
                        for track_id, start_sec, end_sec in loops:
                            score.markers.append(
                                TextMetaSecond(
                                    float(start_sec), f"#LOOP_{int(track_id)}_on"
                                )
                            )
                            score.markers.append(
                                TextMetaSecond(
                                    float(end_sec), f"#LOOP_{int(track_id)}_off"
                                )
                            )
                        score = score.to(ttype="tick")
                        loops_ticks = []
                        loops_on = {}  # {track_id: start_tick}
                        for marker in score.markers:
                            _, track_id, state = marker.text.split("_")
                            if state == "on":
                                loops_on[int(track_id)] = marker.time
                            else:
                                loops_ticks.append(
                                    (
                                        int(track_id),
                                        loops_on.pop(int(track_id)),
                                        marker.time,
                                    )
                                )
                        metadata_row["loops"] = loops_ticks
                    if len(metadata_row) > 0:
                        metadata[md5] = metadata_row

            num_shards[subset][split] = len(list(save_path.glob("*.tar")))
            # Saving metadata for this subset and split
            """
            with (save_path.parent / f"metadata_{subset}_{split}.csv").open("w") as f:
                writer = csv.writer(f)
                writer.writerow(["md5", ])  # TODO header
                for row in metadata:
                    writer.writerow([row])"""
            with (save_path.parent / f"metadata_{subset}_{split}.json").open("w") as f:
                json.dump(metadata, f)

    # Saving n shards
    with (webdataset_path / "n_shards.json").open("w") as f:
        json.dump(num_shards, f, indent=4)


def load_dataset_from_generator(
    dataset_path: Path, num_files_limit: int = 100
) -> Dataset:
    """
    Load the dataset.

    :param dataset_path: path of the directory containing the datasets.
    :param num_files_limit: maximum number of entries/files to retain.
    :return: dataset.
    """
    files_paths = list(dataset_path.glob("**/*.mid"))[:num_files_limit]
    return Dataset.from_dict({"music": [str(path_) for path_ in files_paths]})


if __name__ == "__main__":
    from argparse import ArgumentParser

    from datasets import load_dataset

    from scripts.utils.utils import path_data_directory_local_fs

    parser = ArgumentParser(description="Dataset creation script")
    parser.add_argument(
        "--hf-repo-name", type=str, required=False, default="Metacreation/GigaMIDI"
    )
    parser.add_argument("--hf-token", type=str, required=False, default=None)
    args = vars(parser.parse_args())

    # Load all the raw data (MIDI files, csv and json metadata)
    create_webdataset_gigamidi(path_data_directory_local_fs())

    # Convert the webdataset into parquet
    for subset_name in SUBSET_PATHS:
        dataset_ = load_dataset(
            str(path_data_directory_local_fs() / "GigaMIDI"),
            subset_name,
        )
        for split, subset in dataset_.items():
            subset.to_parquet(
                path_data_directory_local_fs()
                / "GigaMIDI"
                / "parquet"
                / subset_name
                / f"{split}.parquet"
            )

    """from datasets import load_dataset

    dataset_ = load_dataset(
        str(path_data_directory_local_fs() / "GigaMIDI"),
        "no-drums",
        subsets=["no-drums", "all-instruments-with-drums"],
        trust_remote_code=True,
    )
    data = dataset_["train"]
    for i in range(7):
        t = data[i]
        f = 0

    test = data[0]
    print(test)
    from symusic import Score

    score = Score.from_midi(test["music"]["bytes"])
    t = 0"""

    """
    import requests

    headers = {"Authorization": f"Bearer {args['hf_token']}"}
    API_URL = "https://datasets-server.huggingface.co/is-valid?dataset=Metacreation/GigaMIDI"


    def query():
        response = requests.get(API_URL, headers=headers)
        return response.json()


    data = query()"""
