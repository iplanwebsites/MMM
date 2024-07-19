"""The GigaMIDI dataset."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import datasets

if TYPE_CHECKING:
    from collections.abc import Sequence

    from datasets.utils.file_utils import ArchiveIterable

_CITATION = ""
_DESCRIPTION = "A large-scale MIDI symbolic music dataset."
_HOMEPAGE = "https://github.com/Metacreation-Lab/GigaMIDI"
_LICENSE = "CC0, also see https://www.europarl.europa.eu/legal-notice/en/"
_SUBSETS = ["music", "drums"]
_SPLITS = ["train", "validation", "test"]
_BASE_DATA_DIR = "data/"
_N_SHARDS_FILE = _BASE_DATA_DIR + "n_shards.json"
_MUSIC_PATH = (
    _BASE_DATA_DIR + "{subset}/{split}/GigaMIDI_{subset}_{split}_{n_shard}.tar"
)
_METADATA_PATH = _BASE_DATA_DIR + "{subset}/metadata_{subset}_{split}.tsv"
_METADATA_FEATURES = {
    "sid_matches": datasets.Sequence(
        datasets.Sequence(
            {"sid": datasets.Value("string"), "score": datasets.Value("float16")}
        )
    ),
    # "sid_matched": datasets.Value("string"),
    "mbid_matches": datasets.Sequence(datasets.Value("string")),
    # "mbid_matched": datasets.Value("string"),
    "genres_scraped": datasets.Sequence(datasets.Value("string")),
    "genres_discogs": datasets.Sequence(
        {"genre": datasets.Value("string"), "count": datasets.Value("int16")}
    ),
    "genres_tagtraum": datasets.Sequence(
        {"genre": datasets.Value("string"), "count": datasets.Value("int16")}
    ),
    "genres_lastfm": datasets.Sequence(
        {"genre": datasets.Value("string"), "count": datasets.Value("int16")}
    ),
    "median_metric_depth": datasets.Sequence(datasets.Value("int16")),
    # "loops": datasets.Value("string"),
}
_VERSION = "1.0.0"


class GigaMIDIConfig(datasets.BuilderConfig):
    """BuilderConfig for GigaMIDI."""

    def __init__(self, name: Literal["music", "drums", "all"], **kwargs) -> None:
        """
        BuilderConfig for GigaMIDI.

        Args:
        ----
            name: `string` or `List[string]`:
                name of the dataset subset. Must be either "drums" for files containing
                only drum tracks, "music" for others or "all" for all.
            **kwargs: keyword arguments forwarded to super.

        """
        if name == "all":
            self.subsets = _SUBSETS
        else:
            self.subsets = [name]

        super().__init__(name=name, **kwargs)


class GigaMIDI(datasets.GeneratorBasedBuilder):
    """The GigaMIDI dataset."""

    VERSION = datasets.Version(_VERSION)
    BUILDER_CONFIGS = (
        datasets.BuilderConfig(
            name=name,
            version=datasets.Version(_VERSION),
        )
        for name in [*_SUBSETS, "all"]
    )
    DEFAULT_WRITER_BATCH_SIZE = 256

    def _info(self) -> datasets.DatasetInfo:
        features = datasets.Features(
            {
                "md5": datasets.Value("string"),
                "music": datasets.Music(),  # TODO test binary
                "is_drums": datasets.Value("bool"),
                **_METADATA_FEATURES,
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
            version=_VERSION,
        )

    def _split_generators(
        self, dl_manager: datasets.DownloadManager | datasets.StreamingDownloadManager
    ) -> list[datasets.SplitGenerator]:
        n_shards_path = Path(dl_manager.download_and_extract(_N_SHARDS_FILE))
        with n_shards_path.open() as f:
            n_shards = json.load(f)

        music_urls = defaultdict(dict)
        for split in _SPLITS:
            for subset in self.config.subsets:
                music_urls[subset][split] = [
                    _MUSIC_PATH.format(subset=subset, split=split, n_shard=i)
                    for i in range(n_shards[subset][split])
                ]

        meta_urls = defaultdict(dict)
        for split in _SPLITS:
            for subset in self.config.subsets:
                meta_urls[subset.name][split] = _METADATA_PATH.format(
                    subset=subset, split=split
                )

        # dl_manager.download_config.num_proc = len(urls)

        meta_paths = dl_manager.download_and_extract(meta_urls)
        music_paths = dl_manager.download(music_urls)

        local_extracted_music_paths = (
            dl_manager.extract(music_paths)
            if not dl_manager.is_streaming
            else {
                split: {
                    subset: [None] * len(music_paths[subset][split])
                    for subset in self.config.subsets
                }
                for split in _SPLITS
            }
        )

        return [
            datasets.SplitGenerator(
                name=split_name,
                gen_kwargs={
                    "music_shards": {
                        subset: [
                            dl_manager.iter_archive(shard) for shard in subset_shards
                        ]
                        for subset, subset_shards in music_paths[split_name].items()
                    },
                    "local_extracted_shards_paths": local_extracted_music_paths[
                        "train"
                    ],
                    "metadata_paths": meta_paths[split_name],
                },
            )
            for split_name in _SPLITS
        ]

    def _generate_examples(
        self,
        music_shards: dict[str, Sequence[ArchiveIterable]],
        local_extracted_shards_paths: dict[str, Sequence[dict]],
        metadata_paths: dict[str, Path],
    ) -> dict:
        if not (
            len(metadata_paths)
            == len(music_shards)
            == len(local_extracted_shards_paths)
        ):
            msg = "The number of subsets provided are not equals"
            raise ValueError(msg)

        for subset in self.config.subsets:
            if len(music_shards[subset]) != len(local_extracted_shards_paths[subset]):
                msg = "the number of shards must be equal to the number of paths"
                raise ValueError(msg)

            is_drums = subset == "drums"
            meta_path = metadata_paths[subset]
            with meta_path.open() as f:
                metadata = {x["id"]: x for x in csv.DictReader(f, delimiter="\t")}

            for music_shard, local_extracted_shard_path in zip(
                music_shards[subset], local_extracted_shards_paths[subset]
            ):
                for music_file_name, music_file in music_shard:
                    md5 = music_file_name.stem
                    path = (
                        local_extracted_shard_path / music_file_name
                        if local_extracted_shard_path
                        else music_file_name
                    )

                    yield (
                        md5,
                        {
                            "md5": md5,
                            "music": {"path": path, "bytes": music_file.read()},
                            "is_drums": is_drums,
                            **{
                                feature: metadata[md5][feature]
                                for feature in _METADATA_FEATURES
                            },
                        },
                    )
