#!/usr/bin/python3 python

"""Script to compute the MIDI-audio matching for the MetaMIDI dataset."""

import json
from pathlib import Path

import networkx as nx
import numpy as np
import scipy.optimize
from miditok.constants import SCORE_LOADING_EXCEPTION
from symusic import Score
from tqdm import tqdm


def compute_midi_audio_match(data_path: Path, matches_file_path: Path) -> None:
    """
    Clean the MMD dataset in order to keep one MIDI entry per audio entry.

    The match scores between MIDIs and audios can be seen as a big non-connected
    bipartite weighted graph on which we will compute a matching to keep only pairs of
    distinct MIDIs and audios.
    The selected MIDIs will be saved in a json file to be used to tokenize them

    :param data_path: path containing the data to load.
    :param matches_file_path: path to the MIDI - audio matches file.
    """
    # Reads the MIDI-audio matches file and keeps valid MIDIS for matching
    b = nx.Graph()
    midis = []  # stores all midi filenames (md5s) for validation below
    with matches_file_path.open() as matches_file:
        matches_file.seek(0)
        next(matches_file)  # first line skipped
        for line in tqdm(
            matches_file, desc="Reading MMD match file / building the graph"
        ):
            midi_md5, score, audio_sid = line.split()
            midis.append(midi_md5)
            b.add_node(midi_md5, bipartite=0)
            b.add_node(audio_sid, bipartite=1)
            b.add_edge(midi_md5, audio_sid, weight=1 - float(score))

    # Removing invalid MIDIS
    midis = list(set(midis))
    for midi_md5 in tqdm(midis, desc="Checking MIDIs are valid"):
        try:
            _ = Score(data_path / midi_id_to_path(midi_md5))
        except SCORE_LOADING_EXCEPTION:
            b.remove_node(midi_md5)
    b.remove_nodes_from(list(nx.isolates(b)))

    # Computes matchings
    sub_graphs = [
        b.subgraph(nodes) for nodes in list(nx.connected_components(b))
    ]  # connected bipartite sub-graphs
    matchings = [
        match(sub_graph)
        for sub_graph in tqdm(sub_graphs, desc="Computing distinct MIDI-audio matches")
    ]

    # Sorts matchings
    midi_to_audio = {}
    for matching in tqdm(matchings, desc="Sorting matchings"):
        for key, value in matching.items():
            midi_md5, audio_sid = (key, value) if len(key) > 22 else (value, key)
            midi_to_audio[midi_md5] = audio_sid

    # Saves the matching file + conversion params, in txt format
    with (matches_file_path.parent / "midi_audio_matches.json").open("w") as outfile:
        json.dump(midi_to_audio, outfile, indent=2)


def midi_id_to_path(midi_md5: str) -> Path:
    """
    Return the relative path of a MIDI file from its file name (md5).

    :param midi_md5: MIDI file name (md5)
    :return: the relative path
    """
    return Path(midi_md5[0], midi_md5[1], midi_md5[2], midi_md5).with_suffix(".mid")


def match(graph: nx.Graph) -> dict:
    """
    Taken and fixed from nx.algorithms.bipartite.minimum_weight_full_matching.

    :param graph: connected bipartite graph
    :return: the matching
    """
    if len(graph) == 2:
        return {list(graph.nodes)[0]: list(graph.nodes)[1]}  # noqa:RUF015
    left, right = nx.bipartite.sets(graph, None)
    u = list(left)
    v = list(right)
    weights_sparse = nx.algorithms.bipartite.matrix.biadjacency_matrix(
        graph, row_order=u, column_order=v, weight="weight", format="coo"
    )
    weights = np.full(
        weights_sparse.shape,
        20e10,
        dtype=np.float32,
    )  # if this raises error, replace np.inf with a very large number
    weights[weights_sparse.row, weights_sparse.col] = weights_sparse.data
    left_matches = scipy.optimize.linear_sum_assignment(weights)
    return {u[ui]: v[vi] for ui, vi in zip(*left_matches)}


if __name__ == "__main__":
    from utils.utils import path_main_data_directory

    mmd_path = path_main_data_directory() / "MMD"
    matches_path = path_main_data_directory() / "MMD_METADATA" / "MMD_audio_matches.tsv"

    compute_midi_audio_match(mmd_path, matches_path)
