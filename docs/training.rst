.. _mmm-training-label:

===================================
MMM training
===================================

This page presents how to use this package to train a MMM model.
The only thing needed is to use the provided :class:`mmm.DatasetMMM` dataset class to load and tokenize data with the right format. This class:

1. Loads the music file;
2. Shuffles the order of the tracks, and randomly keeps a certain number of tracks (``ratio_random_tracks_range``);
3. Performs data augmentation by randomly shifting the pitches, velocities and duration values;
4. Select randomly whether to perform bar infilling or to generate a new track (``bar_fill_ratio``);
5. Reduce the sequence length so that it does not exceed the limit (``max_seq_len``);
6. If bar infilling: randomly selects the portion supposed to be infilled and move the associated tokens at the end of the sequence for the model to learn;

The DatasetMMM object
-----------------------------

.. autoclass:: mmm.DatasetMMM
    :members:
