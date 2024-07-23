---
annotations_creators: []
license:
- cc-by-4.0
- other
pretty_name: GigaMIDI
size_categories: []
source_datasets: []
tags: []
task_ids: []
---

# Dataset Card for GigaMIDI

## Table of Contents

- [Dataset Description](#dataset-description)
  - [Dataset Summary](#dataset-summary)
  - [How to use](#how-to-use)
- [Dataset Structure](#dataset-structure)
  - [Data Instances](#data-instances)
  - [Data Fields](#data-fields)
  - [Data Splits](#data-splits)
- [Dataset Creation](#dataset-creation)
  - [Curation Rationale](#curation-rationale)
  - [Source Data](#source-data)
  - [Annotations](#annotations)
  - [Personal and Sensitive Information](#personal-and-sensitive-information)
- [Considerations for Using the Data](#considerations-for-using-the-data)
  - [Social Impact of Dataset](#social-impact-of-dataset)
  - [Discussion of Biases](#discussion-of-biases)
  - [Other Known Limitations](#other-known-limitations)
- [Additional Information](#additional-information)
  - [Dataset Curators](#dataset-curators)
  - [Licensing Information](#licensing-information)
<!--  - [Citation Information](#citation-information) -->

## Dataset Description

<!-- - **Homepage:** https://metacreation.net/GigaMIDI -->
- **Repository:** https://github.com/Metacreation-Lab/GigaMIDI
<!-- - **Paper:** https://arxiv.org/abs/1912.06670  -->
- **Point of Contact:** [Keon Ju Maverick Lee](mailto:keon_maverick@sfu.ca)


*Note: special thanks to Nathan Fradet, who assisted in creating this HuggingFace webpage.

### Dataset Summary

The GigaMIDI dataset is a corpus of over 1 million MIDI files covering all music genres.

We provide two subsets: `drums`, which contain MIDI files exclusively containing drum tracks, and `music` for all others. The `all` subset encompasses both of them.

## How to use

The `datasets` library allows you to load and pre-process your dataset in pure Python at scale. The dataset can be downloaded and prepared in one call to your local drive by using the `load_dataset` function.

```python
from datasets import load_dataset

dataset = load_dataset("Metacreation/GigaMIDI", "music", trust_remote_code=True)
```

Using the datasets library, you can also stream the dataset on-the-fly by adding a `streaming=True` argument to the `load_dataset` function call. Loading a dataset in streaming mode loads individual samples of the dataset at a time, rather than downloading the entire dataset to disk.

```python
from datasets import load_dataset

dataset = load_dataset(
    "Metacreation/GigaMIDI", "music", trust_remote_code=True, streaming=True
)

print(next(iter(dataset)))
```

*Bonus*: create a [PyTorch dataloader](https://huggingface.co/docs/datasets/use_with_pytorch) directly with your own datasets (local/streamed).

### Local

```python
from datasets import load_dataset
from torch.utils.data.sampler import BatchSampler, RandomSampler

dataset = load_dataset("Metacreation/GigaMIDI", "music", trust_remote_code=True, split="train")
batch_sampler = BatchSampler(RandomSampler(dataset), batch_size=32, drop_last=False)
dataloader = DataLoader(dataset, batch_sampler=batch_sampler)
```

### Streaming

```python
from datasets import load_dataset
from torch.utils.data import DataLoader

dataset = load_dataset("Metacreation/GigaMIDI", "music", trust_remote_code=True, split="train")
dataloader = DataLoader(dataset, batch_size=32)
```

### Example scripts

MIDI files can be easily loaded and tokenized with [Symusic](https://github.com/Yikai-Liao/symusic) and [MidiTok](https://github.com/Natooz/MidiTok) respectively.

```python
from datasets import load_dataset

dataset = load_dataset("Metacreation/GigaMIDI", "music", trust_remote_code=True, split="train")
```

## Dataset Structure

### Data Instances

A typical data sample comprises the `md5` of the file which corresponds to its file name, a `music` entry containing dictionary mapping to its absolute file `path` and `bytes` that can be loaded with `symusic` as `score = Score.from_midi(dataset[sample_idx]["music"]["bytes"])`.
Metadata accompanies each file, which is introduced in the next section.

A data sample indexed from the dataset may look like this (the `bytes` entry is voluntarily shorten):

```python
{
    'md5': '0211bbf6adf0cf10d42117e5929929a4',
    'music': {'path': '/Users/nathan/.cache/huggingface/datasets/downloads/extracted/cc8e36bbe8d5ec7ecf1160714d38de3f2f670c13bc83e0289b2f1803f80d2970/0211bbf6adf0cf10d42117e5929929a4.mid', 'bytes': b"MThd\x00\x00\x00\x06\x00\x01\x00\x05\x01\x00MTrk\x00"},
    'is_drums': False,
    'sid_matches': {'sid': ['065TU5v0uWSQmnTlP5Cnsz', '29OG7JWrnT0G19tOXwk664', '2lL9TiCxUt7YpwJwruyNGh'], 'score': [0.711, 0.8076, 0.8315]},
    'mbid_matches': {'mbid': ['065TU5v0uWSQmnTlP5Cnsz', '29OG7JWrnT0G19tOXwk664', '2lL9TiCxUt7YpwJwruyNGh'], 'score': [0.711, 0.8076, 0.8315]},
    'artist_scraped': 'Bach, Johann Sebastian',
    'title_scraped': 'Contrapunctus 1 from Art of Fugue',
    'genres_scraped': ['classical', 'romantic'],
    'genres_discogs': {'genre': ['classical', 'classical---baroque'], 'count': [14, 1]},
    'genres_tagtraum': {'genre': ['classical', 'classical---baroque'], 'count': [1, 1]},
    'genres_lastfm': {'genre': [], 'count': []},
    'median_metric_depth': [0, 0, 0, 0]
}
```

### Data Fields

* `md5` (`string`): hash the MIDI file, corresponding to its file name;
* `music` (`dict`): a dictionary containing the absolute `path` to the downloaded file and the file content as `bytes`;
* `is_drums` (`boolean`): whether the sample comes from the `drums` subset, this can be useful when working with the `all` subset;
* `sid_matches` (`dict[str, list[str] | list[float16]]`):
* `mbid_matches` (`dict[str, list[str] | list[float16]]`):
* `artist_scraped` (`string`):
* `title_scraped` (`string`):
* `genres_scraped` (`list[str]`):
* `genres_discogs` (`dict[str, list[str] | list[int16]]`):
* `genres_tagtraum` (`dict[str, list[str] | list[int16]]`):
* `genres_lastfm` (`dict[str, list[str] | list[int16]]`):
* `median_metric_depth` (`list[int16]`):
<!--* `loop` (`string`): -->

### Data Splits

The speech material has been subdivided into portions for dev, train, test, validated, invalidated, reported and other.

The validated data has been validated with reviewers and received upvotes indicating that it is of high quality.

The invalidated data is data that has been invalidated by reviewers
and received downvotes indicating that the data is of low quality.

The reported data is data that has been reported for different reasons.

The other data is data that has not yet been reviewed.

The dev, test, and train are all data that have been reviewed, deemed of high quality, and split into dev, test, and train.

## Data Preprocessing Recommended by Hugging Face

The following are data preprocessing steps advised by the Hugging Face team. They are accompanied by an example code snippet that shows how to put them to practice.

Many examples in this dataset have trailing quotations marks, e.g _“the cat sat on the mat.“_. These trailing quotation marks do not change the actual meaning of the sentence, and it is near impossible to infer whether a sentence is a quotation or not a quotation from audio data alone. In these cases, it is advised to strip the quotation marks, leaving: _the cat sat on the mat_.

In addition, the majority of training sentences end in punctuation ( . or ? or ! ), whereas just a small proportion do not. In the dev set, **almost all** sentences end in punctuation. Thus, it is recommended to append a full-stop ( . ) to the end of the small number of training examples that do not end in punctuation.

```python
from datasets import load_dataset

ds = load_dataset("Metacreation/GigaMIDI", "music", trust_remote_code=True)

def prepare_dataset(batch):
  """Function to preprocess the dataset with the .map method"""
  transcription = batch["sentence"]

  if transcription.startswith('"') and transcription.endswith('"'):
    # we can remove trailing quotation marks as they do not affect the transcription
    transcription = transcription[1:-1]

  if transcription[-1] not in [".", "?", "!"]:
    # append a full-stop to sentences that do not end in punctuation
    transcription = transcription + "."

  batch["sentence"] = transcription

  return batch

ds = ds.map(prepare_dataset, desc="preprocess dataset")
```

## Dataset Creation

### Curation Rationale

[Needs More Information]

### Source Data

#### Initial Data Collection and Normalization

[Needs More Information]

#### Who are the source language producers?

[Needs More Information]

### Annotations

#### Annotation process

[Needs More Information]

#### Who are the annotators?

[Needs More Information]

### Personal and Sensitive Information

The dataset consists of people who have donated their voice online.  You agree to not attempt to determine the identity of speakers in the Common Voice dataset.

## Considerations for Using the Data

### Social Impact of Dataset

The dataset consists of people who have donated their voice online.  You agree to not attempt to determine the identity of speakers in the Common Voice dataset.

### Discussion of Biases

[More Information Needed]

### Other Known Limitations

[More Information Needed]

## Additional Information

### Dataset Curators

[More Information Needed]

### Licensing Information

Public Domain, [CC-0](https://creativecommons.org/share-your-work/public-domain/cc0/)

<!--### Citation Information

```
@inproceedings{commonvoice:2020,
  author = {Ardila, R. and Branson, M. and Davis, K. and Henretty, M. and Kohler, M. and Meyer, J. and Morais, R. and Saunders, L. and Tyers, F. M. and Weber, G.},
  title = {Common Voice: A Massively-Multilingual Speech Corpus},
  booktitle = {Proceedings of the 12th Conference on Language Resources and Evaluation (LREC 2020)},
  pages = {4211--4215},
  year = 2020
}
```-->
