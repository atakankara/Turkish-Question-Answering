from __future__ import absolute_import, division, print_function

import json

import datasets


# BibTeX citation
_CITATION = """
"""

_DESCRIPTION = """\
TSQuAD
"""

_URL = "https://raw.githubusercontent.com/TQuad/turkish-nlp-qa-dataset/master/"
_URLS = {
    "train": _URL + "train-v0.1.json",
    "dev": _URL + "dev-v0.1.json",
}


class SquadTrConfig(datasets.BuilderConfig):
    """BuilderConfig for TSQuAD."""

    def __init__(self, **kwargs):
        """BuilderConfig for TSQuAD.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(SquadTrConfig, self).__init__(**kwargs)


class SquadTr(datasets.GeneratorBasedBuilder):
    """TSQuAD dataset."""

    VERSION = datasets.Version("0.1.0")

    BUILDER_CONFIGS = [
        SquadTrConfig(
            name="v1.1.0",
            version=datasets.Version("1.0.0", ""),
            description="Plain text Turkish squad version 1",
        ),
    ]

    def _info(self):
        # Specifies the datasets.DatasetInfo object
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # datasets.features.FeatureConnectors
            features=datasets.Features(
                {
                    # These are the features of your dataset like images, labels ...
                    "id": datasets.Value("string"),
                    "title": datasets.Value("string"),
                    "context": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "answers": datasets.features.Sequence(
                        {
                            "text": datasets.Value("string"),
                            "answer_start": datasets.Value("int32"),
                        }
                    ),
                }
            ),
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=None,
            # Homepage of the dataset for documentation
            homepage="https://github.com/TQuad/turkish-nlp-qa-dataset",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # Downloads the data and defines the splits
        # dl_manager is a datasets.download.DownloadManager that can be used to

        # download and extract URLs
        dl_dir = dl_manager.download_and_extract(_URLS)
       
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={"filepath": dl_dir["train"]},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={"filepath": dl_dir["dev"]},
            ),
        ]

    def _generate_examples(self, filepath):
        """Yields examples."""
        # Yields (key, example) tuples from the dataset
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
            for example in data["data"]:
                title = example.get("title", "").strip()
                for paragraph in example["paragraphs"]:
                    context = paragraph["context"].strip()
                    for qa in paragraph["qas"]:
                        question = qa["question"].strip()
                        id_ = str(qa["id"])

                        answer_starts = [answer["answer_start"] for answer in qa["answers"]]
                        answers = [answer["text"].strip() for answer in qa["answers"]]

                        yield id_, {
                            "title": title,
                            "context": context,
                            "question": question,
                            "id": id_,
                            "answers": {
                                "answer_start": answer_starts,
                                "text": answers,
                            },
                        }