# Document Ranking with AllenNLP

An AllenNLP project created with [cookiecutter-allennlp](https://github.com/jacobdanovitch/cookiecutter-allennlp).

## Installation

Install dependencies by running `pip install -r requirements.txt`.

## Training

```bash
allennlp train experiments/venue_classifier.json -s /tmp/your_output_dir_here --include-package my_library
```

<hr/>

This project was created using [cookiecutter-allennlp](https://github.com/jacobdanovitch/cookiecutter-allennlp), which is based on [allennlp-as-a-library-example](https://github.com/allenai/allennlp-as-a-library-example).
