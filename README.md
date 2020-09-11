# Document Ranking with AllenNLP

Accompanying code for the AllenNLP guide post [here](https://guide.allennlp.org/document-ranking).

For continued development and updates, take a look at [allenrank](https://github.com/jacobdanovitch/allenrank).

## Usage

### Data

Run `python scripts/data_split.py "https://github.com/microsoft/MIMICS/raw/master/data/MIMICS-ClickExplore.tsv"` to automatically download the dataset to your `/tmp/` directory.

### Training

#### Docker

You can use the `docker-compose` file to start a docker container with the latest image of AllenNLP:

```shell
docker-compose run train
```

#### Manually

First, install the dependencies:

```shell
pip install -r requirements.txt
```

Then:

```shell
allennlp train experiments/mimics.jsonnet -s /tmp/your_output_dir
```
