# Document Ranking with AllenNLP

An implementation of models for document ranking in AllenNLP.

## Training

### Docker

You can use the `docker-compose` file to start a docker container with the latest image of AllenNLP:

```shell
docker-compose run [train/version]
```

### Manually

First, install the dependencies:

```shell
pip install -r requirements.txt
```

Then:

```shell
allennlp train experiments/mimics.jsonnet -s /tmp/your_output_dir
```

<hr/>

This project was created using [cookiecutter-allennlp](https://github.com/jacobdanovitch/cookiecutter-allennlp), which is based on [allennlp-as-a-library-example](https://github.com/allenai/allennlp-as-a-library-example).
