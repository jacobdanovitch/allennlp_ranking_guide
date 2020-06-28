{
  "dataset_reader": {
    "type": "MSMARCODatasetReader",
  },
  "train_data_path": "https://allennlp.s3.amazonaws.com/datasets/sst/train.jsonl",
  "validation_data_path": "https://allennlp.s3.amazonaws.com/datasets/sst/dev.jsonl",
  "test_data_path": "https://allennlp.s3.amazonaws.com/datasets/sst/test.jsonl",
  "model": {
    "type": "DocumentRanker",
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "embedding_dim": 300,
          "pretrained_file": "https://allennlp.s3.amazonaws.com/datasets/glove/glove.840B.300d.txt.gz",
          "trainable": false
        }
      }
    },
    "seq2vec_encoder": {
       "type": "lstm",
       "input_size": 300,
       "hidden_size": 512,
       "num_layers": 2
    }
  },
  "data_loader": {
    "type": "default",
    "batch_size" : 32
  },
  "trainer": {
    "num_epochs": 5,
    "patience": 1,
    "grad_norm": 5.0,
    "validation_metric": "+accuracy",
    "cuda_device": 0,
    "optimizer": {
      "type": "adam",
      "lr": 0.001
    }
  }
}