local MODEL_NAME = "google/bert_uncased_L-2_H-128_A-2";

{
  "dataset_reader": {
    "type": "mimics",
    "tokenizer": {
      "type": "pretrained_transformer",
      "model_name": MODEL_NAME,
    },
    "token_indexers": {
      "tokens": {
        "type": "pretrained_transformer",
        "model_name": MODEL_NAME,
      }
    },
    "max_instances": 25000
  },
  "train_data_path": "/tmp/allenrank/data/mimics-clickexplore/train.tsv",
  // "train_data_path": "https://raw.githubusercontent.com/microsoft/MIMICS/master/data/MIMICS-Manual.tsv",
  // "validation_data_path": "https://allennlp.s3.amazonaws.com/datasets/sst/dev.jsonl",
  // "test_data_path": "https://allennlp.s3.amazonaws.com/datasets/sst/test.jsonl",
  "model": {
    "type": "ranker",
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "pretrained_transformer",
          "model_name": MODEL_NAME,
        }
      }
    },
    "relevance_matcher": {
      "num_classes": 1,
      "input_dim": 128,
      
      // "type": "knrm",
      // "n_kernels": 100

      "type": "bert_cls", 
      "seq2vec_encoder": { "type": "cls_pooler", "embedding_dim": 128 }
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
      "type": "huggingface_adamw",
      "lr": 0.0005
    }
  }
}