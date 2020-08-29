local DATA_ROOT = "/tmp/allenrank/data/mimics-clickexplore/%s";
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
    "max_instances": 50000
  },
  "train_data_path": DATA_ROOT % "train.tsv",
  "validation_data_path": DATA_ROOT % "valid.tsv",
  // "test_data_path": DATA_ROOT % "test.tsv",
  "model": {
    "type": "ranker",
    "dropout": 0.35,
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "pretrained_transformer",
          "model_name": MODEL_NAME,
        }
      }
    },
    "relevance_matcher": {
      // "num_classes": 1,
      "input_dim": 128,
      
      // "type": "knrm",
      // "kernel_function": { "type": "gaussian", "n_kernels": 128 },

      "type": "bert_cls", 
      // "seq2vec_encoder": { "type": "cls_pooler", "embedding_dim": 128 }

      // "type": "matchpyramid",
      // "conv_output_size": [128, 128],
      // "conv_kernel_size": [[2,2], [3,3]],
      // "adaptive_pooling_size": [[2,2], [3,3]]
    }
  },
  "data_loader": {
    "type": "default",
    "batch_size" : 256
  },
  "trainer": {
    "num_epochs": 5,
    // "patience": 3,
    // "grad_norm": 5.0,
    "validation_metric": "+auc",
    "cuda_device": 0,
    "optimizer": {
      "type": "adam", // "huggingface_adamw",
      "lr": 0.0001,
      // "weight_decay": 0
    },

    "learning_rate_scheduler": {
        "type": "reduce_on_plateau",
        "factor": 0.5,
        "mode": "max",
        "patience": 0
    },
  }
}