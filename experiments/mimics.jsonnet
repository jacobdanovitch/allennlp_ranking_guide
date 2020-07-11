// local DATA_ROOT = "/tmp/allenrank/data/mimics-clickexplore/%s";
local DATA_ROOT = "/scratch/jacobgdt/allenrank/data/mimics-clickexplore/%s";
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
      
      "type": "knrm",
      "n_kernels": 256,

      // "type": "bert_cls", 
      // "seq2vec_encoder": { "type": "cls_pooler", "embedding_dim": 128 }

      // "type": "matchpyramid",
      // "conv_output_size": [128, 128],
      // "conv_kernel_size": [[2,2], [3,3]],
      // "adaptive_pooling_size": [[2,2], [3,3]]
    }
  },
  "data_loader": {
    "type": "default",
    "batch_size" : 128
  },
  'distributed': {
      "cuda_devices": [0,1],
  },
  "trainer": {
    "num_epochs": 5,
    "patience": 2,
    "grad_norm": 5.0,
    "validation_metric": "+ndcg",
    // "cuda_device": 0,
    "optimizer": {
      "type": "adam", // "huggingface_adamw",
      "lr": 0.00075
    }
  }
}