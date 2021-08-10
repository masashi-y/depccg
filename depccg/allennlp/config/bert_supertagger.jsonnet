local utils = import 'utils.jsonnet';
local device_info = utils.devices(std.extVar('gpu'));
local train_data = std.extVar('train_data');
local tritrain_data = std.extVar('tritrain_data');
local test_data = std.extVar('test_data');
local vocab = std.extVar('vocab');

local arc_representation_dim = 300;
local tag_representation_dim = 300;
local transformer_model = "bert-base-cased";
local max_length = 512;

local train_dataset = utils.train_dataset_reader(train_data, tritrain_data);

local token_embedder =
  {
    token_indexers: {
      tokens: {
        type: "pretrained_transformer_mismatched",
        model_name: transformer_model,
        max_length: max_length,
      }
    },
    text_field_embedder: {
      token_embedders: {
        tokens: {
          type: "pretrained_transformer_mismatched",
          model_name: transformer_model,
          max_length: max_length
        }
      }
    },
    encoder_input_dim: 768,
  };

{
  vocabulary: {
    type: 'from_files',
    directory: vocab,
  },
  dataset_reader: train_dataset.dataset_reader + {
    token_indexers: token_embedder.token_indexers,
  },
  validation_dataset_reader: {
    type: 'supertagging_dataset',
    token_indexers: token_embedder.token_indexers,
  },
  train_data_path: train_dataset.train_data_path,
  validation_data_path: test_data,
  model: {
    type: 'supertagger',
    text_field_embedder: token_embedder.text_field_embedder,
    encoder: {
      type: 'pass_through',
      input_dim: token_embedder.encoder_input_dim,
    },
    tag_representation_dim: tag_representation_dim,
    arc_representation_dim: arc_representation_dim,
    dropout: 0.32,
    input_dropout: 0.5,
    initializer: {
      regexes: [
        ['.*feedforward.*weight', { type: 'xavier_uniform' }],
        ['.*feedforward.*bias', { type: 'zero' }],
        ['.*tag_bilinear.bias', { type: 'zero' }],
        ['.*tag_bilinear.*', { type: 'xavier_uniform' }],
        ['arc_attention._weight_matrix', { type: 'xavier_uniform' }],
        ['arc_attention._bias', { type: 'zero' }],
      ]
    }
  },
  data_loader: {
    batch_sampler: {
      type: 'bucket',
      batch_size: 32,
      sorting_keys: ['words'],
    },
  },
  trainer: {
    optimizer: {
      type: "huggingface_adamw",
      weight_decay: 0.01,
      parameter_groups: [[["bias", "LayerNorm\\.weight", "layer_norm\\.weight"], {"weight_decay": 0}]],
      lr: 1e-5,
      eps: 1e-8,
      correct_bias: true,
    },
    learning_rate_scheduler: {
      type: "linear_with_warmup",
      warmup_steps: 100,
    },
    validation_metric: '+harmonic_mean',
    grad_norm: 5,
    num_epochs: 100,
    patience: 20,
    [if !device_info.use_multi_devices then 'cuda_device']: device_info.device_ids[0],
  },
  [if device_info.use_multi_devices then 'distributed']: {
    cuda_devices: device_info.device_ids,
    num_nodes: 1,
  },
}
