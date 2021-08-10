local utils = import 'utils.jsonnet';
local device_info = utils.devices(std.extVar('gpu'));
local train_data = std.extVar('train_data');
local tritrain_data = std.extVar('tritrain_data');
local test_data = std.extVar('test_data');
local vocab = std.extVar('vocab');

local arc_representation_dim = 300;
local tag_representation_dim = 300;
local hidden_dim = 300;
local num_layers = 4;

local train_dataset = utils.train_dataset_reader(train_data, tritrain_data);

local token_embedder =
  utils.glove {
    token_indexers+: {
      elmo: {
        type: 'elmo_characters',
      },
    },
    text_field_embedder+: {
      token_embedders+: {
        elmo: {
          type: 'elmo_token_embedder',
          options_file: 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json',
          weight_file: 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5',
          do_layer_norm: false,
          dropout: 0.1,
        },
      },
    },
    encoder_input_dim: super.encoder_input_dim + 1024,
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
      type: 'lstm',
      input_size: token_embedder.encoder_input_dim,
      hidden_size: hidden_dim,
      num_layers: num_layers,
      dropout: 0.32,
      bidirectional: true,
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
        ['.*weight_ih.*', { type: 'xavier_uniform' }],
        ['.*weight_hh.*', { type: 'orthogonal' }],
        ['arc_attention._weight_matrix', { type: 'xavier_uniform' }],
        ['arc_attention._bias', { type: 'zero' }],
        ['.*bias_ih.*', { type: 'zero' }],
        ['.*bias_hh.*', { type: 'lstm_hidden_bias' }],
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
      type: 'dense_sparse_adam',
      betas: [
        0.9,
        0.9,
      ],
    },
    learning_rate_scheduler: {
      type: 'reduce_on_plateau',
      mode: 'max',
      factor: 0.5,
      patience: 5,
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
