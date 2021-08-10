local gpu = std.parseInt(std.extVar('gpu'));
local char_embedding_dim = 100;
local char_embedded_dim = 200;
local afix_embedding_dim = 30;
local arc_representation_dim = 300;
local tag_representation_dim = 300;
local hidden_dim = 300;
local num_layers = 4;
local token_embedding_type = std.extVar('token_embedding_type');
local encoder_type = std.extVar('encoder_type');
local train_data = std.extVar('train_data');
local test_data = std.extVar('test_data');
local vocab = std.extVar('vocab');


local Glove = {
  token_indexers: {
    tokens: {
      type: 'single_id',
      lowercase_tokens: true,
    },
  },
  text_field_embedder: {
    token_embedders: {
      tokens: {
        type: 'embedding',
        pretrained_file: 'https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.100d.txt.gz',
        embedding_dim: 100,
        sparse: true,
      },
    },
  },
  encoder_input_dim: self.text_field_embedder.token_embedders.tokens.embedding_dim,
};

local TokenEmbedding =
  if token_embedding_type == 'char' then
    Glove {
      token_indexers+: {
        token_characters: {
          type: 'characters',
          character_tokenizer: { end_tokens: ['@@PADDING@@', '@@PADDING@@', '@@PADDING@@', '@@PADDING@@'] },
        },
      },
      text_field_embedder+: {
        token_embedders+: {
          token_characters: {
            type: 'character_encoding',
            embedding: {
              embedding_dim: char_embedding_dim,
              sparse: true,
              trainable: true,
            },
            encoder: {
              type: 'cnn',
              embedding_dim: char_embedding_dim,
              num_filters: char_embedded_dim,
              ngram_filter_sizes: [
                5,
              ],
            },
          },
        },
      },
      encoder_input_dim: super.encoder_input_dim + char_embedded_dim,
    }
  else if token_embedding_type == 'afix' then
    Glove {
      token_indexers+: {
        suffixes: {
          type: 'afix_ids',
          afix_type: 'suffix',
        },
        prefixes: {
          type: 'afix_ids',
          afix_type: 'prefix',
        },
      },
      text_field_embedder+: {
        token_embedders+: {
          suffixes: {
            type: 'afix_embedding',
            embedding: {
              embedding_dim: afix_embedding_dim,
              sparse: true,
              trainable: true,
            },
          },
          prefixes: {
            type: 'afix_embedding',
            embedding: {
              embedding_dim: afix_embedding_dim,
              sparse: true,
              trainable: true,
            },
          },
        },
      },
      encoder_input_dim: super.encoder_input_dim + afix_embedding_dim * 8,
    }
  else if token_embedding_type == 'elmo' then
    Glove {
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
    }
  else if token_embedding_type == 'bert' then
    {
      token_indexers: {
        bert: {
          type: "bert-pretrained",
          pretrained_model: "bert-base-cased",
          do_lowercase: false,
          use_starting_offsets: true
        }
      },
      text_field_embedder: {
        allow_unmatched_keys: true,
        embedder_to_indexer_map: {
          bert: ["bert", "bert-offsets"]
        },
        token_embedders: {
          bert: {
            type: "bert-pretrained",
            pretrained_model: "bert-base-cased"
          },
        },
      },
      encoder_input_dim: 768,
  }
  else error 'invalid token embedding type %s' % [token_embedding_type];


// encoder config
local Encoder =
  if encoder_type == 'lstm' then
    {
      type: 'lstm',
      input_size: TokenEmbedding.encoder_input_dim,
      hidden_size: hidden_dim,
      num_layers: num_layers,
      dropout: 0.32,
      bidirectional: true,
    }
  else if encoder_type == 'self_attention' then
    {
      type: 'stacked_self_attention',
      input_dim: TokenEmbedding.encoder_input_dim,
      hidden_dim: 512,
      projection_dim: 64 * 8,
      feedforward_hidden_dim: 512,
      num_layers: 8,
      num_attention_heads: 8,
      use_positional_encoding: true,
      dropout_prob: 0.1,
      residual_dropout_prob: 0.2,
      attention_dropout_prob: 0.1,
    }
  else if encoder_type == 'pass_through' then
    {
      type: 'pass_through',
      input_dim: TokenEmbedding.encoder_input_dim
    }
  else error 'invalid encoder type %s' % [encoder_type];


// main config
{
  vocabulary: {
    type: 'from_files',
    directory: vocab,
  },
  dataset_reader: {
    type: 'supertagging_dataset',
    lazy: true,
    token_indexers: TokenEmbedding.token_indexers,
  },
  validation_dataset_reader: {
    type: 'supertagging_dataset',
    lazy: true,
    token_indexers: TokenEmbedding.token_indexers,
  },
  train_data_path: train_data,
  validation_data_path: test_data,
  model: {
    type: 'supertagger',
    text_field_embedder: TokenEmbedding.text_field_embedder,
    encoder: Encoder,
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
      // sorting_keys: [ 'words', 'num_tokens' ],
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
    cuda_device: gpu,
  },
}
