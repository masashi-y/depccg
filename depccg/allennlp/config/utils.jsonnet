{
  train_dataset_reader(train_data_path, tritrain_data_path=''): {
    dataset_reader:
      if tritrain_data_path == '' then 
    {
      type: 'supertagging_dataset',
    }
      else
    {
      type: 'tritrain_supertagging_dataset',
      tritrain_path: tritrain_data_path,
    },
    train_data_path: train_data_path,
  },

  devices(device_indices): {
    device_ids: [
      std.parseInt(device)
      for device in std.split(device_indices, ',')
    ],
    use_multi_devices: std.length(self.device_ids) > 1,
  },

  glove: {
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
    encoder_input_dim: 100,
  }
}