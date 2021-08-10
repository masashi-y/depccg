{
  devices(device_indices): {
    device_ids: [
      std.parseInt(device)
      for device in std.split(device_indices, ',')
    ],
    use_multi_devices: std.length(self.device_ids) > 1,
  },
}