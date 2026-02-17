---
name: container-manager-networks
description: Manages networks. Use for listing/creating/removing/pruning. Triggers - isolation, connectivity.
---

### Overview
Network isolation for containers.

### Key Tools
- `list_networks`: List all.
- `create_network`: Create. Params: name (required), driver="bridge".
- `remove_network`: Remove. Params: network_id (required).
- `prune_networks`: Clean unused.

### Usage Instructions
1. Default driver: bridge.

### Examples
- Create: `create_network` with name="my-net".

### Error Handling
- In use: Disconnect containers.
