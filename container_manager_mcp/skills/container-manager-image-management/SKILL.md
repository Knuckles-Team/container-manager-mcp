---
name: container-manager-images
description: Manages container images. Use for listing/pulling/removing/pruning images. Triggers - image ops, builds.
---

### Overview
Image lifecycle via MCP. Essential for container setup.

### Key Tools
- `list_images`: List all. Params: manager_type?, silent?, log_file?.
- `pull_image`: Pull image/tag. Params: image (required), tag="latest", platform?.
- `remove_image`: Remove by name/ID. Params: image (required), force?.
- `prune_images`: Clean unused. Params: all?.

### Usage Instructions
1. Parse image:tag; defaults to latest.
2. Chain: list -> pull -> run (from containers skill).

### Examples
- Pull: `pull_image` with image="nginx", tag="latest".
- Prune: `prune_images` with all=true.

### Error Handling
- Not found: Registry issues—check URL.
- In use: Use force or stop containers first.
