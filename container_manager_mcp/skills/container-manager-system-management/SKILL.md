---
name: container-manager-system
description: Manages system resources. Use for full prune. Triggers - cleanup, optimization.
---

### Overview
System-wide cleanup.

### Key Tools
- `prune_system`: Prune all unused (containers/images/volumes/networks). Params: force?, all?.

### Usage Instructions
1. Caution: Destructive—use force/all sparingly.

### Examples
- Prune: `prune_system` with all=true.

### Error Handling
- Partial failure: Check individual prunes.
