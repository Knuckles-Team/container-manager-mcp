from container_manager_mcp.container_manager import DockerManager

try:
    m = DockerManager(silent=True)
    containers = m.list_containers(all=True)
    print("CONTAINERS:", containers)
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
