# MCP_AGENTS.md - Dynamic Agent Registry

This file tracks the generated agents from MCP servers. You can manually modify the 'Tools' list to customize agent expertise.

## Agent Mapping Table

| Name | Description | System Prompt | Tools | Tag | Source MCP |
|------|-------------|---------------|-------|-----|------------|
| Container-Manager Log Specialist | Expert specialist for log domain tasks. | You are a Container-Manager Log specialist. Help users manage and interact with Log functionality using the available tools. | container-manager-mcp_log_toolset | log | container-manager-mcp |
| Container-Manager System Specialist | Expert specialist for system domain tasks. | You are a Container-Manager System specialist. Help users manage and interact with System functionality using the available tools. | container-manager-mcp_system_toolset | system | container-manager-mcp |
| Container-Manager Info Specialist | Expert specialist for info domain tasks. | You are a Container-Manager Info specialist. Help users manage and interact with Info functionality using the available tools. | container-manager-mcp_info_toolset | info | container-manager-mcp |
| Container-Manager Image Specialist | Expert specialist for image domain tasks. | You are a Container-Manager Image specialist. Help users manage and interact with Image functionality using the available tools. | container-manager-mcp_image_toolset | image | container-manager-mcp |
| Container-Manager Network Specialist | Expert specialist for network domain tasks. | You are a Container-Manager Network specialist. Help users manage and interact with Network functionality using the available tools. | container-manager-mcp_network_toolset | network | container-manager-mcp |
| Container-Manager Swarm Specialist | Expert specialist for swarm domain tasks. | You are a Container-Manager Swarm specialist. Help users manage and interact with Swarm functionality using the available tools. | container-manager-mcp_swarm_toolset | swarm | container-manager-mcp |
| Container-Manager Container Specialist | Expert specialist for container domain tasks. | You are a Container-Manager Container specialist. Help users manage and interact with Container functionality using the available tools. | container-manager-mcp_container_toolset | container | container-manager-mcp |
| Container-Manager Compose Specialist | Expert specialist for compose domain tasks. | You are a Container-Manager Compose specialist. Help users manage and interact with Compose functionality using the available tools. | container-manager-mcp_compose_toolset | compose | container-manager-mcp |
| Container-Manager Misc Specialist | Expert specialist for misc domain tasks. | You are a Container-Manager Misc specialist. Help users manage and interact with Misc functionality using the available tools. | container-manager-mcp_misc_toolset | misc | container-manager-mcp |
| Container-Manager Volume Specialist | Expert specialist for volume domain tasks. | You are a Container-Manager Volume specialist. Help users manage and interact with Volume functionality using the available tools. | container-manager-mcp_volume_toolset | volume | container-manager-mcp |

## Tool Inventory Table

| Tool Name | Description | Tag | Source |
|-----------|-------------|-----|--------|
| container-manager-mcp_log_toolset | Static hint toolset for log based on config env. | log | container-manager-mcp |
| container-manager-mcp_system_toolset | Static hint toolset for system based on config env. | system | container-manager-mcp |
| container-manager-mcp_info_toolset | Static hint toolset for info based on config env. | info | container-manager-mcp |
| container-manager-mcp_image_toolset | Static hint toolset for image based on config env. | image | container-manager-mcp |
| container-manager-mcp_network_toolset | Static hint toolset for network based on config env. | network | container-manager-mcp |
| container-manager-mcp_swarm_toolset | Static hint toolset for swarm based on config env. | swarm | container-manager-mcp |
| container-manager-mcp_container_toolset | Static hint toolset for container based on config env. | container | container-manager-mcp |
| container-manager-mcp_compose_toolset | Static hint toolset for compose based on config env. | compose | container-manager-mcp |
| container-manager-mcp_misc_toolset | Static hint toolset for misc based on config env. | misc | container-manager-mcp |
| container-manager-mcp_volume_toolset | Static hint toolset for volume based on config env. | volume | container-manager-mcp |
