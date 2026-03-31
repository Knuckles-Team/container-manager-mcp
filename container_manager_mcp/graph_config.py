"""Container Manager graph configuration — tag prompts and env var mappings.

This is the only file needed to enable graph mode for this agent.
Provides TAG_PROMPTS and TAG_ENV_VARS for create_graph_agent_server().
"""

                                                                       
TAG_PROMPTS: dict[str, str] = {
    "compose": (
        "You are a Container Manager Compose specialist. Help users manage and interact with Compose functionality using the available tools."
    ),
    "container": (
        "You are a Container Manager Container specialist. Help users manage and interact with Container functionality using the available tools."
    ),
    "image": (
        "You are a Container Manager Image specialist. Help users manage and interact with Image functionality using the available tools."
    ),
    "info": (
        "You are a Container Manager Info specialist. Help users manage and interact with Info functionality using the available tools."
    ),
    "network": (
        "You are a Container Manager Network specialist. Help users manage and interact with Network functionality using the available tools."
    ),
    "swarm": (
        "You are a Container Manager Swarm specialist. Help users manage and interact with Swarm functionality using the available tools."
    ),
    "system": (
        "You are a Container Manager System specialist. Help users manage and interact with System functionality using the available tools."
    ),
    "volume": (
        "You are a Container Manager Volume specialist. Help users manage and interact with Volume functionality using the available tools."
    ),
}


                                                                        
TAG_ENV_VARS: dict[str, str] = {
    "compose": "COMPOSETOOL",
    "container": "CONTAINERTOOL",
    "image": "IMAGETOOL",
    "info": "INFOTOOL",
    "network": "NETWORKTOOL",
    "swarm": "SWARMTOOL",
    "system": "SYSTEMTOOL",
    "volume": "VOLUMETOOL",
}
