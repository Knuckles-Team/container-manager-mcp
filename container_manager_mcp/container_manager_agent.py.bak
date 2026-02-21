#!/usr/bin/python
import sys

# coding: utf-8
import json
import os
import argparse
import logging
import uvicorn
import httpx
from typing import Optional, Any, List
from contextlib import asynccontextmanager

from pydantic_ai import Agent, ModelSettings, RunContext
from pydantic_ai.mcp import (
    load_mcp_servers,
    MCPServerStreamableHTTP,
    MCPServerSSE,
)
from pydantic_ai_skills import SkillsToolset
from container_manager_mcp.utils import (
    to_integer,
    to_boolean,
    to_float,
    to_list,
    to_dict,
    get_mcp_config_path,
    get_skills_path,
    create_model,
    tool_in_tag,
    prune_large_messages,
)

from fastapi import FastAPI, Request
from starlette.responses import Response, StreamingResponse
from pydantic import ValidationError
from pydantic_ai.ui import SSE_CONTENT_TYPE
from pydantic_ai.ui.ag_ui import AGUIAdapter

__version__ = "1.3.14"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logging.getLogger("pydantic_ai").setLevel(logging.INFO)
logging.getLogger("fastmcp").setLevel(logging.INFO)
logging.getLogger("httpx").setLevel(logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_HOST = os.getenv("HOST", "0.0.0.0")
DEFAULT_PORT = to_integer(string=os.getenv("PORT", "9000"))
DEFAULT_DEBUG = to_boolean(string=os.getenv("DEBUG", "False"))
DEFAULT_PROVIDER = os.getenv("PROVIDER", "openai")
DEFAULT_MODEL_ID = os.getenv("MODEL_ID", "qwen/qwen3-coder-next")
DEFAULT_LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://host.docker.internal:1234/v1")
DEFAULT_LLM_API_KEY = os.getenv("LLM_API_KEY", "ollama")
DEFAULT_MCP_URL = os.getenv("MCP_URL", None)
DEFAULT_MCP_CONFIG = os.getenv("MCP_CONFIG", get_mcp_config_path())
DEFAULT_CUSTOM_SKILLS_DIRECTORY = os.getenv("CUSTOM_SKILLS_DIRECTORY", None)
DEFAULT_ENABLE_WEB_UI = to_boolean(os.getenv("ENABLE_WEB_UI", "False"))
DEFAULT_SSL_VERIFY = to_boolean(os.getenv("SSL_VERIFY", "True"))

DEFAULT_MAX_TOKENS = to_integer(os.getenv("MAX_TOKENS", "16384"))
DEFAULT_TEMPERATURE = to_float(os.getenv("TEMPERATURE", "0.7"))
DEFAULT_TOP_P = to_float(os.getenv("TOP_P", "1.0"))
DEFAULT_TIMEOUT = to_float(os.getenv("TIMEOUT", "32400.0"))
DEFAULT_TOOL_TIMEOUT = to_float(os.getenv("TOOL_TIMEOUT", "32400.0"))
DEFAULT_PARALLEL_TOOL_CALLS = to_boolean(os.getenv("PARALLEL_TOOL_CALLS", "True"))
DEFAULT_SEED = to_integer(os.getenv("SEED", None))
DEFAULT_PRESENCE_PENALTY = to_float(os.getenv("PRESENCE_PENALTY", "0.0"))
DEFAULT_FREQUENCY_PENALTY = to_float(os.getenv("FREQUENCY_PENALTY", "0.0"))
DEFAULT_LOGIT_BIAS = to_dict(os.getenv("LOGIT_BIAS", None))
DEFAULT_STOP_SEQUENCES = to_list(os.getenv("STOP_SEQUENCES", None))
DEFAULT_EXTRA_HEADERS = to_dict(os.getenv("EXTRA_HEADERS", None))
DEFAULT_EXTRA_BODY = to_dict(os.getenv("EXTRA_BODY", None))

AGENT_NAME = "Container Manager"
AGENT_DESCRIPTION = (
    "A multi-agent system for managing container tasks via delegated specialists."
)


SUPERVISOR_SYSTEM_PROMPT = os.environ.get(
    "SUPERVISOR_SYSTEM_PROMPT",
    default=(
        "You are the Container Manager Supervisor Agent.\n"
        "You orchestrate a team of specialized sub-agents to manage containers, images, volumes, logs, compose projects, swarm clusters, and networks.\n"
        "Your responsibilities:\n"
        "1. Analyze the user's request.\n"
        "2. Delegate tasks to the appropriate sub-agent(s).\n"
        "   - Use 'assign_task_to_info_agent' for system version/info.\n"
        "   - Use 'assign_task_to_image_agent' for pulling, listing, or removing images.\n"
        "   - Use 'assign_task_to_container_agent' for running, stopping, listing, or exec-ing containers.\n"
        "   - Use 'assign_task_to_volume_agent' for managing volumes.\n"
        "   - Use 'assign_task_to_log_agent' for viewing container logs.\n"
        "   - Use 'assign_task_to_compose_agent' for docker-compose operations.\n"
        "   - Use 'assign_task_to_swarm_agent' for docker swarm operations.\n"
        "   - Use 'assign_task_to_system_agent' for general system management.\n"
        "   - Use 'assign_task_to_network_agent' for network management.\n"
        "3. Coordinate complex workflows (e.g., 'pull image X then run container Y').\n"
        "4. Synthesize the results from sub-agents into a final response.\n"
        "5. Always be warm, professional, and helpful."
        "Note: The final response should contain all the relevant information from the tool executions. Never leave out any relevant information or leave it to the user to find it. "
        "You are the final authority on the user's request and the final communicator to the user. Present information as logically and concisely as possible. "
        "Explore using organized output with headers, sections, lists, and tables to make the information easy to navigate. "
        "If there are gaps in the information, clearly state that information is missing. Do not make assumptions or invent placeholder information, only use the information which is available.\n\n"
        "**Routing Guidelines:**\n"
        "- If multiple hosts/servers are connected (orchestration mode), ALWAYS direct tasks to the appropriate host based on the user's request (e.g., 'Update container on Host A')."
    ),
)

INFO_AGENT_PROMPT = os.environ.get(
    "INFO_AGENT_PROMPT",
    default=(
        "You are the Info Agent.\n"
        "Your goal is to provide system information.\n"
        "You can:\n"
        "- Check info: `info` (or equivalent tool)\n"
        "Use this to verify connection and engine details."
    ),
)

IMAGE_AGENT_PROMPT = os.environ.get(
    "IMAGE_AGENT_PROMPT",
    default=(
        "You are the Image Agent.\n"
        "Your goal is to manage container images.\n"
        "You can:\n"
        "- Manage: `prune_images`\n"
        "- Others: `list_images`, `pull_image` (if available)\n"
        "Pruning removes unused images."
    ),
)

CONTAINER_AGENT_PROMPT = os.environ.get(
    "CONTAINER_AGENT_PROMPT",
    default=(
        "You are the Container Operations Agent.\n"
        "Your goal is to manage container lifecycles.\n"
        "You can:\n"
        "- CRUD: `run_container`, `stop_container`, `remove_container`\n"
        "- List: `list_containers`\n"
        "- Action: `exec_in_container`, `prune_containers`\n"
        "Handle container IDs carefully."
    ),
)

VOLUME_AGENT_PROMPT = os.environ.get(
    "VOLUME_AGENT_PROMPT",
    default=(
        "You are the Volume Agent.\n"
        "Your goal is to manage storage volumes.\n"
        "You can:\n"
        "- CRUD: `create_volume`, `remove_volume`\n"
        "- List: `list_volumes`\n"
        "- Maintain: `prune_volumes`"
    ),
)

LOG_AGENT_PROMPT = os.environ.get(
    "LOG_AGENT_PROMPT",
    default=(
        "You are the Log Agent.\n"
        "Your goal is to retrieve container logs.\n"
        "You can:\n"
        "- Read: `get_container_logs`\n"
        "Essential for debugging running containers."
    ),
)

COMPOSE_AGENT_PROMPT = os.environ.get(
    "COMPOSE_AGENT_PROMPT",
    default=(
        "You are the Compose Agent.\n"
        "Your goal is to manage Docker Compose projects.\n"
        "You can:\n"
        "- Lifecycle: `compose_up`, `compose_down`\n"
        "- Info: `compose_ps`, `compose_logs`\n"
        "Requires a valid compose file path."
    ),
)

SWARM_AGENT_PROMPT = os.environ.get(
    "SWARM_AGENT_PROMPT",
    default=(
        "You are the Swarm Agent.\n"
        "Your goal is to manage Docker Swarm clusters.\n"
        "You can:\n"
        "- Cluster: `init_swarm`, `leave_swarm`\n"
        "- Services: `create_service`, `remove_service`, `list_services`\n"
        "- Nodes: `list_nodes`"
    ),
)

SYSTEM_AGENT_PROMPT = os.environ.get(
    "SYSTEM_AGENT_PROMPT",
    default=(
        "You are the System Agent.\n"
        "Your goal is to manage general system cleanup.\n"
        "You can:\n"
        "- Cleanup: `prune_system`\n"
        "Removes stopped containers, networks, and images."
    ),
)

NETWORK_AGENT_PROMPT = os.environ.get(
    "NETWORK_AGENT_PROMPT",
    default=(
        "You are the Network Agent.\n"
        "Your goal is to manage container networks.\n"
        "You can:\n"
        "- CRUD: `create_network`, `remove_network`\n"
        "- List: `list_networks`\n"
        "- Maintain: `prune_networks`"
    ),
)


def create_agent(
    provider: str = DEFAULT_PROVIDER,
    model_id: str = DEFAULT_MODEL_ID,
    base_url: Optional[str] = DEFAULT_LLM_BASE_URL,
    api_key: Optional[str] = DEFAULT_LLM_API_KEY,
    mcp_url: str = DEFAULT_MCP_URL,
    mcp_config: str = DEFAULT_MCP_CONFIG,
    custom_skills_directory: Optional[str] = DEFAULT_CUSTOM_SKILLS_DIRECTORY,
    ssl_verify: bool = DEFAULT_SSL_VERIFY,
) -> Agent:
    """
    Creates the Supervisor Agent with sub-agents registered as tools.
    """
    logger.info("Initializing Multi-Agent System for Container Manager...")

    model = create_model(
        provider=provider,
        model_id=model_id,
        base_url=base_url,
        api_key=api_key,
        ssl_verify=ssl_verify,
        timeout=DEFAULT_TIMEOUT,
    )
    settings = ModelSettings(
        max_tokens=DEFAULT_MAX_TOKENS,
        temperature=DEFAULT_TEMPERATURE,
        top_p=DEFAULT_TOP_P,
        timeout=DEFAULT_TIMEOUT,
        parallel_tool_calls=DEFAULT_PARALLEL_TOOL_CALLS,
        seed=DEFAULT_SEED,
        presence_penalty=DEFAULT_PRESENCE_PENALTY,
        frequency_penalty=DEFAULT_FREQUENCY_PENALTY,
        logit_bias=DEFAULT_LOGIT_BIAS,
        stop_sequences=DEFAULT_STOP_SEQUENCES,
        extra_headers=DEFAULT_EXTRA_HEADERS,
        extra_body=DEFAULT_EXTRA_BODY,
    )

    agent_toolsets = []
    if mcp_url:
        if "sse" in mcp_url.lower():
            server = MCPServerSSE(
                mcp_url,
                http_client=httpx.AsyncClient(
                    verify=ssl_verify, timeout=DEFAULT_TIMEOUT
                ),
            )
        else:
            server = MCPServerStreamableHTTP(
                mcp_url,
                http_client=httpx.AsyncClient(
                    verify=ssl_verify, timeout=DEFAULT_TIMEOUT
                ),
            )
        agent_toolsets.append(server)
        logger.info(f"Connected to MCP Server: {mcp_url}")
    elif mcp_config:
        mcp_toolset = load_mcp_servers(mcp_config)
        for server in mcp_toolset:
            if hasattr(server, "http_client"):
                server.http_client = httpx.AsyncClient(
                    verify=ssl_verify, timeout=DEFAULT_TIMEOUT
                )
        agent_toolsets.extend(mcp_toolset)
        logger.info(f"Connected to MCP Config JSON: {mcp_toolset}")

    # Always load default skills

    skill_dirs = [get_skills_path()]

    if custom_skills_directory and os.path.exists(custom_skills_directory):

        skill_dirs.append(str(custom_skills_directory))

    agent_toolsets.append(SkillsToolset(directories=skill_dirs))

    agent_defs = {
        "container_manager_info": (INFO_AGENT_PROMPT, "Info_Agent"),
        "image_management": (IMAGE_AGENT_PROMPT, "Image_Agent"),
        "container_management": (CONTAINER_AGENT_PROMPT, "Container_Agent"),
        "volume_management": (VOLUME_AGENT_PROMPT, "Volume_Agent"),
        "log_management": (LOG_AGENT_PROMPT, "Log_Agent"),
        "compose_management": (COMPOSE_AGENT_PROMPT, "Compose_Agent"),
        "swarm_management": (SWARM_AGENT_PROMPT, "Swarm_Agent"),
        "system_management": (SYSTEM_AGENT_PROMPT, "System_Agent"),
        "network_management": (NETWORK_AGENT_PROMPT, "Network_Agent"),
    }

    # 1. Identify Universal Skills
    # Universal skills are those in the skills directory that do NOT start with the package prefix
    package_prefix = "container-manager-"
    skills_path = get_skills_path()
    universal_skill_dirs = []

    if os.path.exists(skills_path):
        for item in os.listdir(skills_path):
            item_path = os.path.join(skills_path, item)
            if os.path.isdir(item_path):
                if not item.startswith(package_prefix):
                    universal_skill_dirs.append(item_path)
                    logger.info(f"Identified universal skill: {item}")

    supervisor_skills = []
    child_agents = {}
    supervisor_skills_directories = [get_skills_path()]

    for tag, (system_prompt, agent_name) in agent_defs.items():
        tag_toolsets = []
        for ts in agent_toolsets:

            def filter_func(ctx, tool_def, t=tag):
                return tool_in_tag(tool_def, t)

            if hasattr(ts, "filtered"):
                filtered_ts = ts.filtered(filter_func)
                tag_toolsets.append(filtered_ts)
            else:
                pass

        # Load specific skills for this tag
        skill_dir_name = f"container-manager-{tag.replace('_', '-')}"

        child_skills_directories = []

        # Check custom skills directory
        if custom_skills_directory:
            skill_dir_path = os.path.join(custom_skills_directory, skill_dir_name)
            if os.path.exists(skill_dir_path):
                child_skills_directories.append(skill_dir_path)

        # Check default skills directory
        default_skill_path = os.path.join(get_skills_path(), skill_dir_name)
        if os.path.exists(default_skill_path):
            child_skills_directories.append(default_skill_path)

        # Append Universal Skills to ALL child agents
        if universal_skill_dirs:
            child_skills_directories.extend(universal_skill_dirs)

        if child_skills_directories:
            ts = SkillsToolset(directories=child_skills_directories)
            tag_toolsets.append(ts)
            logger.info(
                f"Loaded specialized skills for {tag} from {child_skills_directories}"
            )

        # Collect tool names for logging
        all_tool_names = []
        for ts in tag_toolsets:
            try:
                # Unwrap FilteredToolset
                current_ts = ts
                while hasattr(current_ts, "wrapped"):
                    current_ts = current_ts.wrapped

                # Check for .tools (e.g. SkillsToolset)
                if hasattr(current_ts, "tools") and isinstance(current_ts.tools, dict):
                    all_tool_names.extend(current_ts.tools.keys())
                # Check for ._tools (some implementations might use private attr)
                elif hasattr(current_ts, "_tools") and isinstance(
                    current_ts._tools, dict
                ):
                    all_tool_names.extend(current_ts._tools.keys())
                # Check for .load method (SkillsToolset)
                elif hasattr(current_ts, "load") and callable(current_ts.load):
                    try:
                        skills = current_ts.load()
                        for s in skills:
                            if hasattr(s, "name"):
                                all_tool_names.append(s.name)
                            else:
                                all_tool_names.append(str(s))
                    except Exception:
                        pass
                else:
                    # Fallback for MCP or others where tools are not available sync
                    all_tool_names.append(f"<{type(current_ts).__name__}>")
            except Exception as e:
                logger.info(f"Unable to retrieve toolset: {e}")
                pass

        tool_list_str = ", ".join(all_tool_names)
        logger.info(f"Available tools for {agent_name} ({tag}): {tool_list_str}")
        child_agent = Agent(
            model=model,
            system_prompt=system_prompt,
            name=agent_name,
            toolsets=tag_toolsets,
            tool_timeout=DEFAULT_TOOL_TIMEOUT,
            model_settings=settings,
        )
        child_agents[tag] = child_agent

    # Create Custom Agent if custom_skills_directory is provided
    if custom_skills_directory:
        custom_agent_tag = "custom_agent"
        custom_agent_name = "Custom_Agent"
        custom_agent_prompt = (
            "You are the Custom Agent.\n"
            "Your goal is to handle custom tasks or general tasks not covered by other specialists.\n"
            "You have access to valid custom skills and universal skills."
        )

        custom_agent_skills_dirs = list(universal_skill_dirs)
        custom_agent_skills_dirs.append(custom_skills_directory)

        custom_toolsets = []
        custom_toolsets.append(SkillsToolset(directories=custom_agent_skills_dirs))

        custom_agent = Agent(
            name=custom_agent_name,
            system_prompt=custom_agent_prompt,
            model=model,
            model_settings=settings,
            toolsets=custom_toolsets,
            tool_timeout=DEFAULT_TOOL_TIMEOUT,
        )
        child_agents[custom_agent_tag] = custom_agent
        logger.info(
            f"Initialized Custom Agent with skills from: {custom_agent_skills_dirs}"
        )

    if custom_skills_directory:
        supervisor_skills_directories.append(custom_skills_directory)
    supervisor_skills.append(SkillsToolset(directories=supervisor_skills_directories))
    logger.info(f"Loaded supervisor skills from: {supervisor_skills_directories}")

    supervisor_agent = Agent(
        model=model,
        system_prompt=SUPERVISOR_SYSTEM_PROMPT,
        model_settings=settings,
        name=AGENT_NAME,
        toolsets=supervisor_skills,
        deps_type=Any,
    )

    @supervisor_agent.tool
    async def assign_task_to_info_agent(ctx: RunContext[Any], task: str) -> str:
        """Assign a task related to version or system info to the Info Agent."""
        return (
            await child_agents["container_manager_info"].run(
                task, usage=ctx.usage, deps=ctx.deps
            )
        ).output

    @supervisor_agent.tool
    async def assign_task_to_image_agent(ctx: RunContext[Any], task: str) -> str:
        """Assign a task related to container images (list, pull, remove, prune) to the Image Agent."""
        return (
            await child_agents["image_management"].run(
                task, usage=ctx.usage, deps=ctx.deps
            )
        ).output

    @supervisor_agent.tool
    async def assign_task_to_container_agent(ctx: RunContext[Any], task: str) -> str:
        """Assign a task related to container operations (run, list, stop, rm, exec) to the Container Agent."""
        return (
            await child_agents["container_management"].run(
                task, usage=ctx.usage, deps=ctx.deps
            )
        ).output

    @supervisor_agent.tool
    async def assign_task_to_volume_agent(ctx: RunContext[Any], task: str) -> str:
        """Assign a task related to volumes (create, list, remove) to the Volume Agent."""
        return (
            await child_agents["volume_management"].run(
                task, usage=ctx.usage, deps=ctx.deps
            )
        ).output

    @supervisor_agent.tool
    async def assign_task_to_log_agent(ctx: RunContext[Any], task: str) -> str:
        """Assign a task related to retrieving/viewing logs to the Log Agent."""
        return (
            await child_agents["log_management"].run(
                task, usage=ctx.usage, deps=ctx.deps
            )
        ).output

    @supervisor_agent.tool
    async def assign_task_to_compose_agent(ctx: RunContext[Any], task: str) -> str:
        """Assign a task related to docker-compose (up, down, logs, ls) to the Compose Agent."""
        return (
            await child_agents["compose_management"].run(
                task, usage=ctx.usage, deps=ctx.deps
            )
        ).output

    @supervisor_agent.tool
    async def assign_task_to_swarm_agent(ctx: RunContext[Any], task: str) -> str:
        """Assign a task related to docker swarm (init, join, service, etc.) to the Swarm Agent."""
        return (
            await child_agents["swarm_management"].run(
                task, usage=ctx.usage, deps=ctx.deps
            )
        ).output

    @supervisor_agent.tool
    async def assign_task_to_system_agent(ctx: RunContext[Any], task: str) -> str:
        """Assign a task related to system management (prune, df, etc.) to the System Agent."""
        return (
            await child_agents["system_management"].run(
                task, usage=ctx.usage, deps=ctx.deps
            )
        ).output

    @supervisor_agent.tool
    async def assign_task_to_network_agent(ctx: RunContext[Any], task: str) -> str:
        """Assign a task related to networks (create, list, inspect, etc.) to the Network Agent."""
        return (
            await child_agents["network_management"].run(
                task, usage=ctx.usage, deps=ctx.deps
            )
        ).output

    if custom_skills_directory:

        @supervisor_agent.tool
        async def assign_task_to_custom_agent(ctx: RunContext[Any], task: str) -> str:
            """
            Assign a task to the Custom Agent. Use this for tasks that don't fit into other specialists' categories
            but might be handled by custom skills or general universal skills.
            """
            return (
                await child_agents["custom_agent"].run(
                    task, usage=ctx.usage, deps=ctx.deps
                )
            ).output

    return supervisor_agent


async def chat(agent: Agent, prompt: str):
    result = await agent.run(prompt)
    print(f"Response:\n\n{result.output}")


async def node_chat(agent: Agent, prompt: str) -> List:
    nodes = []
    async with agent.iter(prompt) as agent_run:
        async for node in agent_run:
            nodes.append(node)
            print(node)
    return nodes


async def stream_chat(agent: Agent, prompt: str) -> None:
    async with agent.run_stream(prompt) as result:
        async for text_chunk in result.stream_text(delta=True):
            print(text_chunk, end="", flush=True)
        print("\nDone!")


def create_agent_server(
    provider: str = DEFAULT_PROVIDER,
    model_id: str = DEFAULT_MODEL_ID,
    base_url: Optional[str] = DEFAULT_LLM_BASE_URL,
    api_key: Optional[str] = DEFAULT_LLM_API_KEY,
    mcp_url: str = DEFAULT_MCP_URL,
    mcp_config: str = DEFAULT_MCP_CONFIG,
    custom_skills_directory: Optional[str] = DEFAULT_CUSTOM_SKILLS_DIRECTORY,
    debug: Optional[bool] = DEFAULT_DEBUG,
    host: Optional[str] = DEFAULT_HOST,
    port: Optional[int] = DEFAULT_PORT,
    enable_web_ui: bool = DEFAULT_ENABLE_WEB_UI,
    ssl_verify: bool = DEFAULT_SSL_VERIFY,
):
    print(
        f"Starting {AGENT_NAME}:"
        f"\tprovider={provider}"
        f"\tmodel={model_id}"
        f"\tbase_url={base_url}"
        f"\tmcp={mcp_url} | {mcp_config}"
        f"\tssl_verify={ssl_verify}"
    )
    agent = create_agent(
        provider=provider,
        model_id=model_id,
        base_url=base_url,
        api_key=api_key,
        mcp_url=mcp_url,
        mcp_config=mcp_config,
        custom_skills_directory=custom_skills_directory,
        ssl_verify=ssl_verify,
    )

    # Skills are loaded per-agent based on tags
    a2a_app = agent.to_a2a(
        name=AGENT_NAME,
        description=AGENT_DESCRIPTION,
        version=__version__,
        skills=[],
        debug=debug,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if hasattr(a2a_app, "router") and hasattr(a2a_app.router, "lifespan_context"):
            async with a2a_app.router.lifespan_context(a2a_app):
                yield
        else:
            yield

    app = FastAPI(
        title=f"{AGENT_NAME} - A2A + AG-UI Server",
        description=AGENT_DESCRIPTION,
        debug=debug,
        lifespan=lifespan,
    )

    @app.get("/health")
    async def health_check():
        return {"status": "OK"}

    app.mount("/a2a", a2a_app)

    @app.post("/ag-ui")
    async def ag_ui_endpoint(request: Request) -> Response:
        accept = request.headers.get("accept", SSE_CONTENT_TYPE)
        try:
            run_input = AGUIAdapter.build_run_input(await request.body())
        except ValidationError as e:
            return Response(
                content=json.dumps(e.json()),
                media_type="application/json",
                status_code=422,
            )

        if hasattr(run_input, "messages"):
            run_input.messages = prune_large_messages(run_input.messages)

        adapter = AGUIAdapter(agent=agent, run_input=run_input, accept=accept)
        event_stream = adapter.run_stream()
        sse_stream = adapter.encode_stream(event_stream)

        return StreamingResponse(
            sse_stream,
            media_type=accept,
        )

    if enable_web_ui:
        web_ui = agent.to_web(instructions=SUPERVISOR_SYSTEM_PROMPT)
        app.mount("/", web_ui)
        logger.info(
            "Starting server on %s:%s (A2A at /a2a, AG-UI at /ag-ui, Web UI: %s)",
            host,
            port,
            "Enabled at /" if enable_web_ui else "Disabled",
        )

    uvicorn.run(
        app,
        host=host,
        port=port,
        timeout_keep_alive=1800,
        timeout_graceful_shutdown=60,
        log_level="debug" if debug else "info",
    )


def agent_server():
    print(f"container_manager_agent v{__version__}")
    parser = argparse.ArgumentParser(
        add_help=False, description=f"Run the {AGENT_NAME} A2A + AG-UI Server"
    )
    parser.add_argument(
        "--host", default=DEFAULT_HOST, help="Host to bind the server to"
    )
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help="Port to bind the server to"
    )
    parser.add_argument("--debug", type=bool, default=DEFAULT_DEBUG, help="Debug mode")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    parser.add_argument(
        "--provider",
        default=DEFAULT_PROVIDER,
        choices=["openai", "anthropic", "google", "huggingface"],
        help="LLM Provider",
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="LLM Model ID")
    parser.add_argument(
        "--base-url",
        default=DEFAULT_LLM_BASE_URL,
        help="LLM Base URL (for OpenAI compatible providers)",
    )
    parser.add_argument("--api-key", default=DEFAULT_LLM_API_KEY, help="LLM API Key")
    parser.add_argument("--mcp-url", default=DEFAULT_MCP_URL, help="MCP Server URL")
    parser.add_argument(
        "--mcp-config", default=DEFAULT_MCP_CONFIG, help="MCP Server Config"
    )
    parser.add_argument(
        "--web",
        action="store_true",
        default=DEFAULT_ENABLE_WEB_UI,
        help="Enable Pydantic AI Web UI",
    )

    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Disable SSL verification for LLM requests (Use with caution)",
    )
    parser.add_argument("--help", action="store_true", help="Show usage")

    args = parser.parse_args()

    if hasattr(args, "help") and args.help:

        parser.print_help()

        sys.exit(0)

    if args.debug:
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler()],
            force=True,
        )
        logging.getLogger("pydantic_ai").setLevel(logging.DEBUG)
        logging.getLogger("fastmcp").setLevel(logging.DEBUG)
        logging.getLogger("httpcore").setLevel(logging.DEBUG)
        logging.getLogger("httpx").setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")

    create_agent_server(
        provider=args.provider,
        model_id=args.model_id,
        base_url=args.base_url,
        api_key=args.api_key,
        mcp_url=args.mcp_url,
        mcp_config=args.mcp_config,
        custom_skills_directory=args.custom_skills_directory,
        debug=args.debug,
        host=args.host,
        port=args.port,
        enable_web_ui=args.web,
        ssl_verify=not args.insecure,
    )


if __name__ == "__main__":
    agent_server()
