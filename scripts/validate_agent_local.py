import asyncio
import logging
import os
import sys

# Add parent directory to path to allow importing the package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from container_manager_mcp.container_manager_agent import create_agent, chat

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

async def main():
    print("Initializing Container Manager Agent...")
    try:
        agent = create_agent(
            provider="openai",
            model_id="qwen/qwen3-4b-2507",
            base_url="http://host.docker.internal:1234/v1",
            api_key="ollama",
            mcp_url="http://localhost:8005/mcp"
        )
        print("Agent initialized successfully.")
    except Exception as e:
        print(f"Agent initialization failed: {e}")
        return

    print("\n--- Starting Sample Chat Validation ---\n")

    questions = [
        "List all running containers.",
        "Prune unused images.",
    ]

    for q in questions:
        print(f"\n\nUser: {q}\n")
        try:
            # We use chat() which prints the result
            await chat(agent, q)
        except Exception as e:
            print(f"\nError processing question '{q}': {e}")

if __name__ == "__main__":
    asyncio.run(main())
