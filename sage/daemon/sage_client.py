#!/usr/bin/env python3
"""
SAGE Daemon Client

Client library for raising sessions to interact with resident SAGE daemon.
Replaces direct model loading with API calls to the daemon.
"""

import httpx
from typing import Dict, Any, List, Optional
from datetime import datetime


class SAGEClient:
    """
    Client for interacting with SAGE resident daemon

    Usage:
        client = SAGEClient()
        response = await client.generate("Hello SAGE, how are you?")
        print(response['text'])
    """

    def __init__(self, base_url: str = "http://localhost:8765"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=300.0)  # 5min timeout for generation

    async def health_check(self) -> Dict[str, Any]:
        """Check if daemon is running and healthy"""
        try:
            response = await self.client.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "model_loaded": False
            }

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate text from SAGE

        Args:
            prompt: User prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            system_prompt: Optional system prompt (identity/role)

        Returns:
            {
                'text': str,              # Generated text
                'tokens_generated': int,   # Number of tokens
                'inference_time_ms': float,# Generation time
                'timestamp': str           # ISO timestamp
            }
        """
        request_data = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p
        }

        if system_prompt:
            request_data["system_prompt"] = system_prompt

        response = await self.client.post(
            f"{self.base_url}/generate",
            json=request_data
        )
        response.raise_for_status()
        return response.json()

    async def get_state(self) -> Dict[str, Any]:
        """Get current SAGE state"""
        response = await self.client.get(f"{self.base_url}/state")
        response.raise_for_status()
        return response.json()

    async def update_state(
        self,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        memory_request: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Update SAGE state"""
        request_data = {}

        if conversation_history is not None:
            request_data["conversation_history"] = conversation_history
        if memory_request is not None:
            request_data["memory_request"] = memory_request
        if metadata is not None:
            request_data["metadata"] = metadata

        response = await self.client.post(
            f"{self.base_url}/state",
            json=request_data
        )
        response.raise_for_status()
        return response.json()

    async def reset_conversation(self) -> Dict[str, Any]:
        """Reset conversation state (keeps identity)"""
        response = await self.client.post(f"{self.base_url}/reset")
        response.raise_for_status()
        return response.json()

    async def close(self):
        """Close client connection"""
        await self.client.aclose()


# Convenience functions for scripts
async def check_sage_daemon() -> bool:
    """Quick check if SAGE daemon is running"""
    client = SAGEClient()
    health = await client.health_check()
    await client.close()

    return health.get('status') == 'healthy' and health.get('model_loaded', False)


async def get_sage_response(
    prompt: str,
    system_prompt: Optional[str] = None,
    **kwargs
) -> str:
    """
    Convenience function for single response

    Usage:
        response = await get_sage_response("What do you think?")
    """
    client = SAGEClient()
    try:
        result = await client.generate(prompt, system_prompt=system_prompt, **kwargs)
        return result['text']
    finally:
        await client.close()


if __name__ == "__main__":
    import asyncio

    async def test():
        """Test client connectivity"""
        print("Testing SAGE client...")

        client = SAGEClient()

        # Health check
        health = await client.health_check()
        print(f"\nHealth: {health}")

        if health.get('model_loaded'):
            # Test generation
            print("\nTesting generation...")
            response = await client.generate(
                "Hello! What is your name?",
                max_tokens=50
            )
            print(f"SAGE: {response['text']}")
            print(f"Tokens: {response['tokens_generated']}")
            print(f"Time: {response['inference_time_ms']:.0f}ms")
        else:
            print("Model not loaded - daemon may not be running")

        await client.close()

    asyncio.run(test())
