# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""HTTP-level simulation using respx for SDK-agnostic LLM testing.

Provides HTTP interception for LLM API calls. The SDK's own JSON parsing
runs on simulated responses, producing real SDK types (`Message`, `TextBlock`, etc.).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any

import httpx
import respx

from tenro._construct.http.builders.factory import ProviderSchemaFactory
from tenro._core.context import get_current_agent_name
from tenro.providers import Provider

if TYPE_CHECKING:
    from tenro._construct.simulate.orchestrator import SimulatedResponse

# Provider API endpoints
PROVIDER_ENDPOINTS: dict[str, str] = {
    "anthropic": "https://api.anthropic.com/v1/messages",
    "openai": "https://api.openai.com/v1/chat/completions",
    "gemini": "https://generativelanguage.googleapis.com/",  # Prefix for pattern match
}


def get_provider_endpoints() -> dict[str, str]:
    """Get provider endpoints, including dynamically registered providers.

    Returns:
        Dict mapping provider names to their endpoint URLs.
    """
    from tenro._construct.http.registry import ProviderRegistry

    endpoints = dict(PROVIDER_ENDPOINTS)
    for name in ProviderRegistry.list_providers():
        if name not in endpoints:
            try:
                config = ProviderRegistry.get_provider(name)
                endpoint = ProviderRegistry.get_endpoint(name)
                endpoints[name] = f"{config.base_url}{endpoint.http_path}"
            except Exception:
                pass
    return endpoints


def get_supported_providers() -> set[str]:
    """Get set of providers that support HTTP interception.

    Returns:
        Set of provider names, including dynamically registered providers.
    """
    return set(get_provider_endpoints().keys())


# Default LLM domains to block when no simulation matches
DEFAULT_BLOCKED_LLM_DOMAINS: list[str] = [
    "api.openai.com",
    "api.anthropic.com",
    "generativelanguage.googleapis.com",
]

# Type for call capture callback:
#   (provider, messages, model, response_text, agent, tool_calls) -> None
CallCaptureCallback = Callable[
    [str, list[dict[str, Any]], str | None, str, str | None, list[dict[str, Any]]], None
]


class HttpInterceptor:
    """Intercepts HTTP requests to LLM APIs using respx.

    When activated, HTTP requests to provider APIs are intercepted and
    return simulated responses. The SDK's JSON parsing still runs, so you
    get real SDK types (`Message`, `TextBlock`, `ChatCompletion`, etc.).

    Examples:
        >>> def test_llm(construct: Construct):
        ...     construct.simulate_llm(Provider.ANTHROPIC, response="Hello!")
        ...     client = anthropic.Anthropic(api_key="test")
        ...     msg = client.messages.create(...)
        ...     isinstance(msg.content[0], TextBlock)  # True!
    """

    def __init__(
        self,
        on_call: CallCaptureCallback | None = None,
        blocked_llm_domains: list[str] | None = None,
    ) -> None:
        """Initialize the HTTP interceptor.

        Args:
            on_call: Optional callback invoked for each intercepted request.
                Receives (`provider`, `messages`, `model`, `response_text`, `agent`,
                `tool_calls`). The `agent` parameter is the name of the
                @link_agent-decorated agent that made this call, or None if called
                outside of an agent context. `tool_calls` is a list of tool call
                dicts the LLM emitted.
            blocked_llm_domains: Domains to block when no simulation matches.
                If None, uses DEFAULT_BLOCKED_LLM_DOMAINS. Pass empty list to
                disable blocking.
        """
        self._blocked_domains = (
            blocked_llm_domains if blocked_llm_domains is not None else DEFAULT_BLOCKED_LLM_DOMAINS
        )
        # Always use assert_all_mocked=False to allow non-blocked requests to pass through.
        # The unified handler explicitly raises UnexpectedLLMCallError for blocked LLM domains
        # that don't have simulations, providing the required blocking behavior.
        self._respx_router = respx.MockRouter(
            assert_all_called=False,
            assert_all_mocked=False,
        )
        # Response queue: provider -> iterator of (json_response, response_text, tool_calls)
        self._response_queue: dict[
            str, Iterator[tuple[dict[str, Any] | Exception, str | None, list[dict[str, Any]]]]
        ] = {}
        self._on_call = on_call

    def simulate_provider(
        self,
        provider: str,
        items: list[SimulatedResponse],
        adapter: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Simulate a provider's API endpoint with the given response items.

        Args:
            provider: Provider name for endpoint routing and tracking (e.g., "mistral").
            items: List of SimulatedResponse items, each with content and optional
                tool_calls.
            adapter: Response format adapter (e.g., "openai"). Defaults to provider.
            **kwargs: Additional response metadata (model, token_usage, etc.).

        Raises:
            ValueError: If provider endpoint is not configured.
        """
        endpoints = get_provider_endpoints()
        endpoint = endpoints.get(provider)
        if not endpoint:
            raise ValueError(
                f"Unknown provider '{provider}'. Available: {', '.join(endpoints.keys())}"
            )

        format_provider = adapter or provider

        if provider in self._response_queue:
            raise ValueError(
                f"Provider '{provider}' already simulated. "
                "Use a single simulate_llm() call for all responses"
            )

        response_tuples: list[
            tuple[dict[str, Any] | Exception, str | None, list[dict[str, Any]]]
        ] = []
        for item in items:
            tool_call_dicts: list[dict[str, Any]] = []
            if item.tool_calls:
                for tc in item.tool_calls:
                    if hasattr(tc, "name"):
                        # SimToolCall object - use call_id attribute
                        tool_call_dicts.append(
                            {
                                "id": getattr(tc, "call_id", None),
                                "name": tc.name,
                                "arguments": getattr(tc, "arguments", {}),
                            }
                        )
                    elif isinstance(tc, dict):
                        tool_call_dicts.append(tc)

            if isinstance(item.content, Exception):
                response_tuples.append((item.content, None, tool_call_dicts))
            else:
                resp_kwargs = dict(kwargs)
                if item.tool_calls:
                    resp_kwargs["tool_calls"] = ProviderSchemaFactory.create_tool_calls(
                        format_provider, item.tool_calls
                    )
                json_response = self._build_response_json(
                    format_provider, item.content, **resp_kwargs
                )
                response_tuples.append((json_response, item.content, tool_call_dicts))

        self._response_queue[provider] = iter(response_tuples)

        domain = self._provider_to_domain(provider)
        if domain and domain not in self._blocked_domains:
            self._respx_router.route(host=domain).mock(
                side_effect=lambda req, d=domain: self._unified_handler(req, d)
            )

    def _build_response_json(
        self,
        provider: str,
        content: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build provider-specific response JSON.

        Uses existing ProviderSchemaFactory to get the correct JSON structure,
        then converts ProviderResponse to plain dict for HTTP response.
        """
        response = ProviderSchemaFactory.create_response(provider, content, **kwargs)
        return dict(response)

    def _handle_request(self, request: httpx.Request, provider: str) -> httpx.Response:
        """Handle intercepted request: parse, callback, return response or raise."""
        try:
            response_item, response_text, tool_calls = next(self._response_queue[provider])
        except StopIteration:
            return httpx.Response(
                500,
                json={
                    "error": {
                        "type": "test_error",
                        "message": (
                            f"No more simulated responses for provider '{provider}'. "
                            "Add more responses to simulate_llm() or check your test "
                            "makes the expected number of LLM calls."
                        ),
                    }
                },
            )

        # Parse request body for callback (must run before exception to mark triggered)
        if self._on_call is not None:
            try:
                messages, model = self._parse_request(request, provider)
                agent_name = get_current_agent_name()
                # For exceptions, response_text is None but callback is still notified
                self._on_call(
                    provider, messages, model, response_text or "", agent_name, tool_calls
                )
            except Exception:
                # Callback failure must not prevent HTTP response or exception
                pass

        if isinstance(response_item, Exception):
            raise response_item

        return httpx.Response(200, json=response_item)

    def _parse_request(
        self, request: httpx.Request, provider: str
    ) -> tuple[list[dict[str, Any]], str | None]:
        """Parse request body to extract messages and model."""
        import json
        import re

        try:
            body = json.loads(request.content)
        except (json.JSONDecodeError, TypeError):
            return [], None

        # Ensure body is a dict (JSON could be list, str, etc.)
        if not isinstance(body, dict):
            return [], None

        # Anthropic/OpenAI use "messages", Gemini uses "contents"
        messages = body.get("messages") or body.get("contents", [])
        model = body.get("model")

        # Gemini: model is in URL path, not body (e.g., /v1beta/models/gemini-pro:generateContent)
        if model is None and provider == Provider.GEMINI:
            match = re.search(r"/models/([^/:]+)", request.url.path)
            if match:
                model = match.group(1)

        return messages, model

    def _add_unified_handler_routes(self) -> None:
        """Add unified handler routes for all blocked LLM domains.

        Each route uses a handler that checks if a simulation exists for
        the request. If yes, provides simulated response. If no, blocks.
        """
        for domain in self._blocked_domains:
            self._respx_router.route(host=domain).mock(
                side_effect=lambda req, d=domain: self._unified_handler(req, d)
            )

    def _unified_handler(self, request: httpx.Request, domain: str) -> httpx.Response:
        """Handle LLM domain request: simulate if possible, otherwise block.

        Args:
            request: The intercepted HTTP request.
            domain: The LLM domain being accessed.

        Returns:
            Simulated response if simulation exists.

        Raises:
            UnexpectedLLMCallError: If no simulation matches this request.

        Note:
            Routes match by domain, not path. A simulation for `openai` intercepts
            all requests to `api.openai.com`, including non-chat endpoints like
            embeddings. This is a known limitation.
        """
        provider = self._domain_to_provider(domain)
        if provider and provider in self._response_queue:
            return self._handle_request(request, provider)

        from tenro.errors import TenroUnexpectedLLMCallError

        raise TenroUnexpectedLLMCallError(domain, str(request.url))

    def _domain_to_provider(self, domain: str) -> str | None:
        """Map domain to provider name by extracting from provider endpoints."""
        for provider, endpoint in get_provider_endpoints().items():
            if domain in endpoint:
                return provider
        return None

    def _provider_to_domain(self, provider: str) -> str | None:
        """Map provider name to domain by extracting from provider endpoints."""
        endpoints = get_provider_endpoints()
        endpoint = endpoints.get(provider)
        if not endpoint:
            return None
        from urllib.parse import urlparse

        return urlparse(endpoint).netloc

    def start(self) -> None:
        """Start intercepting HTTP requests.

        Block routes are added immediately to catch unmatched LLM requests.
        Simulation routes added later via simulate_provider() use more specific
        URL patterns that take precedence over host-based block routes.
        """
        self._add_unified_handler_routes()
        self._respx_router.start()

    def stop(self) -> None:
        """Stop intercepting HTTP requests and clean up routes."""
        self._respx_router.stop()
        self._respx_router.clear()
        self._respx_router.reset()  # Resets call counts and internal state
        self._response_queue.clear()

    def __enter__(self) -> HttpInterceptor:
        """Context manager entry - start intercepting."""
        self.start()
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _exc_tb: Any,
    ) -> None:
        """Context manager exit - stop intercepting."""
        self.stop()
