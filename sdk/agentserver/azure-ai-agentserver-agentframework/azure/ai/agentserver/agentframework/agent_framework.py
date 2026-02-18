# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
# pylint: disable=logging-fstring-interpolation
from __future__ import annotations

import os
from typing import Any, AsyncGenerator, Union

from agent_framework import SupportsAgentRun
from opentelemetry import trace

from azure.ai.agentserver.core import AgentRunContext, FoundryCBAgent
from azure.ai.agentserver.core.constants import Constants as AdapterConstants
from azure.ai.agentserver.core.logger import get_logger
from azure.ai.agentserver.core.models import (
    CreateResponse,
    Response as OpenAIResponse,
    ResponseStreamEvent,
)

from .models.agent_framework_input_converters import transform_input
from .models.agent_framework_output_non_streaming_converter import (
    AgentFrameworkOutputNonStreamingConverter,
)
from .models.agent_framework_output_streaming_converter import AgentFrameworkOutputStreamingConverter
from .models import constants

logger = get_logger()


class AgentFrameworkCBAgent(FoundryCBAgent):
    """
    Adapter class for integrating Agent Framework agents with the FoundryCB agent interface.

    This class wraps an Agent Framework `SupportsAgentRun` instance and provides a unified interface
    for running agents in both streaming and non-streaming modes. It handles input and output
    conversion between the Agent Framework and the expected formats for FoundryCB agents.

    Parameters:
        agent (SupportsAgentRun): An instance of an Agent Framework agent to be adapted.

    Usage:
        - Instantiate with an Agent Framework agent.
        - Call `agent_run` with a `CreateResponse` request body to execute the agent.
        - Supports both streaming and non-streaming responses based on the `stream` flag.
    """

    def __init__(self, agent: SupportsAgentRun):
        super().__init__()
        self.agent = agent
        logger.info(f"Initialized AgentFrameworkCBAgent with agent: {type(agent).__name__}")

    def _resolve_stream_timeout(self, request_body: CreateResponse) -> float:
        """Resolve idle timeout for streaming updates.

        Order of precedence:
        1) request_body.stream_timeout_s (if provided)
        2) env var constants.AGENTS_ADAPTER_STREAM_TIMEOUT_S
        3) constants.DEFAULT_STREAM_TIMEOUT_S

        :param request_body: The CreateResponse request body.
        :type request_body: CreateResponse

        :return: The resolved stream timeout in seconds.
        :rtype: float
        """
        override = request_body.get("stream_timeout_s", None)
        if override is not None:
            return float(override)
        env_val = os.getenv(constants.AGENTS_ADAPTER_STREAM_TIMEOUT_S)
        return float(env_val) if env_val is not None else float(constants.DEFAULT_STREAM_TIMEOUT_S)

    def init_tracing(self):
        exporter = os.environ.get(AdapterConstants.OTEL_EXPORTER_ENDPOINT)
        app_insights_conn_str = os.environ.get(AdapterConstants.APPLICATION_INSIGHTS_CONNECTION_STRING)

        if app_insights_conn_str:
            from azure.monitor.opentelemetry import configure_azure_monitor
            from agent_framework.observability import create_resource, enable_instrumentation

            configure_azure_monitor(
                connection_string=app_insights_conn_str,
                resource=create_resource(),
            )
            enable_instrumentation(enable_sensitive_data=True)
        elif exporter:
            from agent_framework.observability import configure_otel_providers

            os.environ.setdefault("OTEL_EXPORTER_OTLP_ENDPOINT", exporter)
            configure_otel_providers(enable_sensitive_data=True)

        self.tracer = trace.get_tracer(__name__)

    async def agent_run(
        self, context: AgentRunContext
    ) -> Union[
        OpenAIResponse,
        AsyncGenerator[ResponseStreamEvent, Any],
    ]:
        logger.info(f"Starting agent_run with stream={context.stream}")
        message = transform_input(context.request.get("input"))
        logger.debug(f"Transformed input message type: {type(message)}")

        # Use split converters
        if context.stream:
            logger.info("Running agent in streaming mode")
            streaming_converter = AgentFrameworkOutputStreamingConverter(context)

            async def stream_updates():
                update_count = 0
                for ev in streaming_converter.initial_events():
                    yield ev

                # agent.run(stream=True) returns a ResponseStream (async iterable)
                response_stream = self.agent.run(message, stream=True)
                async for update in response_stream:
                    update_count += 1
                    transformed = streaming_converter.transform_output_for_streaming(update)
                    for event in transformed:
                        yield event
                for ev in streaming_converter.completion_events():
                    yield ev
                logger.info("Streaming completed with %d updates", update_count)

            return stream_updates()

        # Non-streaming path
        logger.info("Running agent in non-streaming mode")
        non_streaming_converter = AgentFrameworkOutputNonStreamingConverter(context)
        result = await self.agent.run(message)
        logger.debug(f"Agent run completed, result type: {type(result)}")
        transformed_result = non_streaming_converter.transform_output_for_response(result)
        logger.info("Agent run and transformation completed successfully")
        return transformed_result
