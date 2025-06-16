# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import subprocess
from pathlib import Path
from typing import AsyncIterator

from components.worker import SGLangWorker
from components.kv_router import Router
from pydantic import BaseModel
from utils.protocol import Tokens, PreprocessedRequest, StopConditions, SamplingOptions
from transformers import AutoTokenizer
import json

import dynamo.sdk as sdk
from dynamo.sdk import depends, service, async_on_start, dynamo_context, endpoint, on_shutdown
from dynamo.sdk.lib.config import ServiceConfig
from dynamo.sdk.lib.image import DYNAMO_IMAGE

logger = logging.getLogger(__name__)

# TODO: temp workaround to avoid port conflict with subprocess HTTP server; remove this once ingress is fixed
os.environ["DYNAMO_PORT"] = "3999"

def get_dynamo_run_binary():
    """Find the dynamo-run binary path in SDK or fallback to 'dynamo-run' command."""
    sdk_path = Path(sdk.__file__)
    binary_path = sdk_path.parent / "cli/bin/dynamo-run"
    if not binary_path.exists():
        return "dynamo-run"
    else:
        return str(binary_path)


def get_http_binary_path():
    """Find the HTTP binary path in SDK or fallback to 'http' command."""
    sdk_path = Path(sdk.__file__)
    binary_path = sdk_path.parent / "cli/bin/http"
    if not binary_path.exists():
        return "http"
    else:
        return str(binary_path)


class FrontendConfig(BaseModel):
    """Configuration for the Frontend service including model and HTTP server settings."""

    served_model_name: str
    endpoint: str
    port: int = 8080


@service(
    dynamo={
        "namespace": "dynamo",
    },
    resources={"cpu": "10", "memory": "20Gi"},
    workers=1,
    image=DYNAMO_IMAGE,
)
class Frontend:
    router = depends(Router)
    worker = depends(SGLangWorker)

    def __init__(self):
        """Initialize Frontend service with HTTP server and model configuration."""
        frontend_config = FrontendConfig(**ServiceConfig.get_parsed_config("Frontend"))
        logger.info("=== Frontend config: %s", frontend_config)
        self.frontend_config = frontend_config
        self.process = None
        self.router_client = None
        self.worker_client = None
        
        self.start_ingress_and_processor()
        
    def start_ingress_and_processor(self):
        """Starting dynamo-run based ingress and processor"""
        logger.info(
            f"Starting HTTP server and processor on port {self.frontend_config.port}"
        )
        dynamo_run_binary = get_dynamo_run_binary()
        endpoint = f"dyn://{self.frontend_config.endpoint}"

        logger.info(f"=== frontend: start_ingress_and_processor: dynamo-run with endpoint: {endpoint}")

        self.process = subprocess.Popen(
            [
                dynamo_run_binary,
                "in=http",
                f"out={endpoint}",
                "--http-port",
                str(self.frontend_config.port),
            ],
            stdout=None,
            stderr=None,
        )

    @endpoint(name="chat/completions")
    async def chat_completions(self, request_dict: PreprocessedRequest) -> AsyncIterator[str]:
        """
        Handle chat completions requests (routed from HTTP via llmctl).
        This combines the logic of vLLM's Processor since SGLang has no separate Processor.
        """
        logger.info(f"=== frontend: received parsed request {request_dict}")
        worker_id = ""
        # example request dict {'max_tokens': 30, 'messages': [{'content': 'hello', 'role': 'user'}], 'model': 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B', 'stream': True}   
        
        # Request is already parsed as a dictionary
        # openai_request = request_dict
        # logger.info(f"=== Using parsed OpenAI request: {openai_request}")
        
        # # Extract messages and convert to text (simplified tokenization)
        # messages = openai_request.get("messages", [])
        # if not messages:
        #     yield json.dumps({"error": "No messages provided"})
        #     return
            
        # # Use chat template if available, otherwise simple format
        # try:
        #     if hasattr(self.tokenizer, 'apply_chat_template') and self.tokenizer.chat_template:
        #         # Use the model's chat template
        #         text_prompt = self.tokenizer.apply_chat_template(
        #             messages, 
        #             tokenize=False, 
        #             add_generation_prompt=True
        #         )
        #     else:
        #         # Fallback to simple format
        #         text_prompt = ""
        #         for msg in messages:
        #             role = msg.get("role", "")
        #             content = msg.get("content", "")
        #             text_prompt += f"{role}: {content}\n"
        #         text_prompt += "assistant: "  # Add generation prompt
        # except Exception as e:
        #     logger.warning(f"Failed to apply chat template: {e}, using simple format")
        #     # Fallback to simple format
        #     text_prompt = ""
        #     for msg in messages:
        #         role = msg.get("role", "")
        #         content = msg.get("content", "")
        #         text_prompt += f"{role}: {content}\n"
        #     text_prompt += "assistant: "
        
        # logger.info(f"=== Converted to text prompt: {text_prompt}")
        
        # # Real tokenization using the model's tokenizer
        # try:
        #     tokens = self.tokenizer.encode(text_prompt, add_special_tokens=True)
        #     logger.info(f"=== Tokenized to {len(tokens)} tokens: {tokens[:10]}...")  # Show first 10 tokens
        # except Exception as e:
        #     logger.error(f"Tokenization failed: {e}")
        #     yield json.dumps({"error": "Tokenization failed"})
        #     return
            
        # tokens_obj = Tokens(tokens=tokens)
        
        # # Call router to get best worker
        # logger.info("=== Calling router to select best worker")
        # router_generator = await self.router_client.generate(tokens_obj.model_dump_json())
        # decision = await router_generator.__anext__()
        # worker_id, prefix_hit_rate = decision.data()
        
        # logger.info(f"=== Router selected worker: {worker_id}, prefix_hit_rate: {prefix_hit_rate}")
        
        # # Create SGLang PreprocessedRequest format
        # stop_conditions = StopConditions(
        #     max_tokens=openai_request.get("max_tokens", 100),
        #     stop=openai_request.get("stop"),
        #     ignore_eos=openai_request.get("ignore_eos", False)
        # )
        
        # sampling_options = SamplingOptions(
        #     temperature=openai_request.get("temperature"),
        #     top_p=openai_request.get("top_p"),
        #     top_k=openai_request.get("top_k"),
        #     frequency_penalty=openai_request.get("frequency_penalty"),
        #     presence_penalty=openai_request.get("presence_penalty"),
        #     seed=openai_request.get("seed")
        # )
        
        # preprocessed_request = PreprocessedRequest(
        #     token_ids=tokens,
        #     stop_conditions=stop_conditions,
        #     sampling_options=sampling_options,
        #     eos_token_ids=[self.tokenizer.eos_token_id] if self.tokenizer.eos_token_id else []
        # )
        
        # logger.info("=== frontend: Sending request to sglang worker: %s", preprocessed_request)
        
        # Forward request to selected worker (now in correct format)
        if worker_id == "":
            # No specific worker selected, use default routing
            worker_generator = await self.worker_client.generate(request_dict)
        else:
            # Route to specific worker
            worker_generator = await self.worker_client.direct(request_dict, int(worker_id))
            
        # Stream response from worker
        async for response in worker_generator:
            yield response.data()

    @on_shutdown
    def cleanup(self):
        """Clean up resources before shutdown."""
        logger.info("=== Frontend cleanup")
        
        # Remove model routing
        # subprocess.run(
        #     [
        #         "llmctl",
        #         "http",
        #         "remove",
        #         "chat-models",
        #         self.frontend_config.served_model_name,
        #     ],
        #     check=False,
        # )
