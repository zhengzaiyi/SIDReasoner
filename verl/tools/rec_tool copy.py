import logging
import os
import asyncio  # 增加 lock 支持
from typing import Any, Optional
from uuid import uuid4
from openai import OpenAI
from verl.utils.reward_score import gsm8k
from verl.utils.rollout_trace import rollout_trace_op

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class RecTool(BaseTool):
    """A demo tool for calculating the reward of gsm8k.

    - `to_openai_function_tool_schema`: return the tool schema in OpenAI format.
    - `create`: create a tool instance for a trajectory.
    - `execute`: execute the tool.
    - `calc_reward`: calculate the reward respect to tool state.
    - `release`: release the tool instance.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """
        _tool_schema = OpenAIFunctionToolSchema.model_validate({
            "type": "function",
            "function": {
                "name": "chat_with_user",
                "description": "A tool for chatting with the user to clarify the user intention.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The query to chat with user.",
                        },
                    },
                    "required": ["query"],
                },
            }
        })
        """
        super().__init__(config, tool_schema)
        self._instance_dict = {}
        self._locks: dict[str, asyncio.Lock] = {}  # 安全锁
        self._counts: dict[str, int] = {}  # 调用计数
        self._max_calls = config.get("max_calls", 2)  # 最大调用次数

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, intention: Optional[str] = None, **kwargs) -> str:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "query": "",
            "intention": intention,
            "user_response": "",
        }
        self._locks[instance_id] = asyncio.Lock()  # 初始化锁
        self._counts[instance_id] = 0  # 初始化计数
        return instance_id

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[str, float, dict]:
        # 安全锁 + 调用次数限制
        lock = self._locks.get(instance_id)
        async with lock:
            self._counts[instance_id] += 1
            if self._counts[instance_id] > self._max_calls:
                raise ValueError(f"Instance {instance_id} exceed max_calls={self._max_calls}")

        query = parameters.get("query", "")
        if not isinstance(query, str):
            query = str(query)

        self._instance_dict[instance_id]["query"] = query
        user_response = await self.get_chat_response(instance_id)

        # a constant penalty for launching the chat. Change this later.
        tool_reward = -0.05
        self._instance_dict[instance_id]["user_response"] = user_response

        return f"{user_response}", tool_reward, {}

    async def get_chat_response(self, instance_id: str, **kwargs) -> float:
        query = self._instance_dict[instance_id]["query"]
        intention = self._instance_dict[instance_id]["intention"]

        # 构造用于 OpenAI 的 prompt
        full_prompt = f'''You are acting as a simulated user in an online shopping recommender system.

GOAL:
- You have a predefined intent profile (e.g., likes/dislikes certain genres).
- Your responses must align with this intent and remain consistent.
- When asked by the recommender (assistant), reply truthfully based on your intent.

RULES:
1. Only include a single sentence in each reply—no multi-sentence or paragraph responses.
2. Do NOT reveal your internal intent profile; just answer naturally and concisely.

INPUT VARIABLES:
- {intention}

QUERY:
- {query}
'''

        openai_api_key = os.getenv('OPENAI_API_KEY')
        client = OpenAI(api_key=openai_api_key)

        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": full_prompt}],
        )
        response_str = response.choices[0].message.content
        return response_str

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]
        # 清理状态
        self._locks.pop(instance_id, None)
        self._counts.pop(instance_id, None)

