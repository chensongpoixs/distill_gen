"""
OpenAI-compatible HTTP 客户端模块。

基于 httpx + asyncio 实现异步并发请求，
封装重试、超时、并发控制（Semaphore）。
"""

import asyncio
import logging
from typing import Optional

import httpx

from distill_gen.config import LlamaCppConfig

logger = logging.getLogger(__name__)

# fmt: off
TEMPERATURE_MAP = {
    "初级": "primary",
    "中级": "intermediate",
    "高级": "advanced",
}
# fmt: on


class LLMClient:
    """
    OpenAI-compatible HTTP 客户端，专为 llama.cpp 优化。

    封装:
    - 异步 HTTP 请求（httpx）
    - 指数退避重试
    - Semaphore 并发控制
    - 结构化 JSON 响应解析
    """

    def __init__(self, llama_config: LlamaCppConfig, max_workers: int = 3):
        self.base_url = llama_config.base_url.rstrip("/")
        self.api_key = llama_config.api_key
        self.model = llama_config.model
        self.timeout = llama_config.timeout
        self.max_retries = llama_config.max_retries
        self.retry_delay = llama_config.retry_delay
        self._semaphore = asyncio.Semaphore(max_workers)
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """延迟初始化 httpx AsyncClient（复用连接池）。"""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout, connect=30.0),
                limits=httpx.Limits(max_keepalive_connections=10),
            )
        return self._client

    async def close(self):
        """关闭 HTTP 客户端连接池。"""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def chat(
        self,
        messages: list[dict],
        temperature: float = 0.5,
        max_tokens: int = 4096,
        top_p: float = 0.9,
        seed: int = 42,
    ) -> str:
        """
        发送聊天请求，返回生成的文本。

        Args:
            messages: OpenAI 格式的消息列表 [{"role": ..., "content": ...}]
            temperature: 采样温度
            max_tokens: 最大生成 token 数
            top_p: nucleus sampling 参数
            seed: 随机种子

        Returns:
            str: LLM 生成的完整文本

        Raises:
            RuntimeError: 所有重试均失败时
        """
        last_error: Optional[Exception] = None

        for attempt in range(self.max_retries):
            try:
                async with self._semaphore:
                    client = await self._get_client()
                    headers = {}
                    if self.api_key:
                        headers["Authorization"] = f"Bearer {self.api_key}"
                    response = await client.post(
                        f"{self.base_url}/chat/completions",
                        json={
                            "model": self.model,
                            "messages": messages,
                            "temperature": temperature,
                            "max_tokens": max_tokens,
                            "top_p": top_p,
                            "seed": seed,
                        },
                        headers=headers,
                    )

                    if response.status_code != 200:
                        error_detail = response.text[:500]
                        raise httpx.HTTPStatusError(
                            f"HTTP {response.status_code}: {error_detail}",
                            request=response.request,
                            response=response,
                        )

                    data = response.json()
                    content = data["choices"][0]["message"]["content"]
                    return content.strip()

            except (httpx.TimeoutException, httpx.HTTPStatusError,
                    httpx.RemoteProtocolError, httpx.ConnectError) as e:
                last_error = e
                wait = self.retry_delay * (2 ** attempt)
                logger.warning(
                    f"LLM 请求失败 (尝试 {attempt + 1}/{self.max_retries}): {e}\n"
                    f"  等待 {wait:.1f}s 后重试..."
                )
                await asyncio.sleep(wait)

            except Exception as e:
                last_error = e
                logger.error(f"LLM 请求未知异常: {type(e).__name__}: {e}")
                wait = self.retry_delay * (2 ** attempt)
                await asyncio.sleep(wait)

        raise RuntimeError(
            f"LLM 请求失败，已重试 {self.max_retries} 次。最后错误: {last_error}"
        )

    def get_temperature(self, difficulty: str, temp_config) -> float:
        """
        根据难度等级获取对应温度。

        Args:
            difficulty: 难度标签（初级/中级/高级）
            temp_config: TemperatureConfig 对象

        Returns:
            float: 对应温度值
        """
        key = TEMPERATURE_MAP.get(difficulty, "intermediate")
        return getattr(temp_config, key, 0.5)
