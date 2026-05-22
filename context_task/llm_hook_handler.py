#-*-encoding:utf8-*-
from datetime import datetime

from interceptor.handler import ResponseContext,RequestContext,CallbackHandler
from logger.logger import logger
from storage.stores_factory import make_stores
from storage.session_store import session_manager
from utils.memory_llm_interface import LLMInterface, EmbeddingInterface
from . import get_retrieve_context_task
from .ingest_scheduler import memory_ingest, check_and_trigger_consolidate_by_retrieve
from .prompt.chinese_prompt import MAIN_AGENT_SYSTEM_PROMPT
from config.loader import get_config, AppConfig
from config.models import MemoryConfig, ContextTaskMode


class T2MemoryHandler(CallbackHandler):
    def __init__(self):
        """初始化需要的依赖类"""
        self.config:AppConfig = get_config()
        self.memory_config:MemoryConfig = self.config.memory_config

        # 复用 LLM 和 Embedding 客户端（线程安全，asyncio 安全）
        self._memory_llm = LLMInterface(self.memory_config.memory_llm.model_dump())
        self._embedder = EmbeddingInterface(self.memory_config.embedding.model_dump())

    async def on_request(self, context: RequestContext) -> RequestContext:
        # 记忆初始化的请求不去query，而是做记忆摄入（同步）
        if context.model == "memory-initialize":
            return await self._online_ingest_memory(context)

        session_store = await session_manager.get_or_create_session(context.user_id, context.session_id)

        if self.memory_config.context_task.compress_session:
            # todo: session压缩
            pass

        # 拼装维护的session context
        context.messages = session_store.get_messages_as_list() + context.messages
        session_latest_memory = session_store.latest_memory

        # 查询记忆，并拼装新的request
        fs_store, vec_store, graph_store = make_stores(self.memory_config.storage ,context.user_id, self._embedder)
        retrieve_task = get_retrieve_context_task(self._memory_llm, fs_store, vec_store, graph_store,
                                                  max_turns=self.memory_config.context_task.retrieve_context_task.max_turns)

        # 提取用户最新查询作为 query
        query = ""
        for msg in reversed(context.messages):
            if msg.get("role") == "user":
                query = msg.get("content", "")
                break

        result = await retrieve_task.execute(
            query=query,
            user_id=context.user_id or "default_user",
            session_id=context.session_id,
            messages=context.messages,
            latest_memory=session_latest_memory,
        )

        # 记录 retrieve 任务结果
        context.task_results.append({
            "task_type": "retrieve",
            "input_params": {
                "query": query,
                "user_id": context.user_id or "default_user",
                "session_id": context.session_id,
            },
            "result": result,
        })

        # retrieve 完成后，检查是否需要异步触发 consolidate
        check_and_trigger_consolidate_by_retrieve(
            user_id=context.user_id or "default_user",
            session_id=context.session_id,
            memory_config_dict=self.memory_config.model_dump(),
        )

        memory_ctx = result.get("query_memory", "")
        if memory_ctx:
            logger.info(f"memory_ctx: {memory_ctx}")
            # 更新session的最新记忆
            session_store.latest_memory = memory_ctx

        # 将最新的记忆插入到messages中
        if session_store.latest_memory:
            mode = self.memory_config.context_task.context_task_mode
            if mode == ContextTaskMode.Atomic_Code_T2:
                # Atomic_Code_T2：记忆和问题合并到单条 user message（对齐 T3）
                # session_store.latest_memory 已经是完整的 "# Memory Context\n...\n---\n..." 格式
                # eval 传来的 user_question 已经是 "# Question\n\n{question}\n\n**Important**:..." 格式
                # 这里只做拼接，不额外添加任何 header
                user_question = ""
                if context.messages and context.messages[-1].get("role") == "user":
                    user_question = context.messages[-1].get("content", "")
                    context.messages.pop()

                # 保留前面的对话历史（在线服务需要多轮上下文）
                # 只在末尾追加 system prompt + 合并后的 user message
                combined_user_msg = (
                    f"{session_store.latest_memory.strip()}\n\n"
                    f"{user_question}"
                )
                context.messages.append({"role": "system", "content": MAIN_AGENT_SYSTEM_PROMPT})
                context.messages.append({"role": "user", "content": combined_user_msg})
            else:
                # 其他模式：保持原有逻辑
                context.messages.insert(-1, {"role": "system", "content": session_store.latest_memory})
                context.messages.insert(-1, {"role": "system", "content": MAIN_AGENT_SYSTEM_PROMPT})

        return context
    
    async def on_response(self, context: ResponseContext) -> ResponseContext:
        user_id = context.user_id or "default_user"
        session_id = context.session_id
        session = await session_manager.get_or_create_session(user_id, session_id)

        if context.model == "fake-model":
            if session.messages:
                max_index = max(int(k) for k in session.messages)
                await session.clear_ingested_messages(max_index)
        
        if not context.model == "fake-model" and not self.memory_config.for_evaluation: # 非批量导入记忆请求
            # 更新 session messages
            # 计算从接收到question到获取到response的总耗时（毫秒）
            duration_ms = int(context.duration_seconds * 1000)
            received_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            await session.update_messages(context.messages, duration_ms=duration_ms, received_at=received_at)
        return context

    async def _online_ingest_memory(self, context: RequestContext):
        """使用memory-initialize模型，一次性摄入所有消息，减少评测耗时。

        处理逻辑：
        1. 从 messages 末尾提取连续的 user 角色消息（即用户问题）
        2. 将这些用户问题写入 session_store（供后续 retrieve 时作为上下文）
        3. 剩余的 messages 作为记忆做演进和摄入任务
        """
        messages = context.messages

        # 从末尾提取连续的 user 消息
        trailing_user_messages: list[dict] = []
        for msg in reversed(messages):
            if msg.get("role") == "user":
                trailing_user_messages.append(msg)
            else:
                break
        trailing_user_messages.reverse()  # 恢复原始顺序

        # 分离：剩余的 messages 用于记忆摄入，末尾连续 user 消息写入 session_store
        if trailing_user_messages:
            ingest_messages = messages[:-len(trailing_user_messages)]
        else:
            ingest_messages = messages

        # 将末尾连续的用户问题写入 session_store
        if trailing_user_messages:
            session_store = await session_manager.get_or_create_session(
                context.user_id, context.session_id
            )
            # 构造带序号的消息字典写入 session_store
            existing_count = len(session_store.messages)
            for i, msg in enumerate(trailing_user_messages):
                key = str(existing_count + i + 1)
                session_store.messages[key] = {
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", ""),
                }
            logger.info(
                f"memory-initialize: 将 {len(trailing_user_messages)} 条末尾用户问题写入 session_store, "
                f"session_id={context.session_id}",
            )

        # 剩余的 messages 做记忆摄入；memory_ingest 内部会在成功摄入后按阈值触发演进
        # message_offset 让 ingest agent loop 里的对话编号反映在完整对话中的真实位置
        raw_metadata = context.raw_data.get("metadata", {}) if context.raw_data else {}
        message_offset: int = int(raw_metadata.get("message_offset", 1))
        ingest_results = await memory_ingest(
            user_id=context.user_id,
            session_id=context.session_id,
            messages=ingest_messages,
            memory_config_dict=self.memory_config.model_dump(),
            memory_llm=self._memory_llm,
            embedder=self._embedder,
            message_offset=message_offset,
        )
        context.task_results.extend(ingest_results)
        logger.info(
            f"memory-initialize 一次性摄入完成, 总消息数: {len(messages)}, "
            f"摄入消息数: {len(ingest_messages)}, 写入session问题数: {len(trailing_user_messages)}",
        )

        return context
