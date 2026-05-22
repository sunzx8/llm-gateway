"""
代码工具集

包含代码评测和代码提交相关的工具类。
"""

from typing import Any

from .base_tool import BaseTool


class EvalCodeTool(BaseTool):
    name = "eval_code"
    description = (
        "在隔离的记忆库副本上试运行代码，验证代码的正确性。"
        "会复制当前用户的完整记忆库到临时环境中执行，不会影响用户的真实数据。\n\n"
        "适用场景：\n"
        "- 验证摄入代码是否能正确写入记忆\n"
        "- 验证消费代码是否能正确检索记忆\n"
        "- 调试代码逻辑问题\n\n"
        "注意：每次调用会有一定开销（复制记忆库），请在代码基本完成后再使用。\n"
        "超时限制：30秒。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "要评测的完整 Python 代码文件内容",
            },
            "code_type": {
                "type": "string",
                "enum": ["ingest", "retrieve"],
                "description": "代码类型：ingest（摄入代码）或 retrieve（消费代码）",
            },
            "test_input": {
                "type": "object",
                "description": (
                    "测试输入参数。\n"
                    "ingest 类型需要: {\"messages\": [{\"role\":\"user\",\"content\":\"...\"},...], \"user_id\": \"...\", \"session_id\": \"...\"}\n"
                    "retrieve 类型需要: {\"query\": \"...\", \"messages\": [...], \"user_id\": \"...\", \"session_id\": \"...\"}"
                ),
                "properties": {
                    "messages": {
                        "type": "array",
                        "items": {"type": "object"},
                        "description": "测试消息列表",
                    },
                    "query": {
                        "type": "string",
                        "description": "测试查询（retrieve 类型必填）",
                    },
                    "user_id": {
                        "type": "string",
                        "description": "测试用户 ID（默认使用当前用户）",
                    },
                    "session_id": {
                        "type": "string",
                        "description": "测试会话 ID（默认使用当前会话）",
                    },
                },
            },
        },
        "required": ["code", "code_type", "test_input"],
    }
    is_readonly = True  # 在隔离环境执行，不影响真实数据
    category = "code"

    async def execute(self, args: dict[str, Any], **deps) -> str:
        """在隔离沙箱中执行代码评测。

        deps 中需要: fs, vec, graph, llm, user_id, session_id。
        """
        from context_task.codegen.eval_sandbox import EvalSandbox

        fs = deps.get("fs")
        vec = deps.get("vec")
        graph = deps.get("graph")
        llm = deps.get("llm")
        user_id = deps.get("user_id", "default_user")
        session_id = deps.get("session_id", "")

        if not all([fs, vec, graph, llm]):
            return "ERROR: eval_code 需要 fs/vec/graph/llm 依赖"

        code = args.get("code", "")
        code_type = args.get("code_type", "")
        test_input = args.get("test_input", {})

        if not code:
            return "ERROR: 代码内容为空"
        if code_type not in ("ingest", "retrieve"):
            return "ERROR: code_type 必须为 'ingest' 或 'retrieve'"

        # 从 test_input 中提取参数
        messages = test_input.get("messages", [])
        test_user_id = test_input.get("user_id", user_id)
        test_session_id = test_input.get("session_id", session_id)
        query = test_input.get("query", "")

        if not messages:
            return "ERROR: test_input.messages 不能为空"

        if code_type == "retrieve" and not query:
            return "ERROR: retrieve 类型必须提供 test_input.query"

        # 创建沙箱并执行
        sandbox = EvalSandbox(
            fs=fs,
            vec=vec,
            graph=graph,
            llm=llm,
            user_id=user_id,
        )

        try:
            await sandbox.setup()

            if code_type == "ingest":
                result = await sandbox.run_ingest(
                    code=code,
                    messages=messages,
                    user_id=test_user_id,
                    session_id=test_session_id,
                )
            else:
                result = await sandbox.run_retrieve(
                    code=code,
                    query=query,
                    messages=messages,
                    user_id=test_user_id,
                    session_id=test_session_id,
                )

            return result.to_tool_response()

        except Exception as e:
            return f"ERROR: 评测执行异常 — {e}"

        finally:
            await sandbox.teardown()


class SubmitIngestCodeTool(BaseTool):
    name = "submit_ingest_code"
    description = (
        "提交生成的摄入代码文件。代码会被自动验证（importlib 加载），"
        "如果验证失败会返回错误信息，你需要修复后重新提交。\n\n"
        "代码必须是完整的 .py 文件内容，包含 import 语句、类定义和方法实现。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "完整的 Python 代码文件内容（包含 import、类定义和方法实现）",
            },
        },
        "required": ["code"],
    }
    is_readonly = False
    category = "code"

    async def execute(self, args: dict[str, Any], **deps) -> str:
        """submit_ingest_code 在 agent loop 中被直接拦截处理，不通过 dispatch_tool 执行。"""
        return "ERROR: submit_ingest_code 应在 agent loop 中直接处理，不应通过 dispatch_tool 调用"


class SubmitRetrieveCodeTool(BaseTool):
    name = "submit_retrieve_code"
    description = (
        "提交生成的消费代码文件。代码会被自动验证（importlib 加载），"
        "如果验证失败会返回错误信息，你需要修复后重新提交。\n\n"
        "代码必须是完整的 .py 文件内容，包含 import 语句、类定义和方法实现。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "完整的 Python 代码文件内容（包含 import、类定义和方法实现）",
            },
        },
        "required": ["code"],
    }
    is_readonly = False
    category = "code"

    async def execute(self, args: dict[str, Any], **deps) -> str:
        """submit_retrieve_code 在 agent loop 中被直接拦截处理，不通过 dispatch_tool 执行。"""
        return "ERROR: submit_retrieve_code 应在 agent loop 中直接处理，不应通过 dispatch_tool 调用"


class SubmitRetrieveCodeFuncTool(BaseTool):
    """提交 retrieve_memory 函数体的工具（函数级别提交）。"""
    name = "submit_retrieve_code_func"
    description = (
        "提交生成的消费代码。代码会被自动验证（语法检查），"
        "如果验证失败会返回错误信息，你需要修复后重新提交。\n\n"
        "代码应该是 retrieve_memory 函数的实现代码（函数体），"
        "会被嵌入到消费代码骨架中。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "retrieve_memory 函数的实现代码（包含函数定义和函数体）",
            },
        },
        "required": ["code"],
    }
    is_readonly = False
    category = "code"

    async def execute(self, args: dict[str, Any], **deps) -> str:
        """submit_retrieve_code_func 在 agent loop 中被直接拦截处理，不通过 dispatch_tool 执行。"""
        return "ERROR: submit_retrieve_code_func 应在 agent loop 中直接处理，不应通过 dispatch_tool 调用"


class SubmitStrategyTool(BaseTool):
    """提交策略文档的工具。

    用于策略生成阶段，模型通过此工具提交完整的策略文档内容，
    避免在 finish 工具的 result 参数中塞入过长的文本导致截断或空输出。
    """
    name = "submit_strategy"
    description = (
        "提交策略文档内容。请在策略设计完成后，通过此工具提交完整的策略文档。\n\n"
        "**重要**：\n"
        "- 必须提交完整的策略文档（Markdown 格式），不可为空\n"
        "- 提交后请调用 `finish` 工具结束任务\n"
        "- 如果内容过长，可以分多次调用此工具追加内容（设置 append=true）\n\n"
        "提交成功后会返回确认信息，然后你应该调用 `finish` 工具结束任务。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "策略文档内容（Markdown 格式），必须非空",
            },
            "append": {
                "type": "boolean",
                "description": "是否追加到已有内容后面（默认 false，即覆盖）",
            },
        },
        "required": ["content"],
    }
    is_readonly = False
    category = "code"

    async def execute(self, args: dict[str, Any], **deps) -> str:
        """submit_strategy 在 agent loop 中被直接拦截处理。"""
        return "ERROR: submit_strategy 应在 agent loop 中直接处理，不应通过 dispatch_tool 调用"


class SubmitBaseClassCodeTool(BaseTool):
    """提交增强基类代码的工具。"""
    name = "submit_base_class_code"
    description = (
        "提交生成的增强基类代码文件。代码会被自动验证（importlib 加载），"
        "如果验证失败会返回错误信息，你需要修复后重新提交。\n\n"
        "代码必须是完整的 .py 文件内容，包含 import 语句、类定义和所有原子方法实现。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "完整的 Python 代码文件内容（包含 import、类定义和所有原子方法实现）",
            },
        },
        "required": ["code"],
    }
    is_readonly = False
    category = "code"

    async def execute(self, args: dict[str, Any], **deps) -> str:
        """submit_base_class_code 在 agent loop 中被直接拦截处理。"""
        return "ERROR: submit_base_class_code 应在 agent loop 中直接处理，不应通过 dispatch_tool 调用"


class SubmitSubclassCodeTool(BaseTool):
    """提交子类代码的工具（含版本号参数）。"""
    name = "submit_subclass_code"
    description = (
        "提交生成的子类代码文件。代码会被自动验证（importlib 加载），"
        "如果验证失败会返回错误信息，你需要修复后重新提交。\n\n"
        "代码必须是完整的 .py 文件内容，包含 import 语句、类定义和主方法实现。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "完整的 Python 代码文件内容（包含 import、类定义和主方法实现）",
            },
            "version": {
                "type": "integer",
                "description": "子类版本号（从 1 开始），对应策略方案的编号",
            },
        },
        "required": ["code", "version"],
    }
    is_readonly = False
    category = "code"

    async def execute(self, args: dict[str, Any], **deps) -> str:
        """submit_subclass_code 在 agent loop 中被直接拦截处理。"""
        return "ERROR: submit_subclass_code 应在 agent loop 中直接处理，不应通过 dispatch_tool 调用"


class PythonSyntaxCheckTool(BaseTool):
    """Python 语法检查工具，基于 ast 模块实现静态语法校验。"""

    name = "python_syntax_check"
    description = (
        "对 Python 代码进行语法检查（基于 ast.parse），不执行代码。\n"
        "适用于在提交代码前快速验证语法是否正确，比 eval_code 更轻量（无需复制记忆库）。\n\n"
        "返回：\n"
        "- 语法正确时返回 '✅ 语法检查通过'\n"
        "- 语法错误时返回错误类型、行号、列号和详细信息"
    )
    parameters = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "要检查的完整 Python 代码内容",
            },
        },
        "required": ["code"],
    }
    is_readonly = True
    category = "code"

    async def execute(self, args: dict[str, Any], **deps) -> str:
        """使用 ast.parse 对代码进行语法检查。"""
        import ast

        code = args.get("code", "")
        if not code:
            return "ERROR: 代码内容为空"

        try:
            ast.parse(code, filename="<submitted_code>")
            return "✅ 语法检查通过（共 {} 行）".format(code.count("\n") + 1)
        except SyntaxError as e:
            parts = [f"❌ 语法错误: {e.msg}"]
            if e.lineno is not None:
                parts.append(f"  行号: {e.lineno}")
            if e.offset is not None:
                parts.append(f"  列号: {e.offset}")
            if e.text is not None:
                parts.append(f"  出错行: {e.text.rstrip()}")
                if e.offset is not None:
                    # 用箭头指示出错位置
                    parts.append(f"          {' ' * (e.offset - 1)}^")
            return "\n".join(parts)
