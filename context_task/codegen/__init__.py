"""codegen — 记忆摄入/消费代码生成的抽象基类、动态加载器和评测沙箱。

本模块提供：
- BaseMemoryIngestor: 记忆摄入抽象基类
- BaseMemoryConsumer: 记忆消费抽象基类
- load_ingestor_class: 动态加载摄入实现类
- load_consumer_class: 动态加载消费实现类
- EvalSandbox: 代码评测沙箱（隔离记忆库副本执行代码）
- EVAL_CODE_TOOL: eval_code 工具定义（向后兼容）
"""

from context_task.codegen.base_memory_ingestor import BaseMemoryIngestor
from context_task.codegen.base_memory_consumer import BaseMemoryConsumer
from context_task.codegen.loader import load_ingestor_class, load_consumer_class
from context_task.codegen.eval_sandbox import EvalSandbox, EvalResult
from context_task.tool import EVAL_CODE_TOOL

__all__ = [
    "BaseMemoryIngestor",
    "BaseMemoryConsumer",
    "load_ingestor_class",
    "load_consumer_class",
    "EvalSandbox",
    "EvalResult",
    "EVAL_CODE_TOOL",
]
