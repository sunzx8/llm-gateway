"""检索方案模板包。

本目录下的所有文件会在初始化时被**整体复制**到用户记忆库的
`<fs.base_path>/.codegen/retrieve/` 目录下，作为该用户独立可演进的副本。

复制后的副本与项目代码完全解耦，LLM 演进时可以：
- 修改 atomic.py 里现有的原子操作
- 在 atomic.py 里添加新的原子操作
- 在 plans/ 下新增方案文件，自由组合原子操作

注意：`base_plan.py` 不在本目录，留在项目代码（`atomic_code_t2/retrieve/base_plan.py`）。
原因：BasePlan 是稳定的抽象基类约定（`__init__` + `async run`），不应被 LLM 演进修改。
所有方案文件通过 `from atomic_code_t2.retrieve.base_plan import BasePlan` 引用项目级抽象。
"""
