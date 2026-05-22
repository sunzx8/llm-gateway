"""项目内置的检索方案模板集合。

这些 .py 文件**作为模板**——retrieve_task 会在用户记忆库下
`<fs.base_path>/.codegen/retrieve/<plan_name>.py` 缺失时，把它们复制过去，
然后通过 `loader.load_plan_class()` 动态加载执行。

加一个新的内置方案 = 在本目录下新增一个 .py 文件即可。
"""
