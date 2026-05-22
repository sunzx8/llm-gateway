"""llm_gateway.rl — 离线 RL 管线（rl_env + slime_train + data_gen）。

本包面向训练侧 / 数据侧，**不会被 gateway 主服务运行时导入**，因此对主服务零启动开销。

子包：
- rl_env: 围绕 llm_gateway.storage + llm_gateway.{atomic_code_t2,context_task} 的 RL sandbox。
- slime_train: Slime 训练定制脚本（reward / custom_generate / agentic rollout）。
- data_gen: 数据生成（占位；当前仍在老 workspace 运行，按需吸收）。
"""
