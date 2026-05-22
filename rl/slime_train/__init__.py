"""llm_gateway.rl.slime_train — Slime 训练定制脚本。

主要入口：

- ``llm_gateway.rl.slime_train.tasks.{ingest,consolidate,retrieve}_reward.reward.reward_func``
  作为 Slime ``--custom-rm-path``。
- ``llm_gateway.rl.slime_train.memory_rl.custom_generate.custom_generate``
  作为 Slime ``--custom-generate-function-path``。
- ``llm_gateway.rl.slime_train.memory_rl.smoke_vllm_rollout_reward``
  纯 vLLM 端到端 smoke 测试。
"""
