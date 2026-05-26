"""快速测试：验证 VectorStore 中 embedder 是否正确注入、embedding 服务是否可达。

运行方式（项目根目录）：
    python -m pytest tests/test_embedder_wiring.py -v -s
    或直接：
    python tests/test_embedder_wiring.py
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 直接使用 config.yaml 中 embedding 段的硬编码配置（无需环境变量）
EMBEDDING_CONFIG = {
    "provider": "openai_compat",
    "model": "bge-m3",
    "api_key": "EMPTY",
    "base_url": "http://81.70.50.154:8082/v1",
    "batch_size": 64,
    "dimensions": None,
    "timeout": 120.0,
    "max_retries": 3,
}


def test_embedder_not_none_in_vectorstore():
    """验证通过 VectorStore 创建时，embedder 不为 None。"""
    from utils.memory_llm_interface import EmbeddingInterface
    from storage.vector_stores import VectorStore

    embedder = EmbeddingInterface(EMBEDDING_CONFIG)
    vec = VectorStore(embedding_interface=embedder)

    # 核心断言：embedder 必须非空
    assert vec.embedder is not None, "❌ VectorStore.embedder is None!"
    assert vec.embedder is embedder, "❌ VectorStore.embedder 不是传入的那个实例!"
    print("✅ VectorStore.embedder 已正确注入，非 None")
    print(f"   model={vec.embedder.model}, base_url={vec.embedder.base_url}")


def test_embedder_config_matches_yaml():
    """验证 embedder 的配置和 config.yaml 一致。"""
    from utils.memory_llm_interface import EmbeddingInterface

    embedder = EmbeddingInterface(EMBEDDING_CONFIG)

    assert embedder.model == "bge-m3", f"❌ 期望 model=bge-m3，实际={embedder.model}"
    assert "81.70.50.154:8082" in (embedder.base_url or ""), \
        f"❌ 期望 base_url 包含 81.70.50.154:8082，实际={embedder.base_url}"
    print("✅ embedder 配置与 config.yaml 一致")
    print(f"   provider={embedder.provider}, model={embedder.model}")
    print(f"   base_url={embedder.base_url}")


def test_embedding_service_reachable():
    """验证 embedding 服务实际可达（发一个真实请求）。"""
    from utils.memory_llm_interface import EmbeddingInterface

    embedder = EmbeddingInterface(EMBEDDING_CONFIG)

    async def _test():
        try:
            result = await embedder.embed(["hello world test"])
            assert len(result) == 1, f"❌ 期望返回 1 个向量，实际={len(result)}"
            assert len(result[0]) > 0, "❌ 返回的向量维度为 0"
            print(f"✅ embedding 服务可达，返回向量维度={len(result[0])}")
            return True
        except Exception as e:
            print(f"❌ embedding 服务不可达或超时: {type(e).__name__}: {e}")
            return False

    reachable = asyncio.run(_test())
    assert reachable, "embedding 服务连接失败"


def test_vectorstore_add_produces_embeddings():
    """验证 VectorStore.add 后条目确实有非空 embedding。"""
    from utils.memory_llm_interface import EmbeddingInterface
    from storage.vector_stores import VectorStore

    embedder = EmbeddingInterface(EMBEDDING_CONFIG)
    vec = VectorStore(embedding_interface=embedder)

    async def _test():
        vec.create_collection("test_coll")
        ids = await vec.add(
            collection="test_coll",
            texts=["我喜欢吃火锅", "今天天气不错"],
            metadatas=[{"type": "fact"}, {"type": "event"}],
        )
        assert len(ids) == 2, f"❌ 期望 add 返回 2 个 id，实际={len(ids)}"

        # 检查每个条目的 embedding 是否非空
        for entry in vec._collections["test_coll"]:
            emb = entry.get("embedding", [])
            if not emb:
                print(f"❌ 条目 {entry['id']} 的 embedding 为空！embedder 可能未生效")
                assert False, f"条目 {entry['id']} embedding 为空"
            else:
                print(f"✅ 条目 {entry['id']}: embedding dim={len(emb)}")

        # 测试检索也能用
        results = await vec.search("test_coll", "火锅", top_k=2)
        assert len(results) > 0, "❌ 检索返回 0 条结果"
        print(f"✅ 检索正常，top1 score={results[0]['score']:.4f}, text={results[0]['text']}")

    asyncio.run(_test())


def test_graph_embedder_wiring():
    """验证 graph store 的 embedder 传递路径。"""
    from utils.memory_llm_interface import EmbeddingInterface
    from storage.vector_stores import VectorStore
    from storage.graph_stores import GraphStore

    embedder = EmbeddingInterface(EMBEDDING_CONFIG)

    # 模拟 make_stores 中 memory 后端的行为
    vec = VectorStore(embedding_interface=embedder)
    graph = GraphStore()

    # GraphStore（内存版）默认没有 embedder 属性，需要看 ingest 里如何赋值
    graph_embedder = getattr(graph, "embedder", "NOT_SET")
    if graph_embedder == "NOT_SET":
        print("⚠️  GraphStore 没有 embedder 属性（内存版默认不带）")
        print("   ingest_task 通过 getattr(self.graph, 'embedder', None) 获取")
        print("   需确认 graph.embedder 是否在其他地方被赋值")
    elif graph_embedder is None:
        print("⚠️  GraphStore.embedder = None（graph node/edge embedding 将不生效）")
    else:
        print(f"✅ GraphStore.embedder 已设置: model={graph_embedder.model}")


if __name__ == "__main__":
    print("=" * 60)
    print("Embedder 注入 & 服务可达性快速测试")
    print("=" * 60)

    tests = [
        ("1. VectorStore embedder 非空", test_embedder_not_none_in_vectorstore),
        ("2. embedder 配置匹配 yaml", test_embedder_config_matches_yaml),
        ("3. embedding 服务可达", test_embedding_service_reachable),
        ("4. VectorStore.add 产生非空 embedding", test_vectorstore_add_produces_embeddings),
        ("5. GraphStore embedder 检查", test_graph_embedder_wiring),
    ]

    failed = 0
    for name, fn in tests:
        print(f"\n--- {name} ---")
        try:
            fn()
        except Exception as e:
            print(f"❌ FAILED: {e}")
            failed += 1

    print("\n" + "=" * 60)
    if failed:
        print(f"结果: {len(tests) - failed}/{len(tests)} 通过, {failed} 失败")
        sys.exit(1)
    else:
        print(f"结果: 全部 {len(tests)} 项通过 ✅")
