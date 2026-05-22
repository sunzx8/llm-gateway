#!/usr/bin/env python3
"""从 traj_cpx_8e9dec70 目录加载一个完整快照并打印所有内容。

用法:
    cd /data/cloud_disk_1/erenpeng/llm-gateway
    python -m rl.rl_env.load_snapshot_demo

或直接:
    PYTHONPATH=/data/cloud_disk_1/erenpeng/llm-gateway python /data/cloud_disk_1/erenpeng/llm-gateway/rl/rl_env/load_snapshot_demo.py
"""

from __future__ import annotations

import io
import json
import os
import sys
import tempfile

# 确保能找到项目模块
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import orjson
import zstandard as zstd

from rl.rl_env.serialize import (
    EncodedSnapshot,
    load_encoded,
    _zstd_decompress,
    _loads_json_with_bytes,
    _ndarray_from_bytes,
)

# ===========================================================================
# 配置
# ===========================================================================
SNAPSHOT_DIR = "/data/cloud_disk_1/erenpeng/datasets/merged_stage1_e2e/snapshots/traj_cpx_8e9dec70"


def load_and_print_snapshot(cbsnap_path: str) -> None:
    """加载单个 .cbsnap 文件并打印其全部内容。"""
    print(f"\n{'='*80}")
    print(f"📂 加载快照文件: {os.path.basename(cbsnap_path)}")
    print(f"   路径: {cbsnap_path}")
    print(f"   大小: {os.path.getsize(cbsnap_path) / 1024:.1f} KB")
    print(f"{'='*80}")

    # 1. 加载二进制 .cbsnap
    encoded = load_encoded(cbsnap_path)
    print(f"\n📋 基本信息:")
    print(f"   snapshot_id: {encoded.snapshot_id}")
    print(f"   fs_mode:     {encoded.fs_mode}")
    print(f"   blob 大小:   {len(encoded.blob) / 1024:.1f} KB")
    print(f"   fs_blobs 数: {len(encoded.fs_blobs)}")

    # 2. 解压 blob 获取 JSON 结构
    raw_json = _zstd_decompress(encoded.blob)
    doc = _loads_json_with_bytes(raw_json)

    # 3. 打印 meta
    meta = doc.get("meta", {})
    print(f"\n📝 Meta 元数据:")
    if meta:
        print(json.dumps(meta, indent=2, ensure_ascii=False, default=str))
    else:
        print("   (空)")

    # 4. 打印 FS (文件系统)
    print(f"\n📁 文件系统 (FS):")
    fs_payload = doc.get("fs", {})
    fs_mode = fs_payload.get("mode", "unknown")
    print(f"   mode: {fs_mode}")

    if fs_mode == "tar_zst":
        import tarfile
        tar_bytes = _zstd_decompress(fs_payload["tar_zst"])
        print(f"   tar 解压后大小: {len(tar_bytes) / 1024:.1f} KB")
        with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r") as tf:
            members = tf.getmembers()
            print(f"   文件数量: {len(members)}")
            print(f"\n   文件列表:")
            for m in sorted(members, key=lambda x: x.name):
                if m.isfile():
                    print(f"     📄 {m.name} ({m.size} bytes)")
                elif m.isdir():
                    print(f"     📁 {m.name}/")

            # 打印每个文件的内容
            print(f"\n   文件内容:")
            for m in sorted(members, key=lambda x: x.name):
                if m.isfile() and m.size > 0:
                    print(f"\n   {'─'*60}")
                    print(f"   📄 {m.name} ({m.size} bytes):")
                    print(f"   {'─'*60}")
                    f = tf.extractfile(m)
                    if f:
                        content = f.read()
                        try:
                            text = content.decode("utf-8")
                            # 限制打印长度
                            if len(text) > 3000:
                                print(f"   {text[:3000]}")
                                print(f"   ... (截断, 共 {len(text)} 字符)")
                            else:
                                # 缩进打印
                                for line in text.split("\n"):
                                    print(f"   {line}")
                        except UnicodeDecodeError:
                            print(f"   (二进制内容, {len(content)} bytes)")

    elif fs_mode == "cas":
        tree = fs_payload.get("tree", [])
        empty_dirs = fs_payload.get("empty_dirs", [])
        print(f"   文件数量: {len(tree)}")
        print(f"   空目录数: {len(empty_dirs)}")
        for node in tree:
            print(f"     📄 {node['path']} (sha256={node['sha256'][:12]}..., {node['size']} bytes)")

    # 5. 打印 Vec (向量存储)
    print(f"\n🔢 向量存储 (Vec):")
    vec_payload = doc.get("vec", {})
    id_counter = vec_payload.get("id_counter", 0)
    collections = vec_payload.get("collections", {})
    print(f"   id_counter: {id_counter}")
    print(f"   collections 数量: {len(collections)}")

    for coll_name, coll_data in collections.items():
        entries = coll_data.get("entries", [])
        dim = coll_data.get("dim", 0)
        embeddings_raw = coll_data.get("embeddings", b"")
        has_emb_raw = coll_data.get("has_embedding", b"")

        print(f"\n   📦 Collection: '{coll_name}'")
        print(f"      条目数: {len(entries)}")
        print(f"      向量维度: {dim}")

        # 解析 embeddings
        has_emb_arr = None
        if has_emb_raw and isinstance(has_emb_raw, bytes) and len(has_emb_raw) > 0:
            has_emb_arr = _ndarray_from_bytes(has_emb_raw)

        for i, entry in enumerate(entries):
            has_embedding = "✓" if (has_emb_arr is not None and i < len(has_emb_arr) and has_emb_arr[i]) else "✗"
            text_preview = entry.get("text", "")
            if len(text_preview) > 200:
                text_preview = text_preview[:200] + "..."
            metadata = entry.get("metadata", {})

            print(f"\n      [{i}] id={entry.get('id')} (embedding: {has_embedding})")
            print(f"          text: {text_preview}")
            if metadata:
                meta_str = json.dumps(metadata, ensure_ascii=False, default=str)
                if len(meta_str) > 300:
                    meta_str = meta_str[:300] + "..."
                print(f"          metadata: {meta_str}")

    # 6. 打印 Graph (图存储)
    print(f"\n🕸️  图存储 (Graph):")
    graph_payload = doc.get("graph", {})
    nodes = graph_payload.get("nodes", {})
    edges = graph_payload.get("edges", [])
    edge_counter = graph_payload.get("edge_counter", 0)
    node_embeddings = graph_payload.get("node_embeddings", {})
    edge_embeddings = graph_payload.get("edge_embeddings", {})

    print(f"   节点数: {len(nodes)}")
    print(f"   边数:   {len(edges)}")
    print(f"   edge_counter: {edge_counter}")
    print(f"   有 node embedding 的节点数: {len(node_embeddings)}")
    print(f"   有 edge embedding 的边数:   {len(edge_embeddings)}")

    if nodes:
        print(f"\n   节点:")
        for node_id, node_data in list(nodes.items())[:50]:  # 限制打印前50个
            label = node_data.get("label", "")
            props = node_data.get("properties", {})
            props_str = json.dumps(props, ensure_ascii=False, default=str)
            if len(props_str) > 200:
                props_str = props_str[:200] + "..."
            print(f"     🔵 [{node_id}] label={label}")
            print(f"        properties: {props_str}")
        if len(nodes) > 50:
            print(f"     ... (还有 {len(nodes) - 50} 个节点)")

    if edges:
        print(f"\n   边:")
        for edge in edges[:50]:  # 限制打印前50条
            src = edge.get("source", "")
            tgt = edge.get("target", "")
            rel = edge.get("relation", "")
            props = edge.get("properties", {})
            props_str = json.dumps(props, ensure_ascii=False, default=str)
            if len(props_str) > 200:
                props_str = props_str[:200] + "..."
            print(f"     ➡️  {src} --[{rel}]--> {tgt}")
            if props_str != "{}":
                print(f"        properties: {props_str}")
        if len(edges) > 50:
            print(f"     ... (还有 {len(edges) - 50} 条边)")

    print(f"\n{'='*80}")
    print(f"✅ 快照加载完成!")
    print(f"{'='*80}\n")


def main():
    # 获取目录下的所有 .cbsnap 文件
    cbsnap_files = sorted([
        os.path.join(SNAPSHOT_DIR, f)
        for f in os.listdir(SNAPSHOT_DIR)
        if f.endswith(".cbsnap")
    ])

    print(f"📂 快照目录: {SNAPSHOT_DIR}")
    print(f"📊 共找到 {len(cbsnap_files)} 个快照文件")

    if not cbsnap_files:
        print("❌ 未找到任何 .cbsnap 文件")
        return

    # 加载并打印第一个快照（作为演示）
    # 如果需要加载所有快照，取消下面循环的注释
    print(f"\n🎯 加载第一个快照作为演示...")
    load_and_print_snapshot(cbsnap_files[0])

    # 如需加载所有快照，取消以下注释:
    # for cbsnap_path in cbsnap_files:
    #     load_and_print_snapshot(cbsnap_path)


if __name__ == "__main__":
    main()
