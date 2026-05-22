"""Eval Dashboard 服务。

基于 Python 内置 http.server 提供评测结果可视化面板，
无需额外安装 Web 框架依赖。

使用方式::

    # 方式 1：直接运行
    python -m eval.dashboard.server --results-dir eval/results

    # 方式 2：指定端口
    python -m eval.dashboard.server --port 8080 --results-dir eval/results
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import sys
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

logger = logging.getLogger(__name__)

# 模板目录
_TEMPLATE_DIR = str(Path(__file__).resolve().parent / "templates")

# 默认结果目录
_DEFAULT_RESULTS_DIR = str(
    Path(__file__).resolve().parent.parent / "results"
)


class DashboardHandler(SimpleHTTPRequestHandler):
    """Dashboard HTTP 请求处理器。"""

    results_dir: str = _DEFAULT_RESULTS_DIR

    def __init__(self, *args, results_dir: str = _DEFAULT_RESULTS_DIR, **kwargs):
        self.results_dir = results_dir
        super().__init__(*args, directory=_TEMPLATE_DIR, **kwargs)

    def do_GET(self):
        """处理 GET 请求。"""
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/index.html":
            # 返回主页面
            self._serve_template("index.html")
        elif path == "/api/results":
            # 返回所有评测结果文件列表（含摘要信息）
            self._serve_results_list()
        elif path.startswith("/api/result/"):
            # 返回指定评测结果文件内容
            filename = path[len("/api/result/"):]
            self._serve_result_file(filename)
        else:
            # 静态文件
            super().do_GET()

    def _serve_template(self, template_name: str) -> None:
        """返回模板文件。"""
        template_path = os.path.join(_TEMPLATE_DIR, template_name)
        if not os.path.exists(template_path):
            self.send_error(404, f"Template not found: {template_name}")
            return
        with open(template_path, "r", encoding="utf-8") as f:
            content = f.read()
        body = content.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.end_headers()
        self.wfile.write(body)

    def _serve_results_list(self) -> None:
        """返回结果目录下所有 JSON 文件列表，包含每个文件的摘要信息。

        摘要信息包含 benchmark 级别的 stats、metrics 等，
        但不包含 per_conversation_results 中的详细数据，
        以避免大文件导致前端加载失败。
        """
        results_dir = self.results_dir
        files: list[dict[str, Any]] = []

        if os.path.isdir(results_dir):
            for fname in sorted(os.listdir(results_dir), reverse=True):
                if fname.endswith(".json"):
                    fpath = os.path.join(results_dir, fname)
                    stat = os.stat(fpath)
                    file_info: dict[str, Any] = {
                        "filename": fname,
                        "size_bytes": stat.st_size,
                        "modified_time": stat.st_mtime,
                    }
                    # 提取摘要信息（去掉 per_conversation_results 以减小体积）
                    summary = self._extract_file_summary(fpath)
                    if summary:
                        file_info["summary"] = summary
                    files.append(file_info)

        self._send_json({"files": files})

    def _extract_file_summary(self, fpath: str) -> dict[str, Any] | None:
        """从评测结果文件中提取摘要信息（不含 per_conversation_results）。

        对于大文件（>10MB），使用 jq 命令行工具提取以避免内存问题；
        对于小文件，直接使用 json.load。
        """
        try:
            file_size = os.path.getsize(fpath)
            if file_size > 10 * 1024 * 1024:  # >10MB
                return self._extract_summary_via_jq(fpath)
            return self._extract_summary_via_json(fpath)
        except Exception as e:
            logger.warning("提取文件摘要失败 %s: %s", fpath, e)
            return None

    def _extract_summary_via_json(self, fpath: str) -> dict[str, Any] | None:
        """使用 json.load 提取摘要（适用于小文件）。"""
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)

        return self._build_summary_from_data(data)

    def _extract_summary_via_jq(self, fpath: str) -> dict[str, Any] | None:
        """使用 jq 命令行工具提取摘要（适用于大文件，避免内存问题）。"""
        import subprocess

        jq_filter = (
            '{eval_run_id: .eval_run_id, created_at: .created_at, '
            'eval_mode: .eval_mode, base_model: .base_model, memory_model: .memory_model, '
            'results: [.results[] | {benchmark_name, metrics, elapsed_seconds, stats, judge_scores, '
            'conversation_summaries: [.per_conversation_results[]? | '
            '{conv_id, mcq_accuracy, elapsed_seconds, predictions_count: (.predictions | length), error}]}]}'
        )
        try:
            result = subprocess.run(
                ["jq", jq_filter, fpath],
                capture_output=True, text=True, timeout=60,
            )
            if result.returncode != 0:
                logger.warning("jq 提取摘要失败 %s: %s", fpath, result.stderr)
                # 回退到 json.load
                return self._extract_summary_via_json(fpath)
            return json.loads(result.stdout)
        except FileNotFoundError:
            logger.warning("jq 未安装，回退到 json.load 方式提取摘要")
            return self._extract_summary_via_json(fpath)
        except subprocess.TimeoutExpired:
            logger.warning("jq 提取摘要超时 %s", fpath)
            return None

    def _build_summary_from_data(self, data: dict[str, Any]) -> dict[str, Any]:
        """从完整数据中构建摘要字典。"""
        summary: dict[str, Any] = {
            "eval_run_id": data.get("eval_run_id", ""),
            "created_at": data.get("created_at", ""),
            "eval_mode": data.get("eval_mode", "memory"),
            "base_model": data.get("base_model", ""),
            "memory_model": data.get("memory_model", ""),
        }

        results = data.get("results", [])
        if not results and data.get("benchmark_name"):
            # 兼容旧格式：顶层就是单个 benchmark 结果
            results = [data]

        summary_results = []
        for r in results:
            bench_summary = {
                "benchmark_name": r.get("benchmark_name", "Unknown"),
                "metrics": r.get("metrics", {}),
                "elapsed_seconds": r.get("elapsed_seconds", 0),
                "stats": r.get("stats", {}),
                "judge_scores": r.get("judge_scores", []),
            }
            # 提取 per_conversation 摘要（仅关键字段，减小体积）
            per_conv = r.get("per_conversation_results", [])
            if per_conv:
                conv_summaries = []
                for c in per_conv:
                    conv_summaries.append({
                        "conv_id": c.get("conv_id", ""),
                        "mcq_accuracy": c.get("mcq_accuracy", 0),
                        "elapsed_seconds": c.get("elapsed_seconds", 0),
                        "predictions_count": len(c.get("predictions", [])),
                        "error": c.get("error", ""),
                    })
                bench_summary["conversation_summaries"] = conv_summaries
            summary_results.append(bench_summary)

        summary["results"] = summary_results
        return summary

    def _serve_result_file(self, filename: str) -> None:
        """返回指定结果文件的内容。

        直接读取原始 JSON 文件并以 gzip 压缩流式发送，
        避免 json.load + json.dumps 对大文件的巨大开销。
        """
        # 安全检查：防止路径穿越
        if ".." in filename or "/" in filename or "\\" in filename:
            self.send_error(400, "Invalid filename")
            return

        fpath = os.path.join(self.results_dir, filename)
        if not os.path.exists(fpath):
            self.send_error(404, f"Result file not found: {filename}")
            return

        try:
            # 检查客户端是否支持 gzip
            accept_encoding = self.headers.get("Accept-Encoding", "")
            use_gzip = "gzip" in accept_encoding

            file_size = os.path.getsize(fpath)

            if use_gzip:
                # 读取原始文件并 gzip 压缩后发送（32MB JSON 压缩后通常只有几 MB）
                with open(fpath, "rb") as f:
                    raw = f.read()
                body = gzip.compress(raw, compresslevel=6)
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Encoding", "gzip")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(body)
            else:
                # 不支持 gzip 时，直接流式发送原始文件（避免 json.load/dumps）
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(file_size))
                self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                with open(fpath, "rb") as f:
                    while True:
                        chunk = f.read(65536)  # 64KB 分块发送
                        if not chunk:
                            break
                        self.wfile.write(chunk)
        except Exception as e:
            self.send_error(500, f"Failed to read result file: {e}")

    def _send_json(self, data: Any) -> None:
        """发送 JSON 响应。"""
        body = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        """使用 logging 输出日志。"""
        logger.info(format, *args)


def run_server(port: int = 8088, results_dir: str = _DEFAULT_RESULTS_DIR) -> None:
    """启动 Dashboard HTTP 服务。"""
    handler = partial(DashboardHandler, results_dir=results_dir)
    server = HTTPServer(("0.0.0.0", port), handler)
    print(f"🚀 Eval Dashboard 已启动: http://localhost:{port}")
    print(f"📁 结果目录: {results_dir}")
    print("按 Ctrl+C 停止服务")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n⏹ Dashboard 已停止")
        server.server_close()


def main() -> None:
    """CLI 入口。"""
    parser = argparse.ArgumentParser(
        description="Eval Dashboard —— 评测结果可视化面板",
    )
    parser.add_argument(
        "--port", type=int, default=8088,
        help="服务端口（默认 8088）",
    )
    parser.add_argument(
        "--results-dir", type=str, default=_DEFAULT_RESULTS_DIR,
        help="评测结果 JSON 文件目录",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    run_server(port=args.port, results_dir=args.results_dir)


if __name__ == "__main__":
    main()
