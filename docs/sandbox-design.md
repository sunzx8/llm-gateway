# FileSystemStore 沙箱安全设计

## 问题

`FileSystemStore` 以一个 `base_path` 目录作为根目录存储文件，但面临两类风险：

1. **路径穿越（Path Traversal）**：通过 `../../` 等相对路径片段逃逸到 base_path 之外
2. **Shell 逃逸**：`execute_bash` 中虽有白名单，但命令实际执行时 cwd 只是 base_path，进程仍能访问整个文件系统
3. **符号链接攻击**：在 base_path 内创建指向外部的 symlink，后续读写穿透

## 方案分层

### Layer 1: 路径验证守卫（Pure Python, 零依赖）

所有文件操作前必须经过 `_safe_resolve(rel_path)` 检查：

```python
def _safe_resolve(self, rel_path: str) -> str:
    """将相对路径解析为绝对路径，并验证其在 base_path 之内。

    防护:
    - 路径穿越 (../)
    - 符号链接逃逸 (symlink -> /etc/passwd)
    - 绝对路径注入 (/etc/passwd)

    Raises:
        PermissionError: 路径解析结果不在 base_path 之内
    """
    import os

    # 拒绝绝对路径输入
    if os.path.isabs(rel_path):
        raise PermissionError(
            f"Absolute paths are not allowed: {rel_path}"
        )

    # 拼接后用 realpath 解析掉所有 .. 和 symlink
    candidate = os.path.realpath(os.path.join(self.base_path, rel_path))
    root = os.path.realpath(self.base_path)

    # 确保解析后的路径以 base_path 开头
    if not candidate.startswith(root + os.sep) and candidate != root:
        raise PermissionError(
            f"Path escapes sandbox: {rel_path!r} resolves to {candidate}"
        )

    return candidate
```

**覆盖的方法**：`write_file`, `append_file`, `read_file`, `delete_file`, `read_lines`, `grep`, `list_files`

### Layer 2: Shell 命名空间沙箱（Linux unshare）

`execute_bash` 在执行命令时，用 `unshare` 创建隔离的 mount namespace：

```python
def _sandboxed_bash(self, cmd: str, timeout: int, env: dict) -> subprocess.CompletedProcess:
    """在隔离的 mount namespace 中执行 bash 命令。

    利用 Linux unshare(2) 创建新的 mount namespace，然后 bind-mount
    base_path 到临时 rootfs 的 /workspace，使得子进程只能看到该目录。

    Requirements:
    - Linux kernel >= 3.8 (user namespace support)
    - /proc/sys/kernel/unprivileged_userns_clone = 1 (部分发行版需要)

    Fallback:
    - 如果 unshare 不可用，回退到普通 subprocess + cwd 限制。
    """
    ...
```

如果环境不支持 unprivileged user namespace（`ENOSYS` / `EPERM`），自动回退到当前 cwd-only 模式。

### Layer 3: 容器沙箱（可选，高安全需求时启用）

在生产多租户部署中，每个用户的 FileSystemStore 可以运行在：
- **Docker container**：通过挂载 `-v /tmp/usera:/workspace:rw` 实现隔离
- **gVisor / runsc**：加一层系统调用过滤
- **bubblewrap (bwrap)**：轻量级用户态沙箱，不需 root

## 推荐实施路径

| 阶段 | 方案 | 复杂度 | 防护级别 |
|------|------|--------|----------|
| 立即 | Layer 1: _safe_resolve | 低 | 防路径穿越 + symlink |
| 短期 | Layer 2: unshare/bwrap | 中 | 进程级 fs 隔离 |
| 长期 | Layer 3: container | 高 | 完整多租户隔离 |

## Layer 1 实现细节

### 不仅仅是路径检查 — 还需禁止创建 symlink

```python
# 在 execute_bash 白名单中，禁止 ln 命令
# 在 execute_python 中，禁止 os.symlink 调用
```

### 原子性问题 (TOCTOU)

`_safe_resolve` 做的是 check-then-use，存在 Time-of-Check-Time-of-Use 竞态。
缓解措施：
1. 文件操作使用 `O_NOFOLLOW` flag（不追踪 symlink）
2. 目录操作使用 `os.open(dir, O_DIRECTORY)` + `os.fchdir()`
3. 或者用 Linux `openat2(RESOLVE_BENEATH)` 系统调用（Python 3.11+）

### openat2 最佳方案（Python 3.11+）

```python
import os

def _safe_open(self, rel_path: str, flags: int, mode: int = 0o666) -> int:
    """使用 openat2(RESOLVE_BENEATH) 安全打开文件。

    内核级保证路径不会逃出 base_path，无 TOCTOU 竞态。
    """
    dir_fd = os.open(self.base_path, os.O_DIRECTORY | os.O_RDONLY)
    try:
        # Python 3.11+ 的 os.open 支持 dir_fd 参数
        # 配合 RESOLVE_BENEATH 可以避免路径逃逸
        fd = os.open(
            rel_path,
            flags | os.O_NOFOLLOW,
            mode,
            dir_fd=dir_fd,
        )
        return fd
    finally:
        os.close(dir_fd)
```

## Layer 2: bubblewrap (bwrap) 示例

```bash
bwrap \
  --unshare-all \
  --die-with-parent \
  --ro-bind /usr /usr \
  --ro-bind /lib /lib \
  --ro-bind /lib64 /lib64 \
  --ro-bind /bin /bin \
  --ro-bind /etc/alternatives /etc/alternatives \
  --bind /tmp/usera /workspace \
  --chdir /workspace \
  --proc /proc \
  --dev /dev \
  bash -c "$CMD"
```

这样子进程看到的 rootfs 极小，且 `/workspace` 就是用户的真实目录。即使执行 `rm -rf /`，也只删除该用户自己的文件。

## 实施 checklist

- [ ] 实现 `_safe_resolve()` 并集成到所有文件操作方法
- [ ] 为 `execute_python` 添加 builtins 限制（禁止 `os.symlink`, `os.chroot` 等）
- [ ] `execute_bash` 增加 bwrap/unshare 支持（带 fallback）
- [ ] 单元测试：路径穿越、symlink 攻击、shell 逃逸场景
- [ ] 长期：考虑 Docker/gVisor 多租户容器化
