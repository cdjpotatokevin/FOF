"""iFinD 数据通道：封装对 ifind-finance-data skill 的 call.py 的调用。

为什么用子进程：skill 的 call.py 在导入时即以"当前工作目录"读取 mcp_config.json，
并自带 BASE/URL/密钥逻辑。最稳的复用方式是在 skill 目录下以子进程跑它，
把 (server_type, tool_name, params) 传进去、JSON 结果打回来——不重复造轮子、
也不碰密钥明文。

注意：skill 目录路径里含会话 UUID，跨会话会变。定位顺序：
  1) 显式传入 skill_dir
  2) 环境变量 IFIND_SKILL_DIR
  3) 在 ~/Library/Application Support/Claude 下 glob 搜 ifind-finance-data（取最新）
"""
from __future__ import annotations
import json
import os
import subprocess
import sys
from pathlib import Path


_RUNNER = (
    "import os, sys, json\n"
    "sys.path.insert(0, os.getcwd())\n"
    "import call as _c\n"
    "st=os.environ['IF_ST']; tn=os.environ['IF_TN']; pr=json.loads(os.environ['IF_PARAMS'])\n"
    "res=_c.call(st, tn, pr)\n"
    "sys.stdout.write('__IF_RESULT__' + json.dumps(res, ensure_ascii=False))\n"
)


def find_skill_dir(skill_dir: str | None = None) -> Path:
    if skill_dir:
        p = Path(skill_dir)
        if (p / "call.py").exists():
            return p
        raise FileNotFoundError(f"skill_dir 下没有 call.py: {p}")
    env = os.environ.get("IFIND_SKILL_DIR")
    if env and (Path(env) / "call.py").exists():
        return Path(env)
    base = Path.home() / "Library/Application Support/Claude"
    cands = sorted(base.glob("**/skills/ifind-finance-data"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    for c in cands:
        if (c / "call.py").exists():
            return c
    raise FileNotFoundError(
        "未找到 ifind-finance-data skill 目录。请设置环境变量 IFIND_SKILL_DIR "
        "指向含 call.py 的 skill 目录。"
    )


class IFindClient:
    """对 skill call.py 的瘦封装。call() 直接返回 skill 的结果 dict。"""

    def __init__(self, skill_dir: str | None = None, timeout: int = 90):
        self.skill_dir = find_skill_dir(skill_dir)
        self.timeout = timeout
        self._check_token()

    def _check_token(self) -> None:
        cfg = self.skill_dir / "mcp_config.json"
        try:
            tok = json.loads(cfg.read_text(encoding="utf-8")).get("auth_token", "")
        except Exception:  # noqa: BLE001
            tok = ""
        if (not tok) or any(x in tok for x in ("您的", "your", "YOUR", "请填", "<")):
            raise RuntimeError(
                f"iFinD 密钥未配置：{cfg} 里的 auth_token 仍是占位符。"
                "请到 MCP官网→个人中心→密钥 获取后填入。"
            )

    def call(self, server_type: str, tool_name: str, params: dict) -> dict:
        env = dict(os.environ)
        env.update({
            "IF_ST": server_type,
            "IF_TN": tool_name,
            "IF_PARAMS": json.dumps(params, ensure_ascii=False),
        })
        proc = subprocess.run(
            [sys.executable, "-c", _RUNNER],
            cwd=str(self.skill_dir),
            env=env,
            capture_output=True,
            text=True,
            timeout=self.timeout,
        )
        out = proc.stdout
        marker = "__IF_RESULT__"
        if marker in out:
            payload = out.split(marker, 1)[1].strip()
            return json.loads(payload)
        raise RuntimeError(
            f"iFinD 调用无有效返回。stderr={proc.stderr[-500:]} stdout={out[-300:]}"
        )


def extract_text_blocks(result: dict) -> str:
    """从 MCP 工具返回里抽取文本内容（result.content[].text 拼接）。

    标准 MCP tools/call 返回形如：
      {"ok":True,"data":{"result":{"content":[{"type":"text","text":"..."}]}}}
    这里做容错：兼容 data 直接是 result、或 content 在不同层级的情况。
    """
    if not isinstance(result, dict):
        return str(result)
    data = result.get("data", result)
    # 逐层找 content 列表
    node = data
    for key in ("result", "content"):
        if isinstance(node, dict) and key in node:
            node = node[key]
    texts: list[str] = []
    if isinstance(node, list):
        for blk in node:
            if isinstance(blk, dict) and "text" in blk:
                texts.append(str(blk["text"]))
            else:
                texts.append(str(blk))
    elif isinstance(node, str):
        texts.append(node)
    else:
        texts.append(json.dumps(data, ensure_ascii=False))
    return "\n".join(texts)
