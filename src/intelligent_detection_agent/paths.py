"""项目级资源路径。"""

from pathlib import Path


# 源码采用 src 布局；数据、模型、前端和运行产物仍位于仓库根目录。
PROJECT_ROOT = Path(__file__).resolve().parents[2]

