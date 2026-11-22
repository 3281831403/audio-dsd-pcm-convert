"""让 pytest 从仓库根目录导入 dsd 包，保证离线直接运行。"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
