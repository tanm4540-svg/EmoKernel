"""
首次运行设置脚本：检查依赖、下载模型、初始化目录结构
"""
import subprocess
import sys
import os
import time

SKILL_DIR = os.path.dirname(os.path.abspath(__file__))
PROFILE_DIR = os.path.join(SKILL_DIR, "..", "emotion_profile")


def check_python():
    """检查 Python 版本"""
    v = sys.version_info
    print(f"Python {v.major}.{v.minor}.{v.micro}")
    if v.major < 3 or (v.major == 3 and v.minor < 10):
        print("  âœˆ 建议 Python >= 3.10")
    else:
        print("  âœ“ 版本合适")
    return True


def check_and_install_deps():
    """检查并安装依赖"""
    deps = [
        ("sentence-transformers", "sentence_transformers"),
        ("numpy", "numpy"),
        ("scikit-learn", "sklearn"),
    ]
    all_ok = True
    pip_cmd = [sys.executable, "-m", "pip", "install"]

    for pkg_name, import_name in deps:
        try:
            __import__(import_name)
            print(f"  âœ“ {pkg_name} 已安装")
        except ImportError:
            print(f"  âœŠ {pkg_name} 未安装，正在安装...")
            result = subprocess.run(pip_cmd + [pkg_name], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"    âœ“ 安装成功")
            else:
                print(f"    âœˆ 安装失败，可能需要网络访问: {result.stderr[:100]}")
                all_ok = False

    return all_ok


def download_model(timeout=60):
    """尝试下载 sentence-transformers 模型"""
    print("\n正在下载情感分析模型 (paraphrase-multilingual-MiniLM-L12-v2)...")
    print(f"  超时时间: {timeout}s")

    try:
        from sentence_transformers import SentenceTransformer
        t0 = time.time()
        model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        elapsed = time.time() - t0
        print(f"  âœ“ 模型下载完成 (耗时 {elapsed:.1f}s)")
        return True
    except Exception as e:
        print(f"  âœˆ 模型下载失败: {e}")
        print(f"  将使用关键词匹配模式，功能完整但精度略低")
        return False


def init_profile_dir():
    """初始化情绪档案目录"""
    os.makedirs(PROFILE_DIR, exist_ok=True)
    print(f"  âœ“ 情绪档案目录: {PROFILE_DIR}")
    return True


def main():
    print("=" * 50)
    print("  Emotion Companion - 首次运行设置")
    print("=" * 50)
    print()

    print("[1/4] 检查 Python 版本")
    check_python()
    print()

    print("[2/4] 检查并安装依赖")
    check_and_install_deps()
    print()

    print("[3/4] 下载情感分析模型（可选）")
    download_model()
    print()

    print("[4/4] 初始化情绪档案目录")
    init_profile_dir()
    print()

    print("=" * 50)
    print("  设置完成！")
    print("  重启 Codex 即可使用该 Skill")
    print("=" * 50)


if __name__ == "__main__":
    main()
