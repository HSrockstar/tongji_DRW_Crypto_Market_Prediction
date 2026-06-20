import importlib
import platform
import sys


PACKAGES = [
    "kaggle",
    "pandas",
    "numpy",
    "scipy",
    "sklearn",
    "pyarrow",
    "polars",
    "tqdm",
    "matplotlib",
    "seaborn",
    "lightgbm",
    "xgboost",
    "catboost",
    "joblib",
    "psutil",
]


def check_package(package_name: str) -> tuple[bool, str]:
    try:
        importlib.import_module(package_name)
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return True, ""


def main() -> int:
    print("Python 版本:")
    print(sys.version)
    print()
    print("系统信息:")
    print(f"系统: {platform.system()}")
    print(f"版本: {platform.version()}")
    print(f"平台: {platform.platform()}")
    print(f"处理器: {platform.processor()}")
    print(f"Python 可执行文件: {sys.executable}")
    print()

    failed_packages = []
    print("包导入检查:")
    for package_name in PACKAGES:
        ok, message = check_package(package_name)
        if ok:
            print(f"{package_name}: OK")
        else:
            failed_packages.append(package_name)
            print(f"{package_name}: FAIL ({message})")

    print()
    if failed_packages:
        print("环境校验失败，失败包名:")
        print(", ".join(failed_packages))
        return 1

    print("环境校验通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
