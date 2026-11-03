# Windows fix: load pyarrow's dataset DLLs BEFORE torch. If torch is imported
# first, a later `import datasets` can crash with "access violation".
# Every script imports `src` before `torch` so this always runs first.
try:
    import pyarrow.dataset  # noqa: F401
except ImportError:
    pass
