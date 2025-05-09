#!/bin/bash

# Install dependencies
uv pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# Install QFactory
uv pip install . -i https://pypi.tuna.tsinghua.edu.cn/simple --reinstall

# Test the installation
python3 tests/import_test.py
