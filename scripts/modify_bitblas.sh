#!/bin/bash

# Target file path
FILE_PATH=".venv-kernel/lib/python3.12/site-packages/bitblas/ops/general_matmul/__init__.py"

# Check if file exists
if [ ! -f "$FILE_PATH" ]; then
    echo "Error: File $FILE_PATH does not exist"
    exit 1
fi

# Verify the file has at least 130 lines
TOTAL_LINES=$(wc -l < "$FILE_PATH")
if [ "$TOTAL_LINES" -lt 130 ]; then
    echo "Error: File $FILE_PATH has fewer than 130 lines"
    exit 1
fi

# Check if line 130 contains the expected pattern
TARGET_LINE=$(sed -n '130p' "$FILE_PATH")
EXPECTED_PATTERN='object.__setattr__(self, "propagate_b", TransformKind.LDMatrixTransform)'
if [[ ! "$TARGET_LINE" == *"$EXPECTED_PATTERN"* ]]; then
    echo "Error: Line 130 does not contain the expected content"
    echo "Actual content: $TARGET_LINE"
    exit 1
fi

# Modify line 130
sed -i '130s/TransformKind.LDMatrixTransform/TransformKind.NonTransform/' "$FILE_PATH"

echo "File modified successfully!"
exit 0
