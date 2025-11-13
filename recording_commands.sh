#!/bin/bash
# 视频录制命令清单 - 按顺序执行
# 可以直接复制粘贴到终端

echo "============================================"
echo "NTD Analysis - Video Recording Commands"
echo "============================================"
echo ""

# SECTION 1: Introduction
echo "=== SECTION 1: Introduction ==="
ls -lh *.py
ls CODE_DOCUMENTATION.txt README_CODE.md
head -30 CODE_DOCUMENTATION.txt

echo ""
echo "Press ENTER to continue to Section 2..."
read

# SECTION 2: Data Cleaning
echo "=== SECTION 2: Data Cleaning ==="
# Clean previous outputs for fresh demo
rm -rf CleanedData/ CleanedDataPlt/
mkdir -p CleanedData CleanedDataPlt

# Show the classification function (open in editor manually)
echo "Now open CleanData.py and scroll to lines 26-46"
echo "Press ENTER when ready to run the script..."
read

# Run data cleaning
python3 CleanData.py

echo ""
echo "Check the generated files:"
ls -lh CleanedData/
echo ""
echo "Press ENTER to continue to Section 3..."
read

# SECTION 3: Logistic Regression
echo "=== SECTION 3: Logistic Regression ==="
echo "Now open DataFit.py and show lines 14-16 and 24-29"
echo "Press ENTER when ready to run..."
read

python3 DataFit.py

echo ""
echo "Check the visualization:"
ls -lh CleanedDataPlt/coefficients_plot.png
echo "Now open this PNG file in image viewer"
echo ""
echo "Press ENTER to continue to Section 4..."
read

# SECTION 4: Network Analysis
echo "=== SECTION 4: Network Analysis ==="
echo "Now open Network.py and scroll to lines 36-44"
echo "Press ENTER when ready to run..."
read

python3 Network.py

echo ""
echo "Show network statistics:"
head -10 CleanedData/network_statistics.csv
echo ""
echo "Press ENTER to continue to Section 5..."
read

# SECTION 5: Additional Analyses
echo "=== SECTION 5: Drug Extraction and Visualization ==="
echo "Now open ExtractDrug.py and show line 50 (regex)"
echo "Press ENTER when ready to run..."
read

python3 ExtractDrug.py

echo ""
echo "Now run visualization:"
python3 visualization.py

echo ""
echo "Show the sponsor distribution chart:"
ls -lh CleanedDataPlt/sponsor_distribution.png
echo "Now open this PNG file"
echo ""
echo "Press ENTER to continue to Section 6..."
read

# SECTION 6: Conclusion
echo "=== SECTION 6: Conclusion ==="
echo "Count CSV files:"
ls CleanedData/*.csv | wc -l

echo ""
echo "List visualizations:"
ls CleanedDataPlt/*.png

echo ""
echo "Show project summary:"
cat README_CODE.md

echo ""
echo "============================================"
echo "Recording Complete!"
echo "============================================"
