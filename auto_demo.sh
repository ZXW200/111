#!/bin/bash

# 自动演示脚本 - 录视频时使用
# 这个脚本会自动运行所有命令，你只需要读台词

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 清屏函数
clear_screen() {
    sleep 1
    clear
}

# 暂停函数（自动继续，方便录制）
auto_pause() {
    local seconds=$1
    local message=$2
    echo -e "${YELLOW}>>> $message${NC}"
    echo -e "${BLUE}[Auto-continuing in $seconds seconds...]${NC}"
    sleep $seconds
}

echo -e "${GREEN}"
echo "============================================"
echo "  NTD Clinical Trials Analysis"
echo "  Group 16 - Video Demonstration"
echo "============================================"
echo -e "${NC}"
sleep 2

# ============================================
# SECTION 1: Introduction
# ============================================
clear_screen
echo -e "${GREEN}=== SECTION 1: Project Introduction ===${NC}\n"

echo -e "${BLUE}Our project files:${NC}"
ls -lh *.py
echo ""

echo -e "${BLUE}Documentation files:${NC}"
ls -lh CODE_DOCUMENTATION.txt README_CODE.md
echo ""

auto_pause 3 "Showing documentation preview..."
head -25 CODE_DOCUMENTATION.txt

auto_pause 5 "Moving to data cleaning..."

# ============================================
# SECTION 2: Data Cleaning
# ============================================
clear_screen
echo -e "${GREEN}=== SECTION 2: Data Cleaning ===${NC}\n"

# 准备干净的环境
rm -rf CleanedData/ CleanedDataPlt/
mkdir -p CleanedData CleanedDataPlt

echo -e "${BLUE}Running CleanData.py...${NC}\n"
python3 CleanData.py

echo ""
auto_pause 3 "Checking generated files..."

echo -e "${BLUE}Generated cleaned datasets:${NC}"
ls -lh CleanedData/
echo ""

auto_pause 5 "Moving to logistic regression..."

# ============================================
# SECTION 3: Logistic Regression
# ============================================
clear_screen
echo -e "${GREEN}=== SECTION 3: Logistic Regression Analysis ===${NC}\n"

echo -e "${BLUE}Running DataFit.py (Logistic Regression)...${NC}\n"
python3 DataFit.py

echo ""
auto_pause 2 "Checking visualization..."

echo -e "${BLUE}Generated visualization:${NC}"
ls -lh CleanedDataPlt/coefficients_plot.png

echo ""
echo -e "${YELLOW}>>> MANUAL ACTION: Open coefficients_plot.png in image viewer${NC}"
auto_pause 6 "Showing regression results..."

# ============================================
# SECTION 4: Network Analysis
# ============================================
clear_screen
echo -e "${GREEN}=== SECTION 4: Network Analysis ===${NC}\n"

echo -e "${BLUE}Running Network.py...${NC}\n"
python3 Network.py

echo ""
auto_pause 2 "Showing network statistics..."

echo -e "${BLUE}Top countries by collaboration:${NC}"
head -10 CleanedData/network_statistics.csv

auto_pause 5 "Moving to additional analyses..."

# ============================================
# SECTION 5: Additional Analyses
# ============================================
clear_screen
echo -e "${GREEN}=== SECTION 5: Drug Extraction & Visualization ===${NC}\n"

echo -e "${BLUE}Running ExtractDrug.py...${NC}\n"
python3 ExtractDrug.py

echo ""
auto_pause 2 "Running visualizations..."

echo -e "${BLUE}Running visualization.py...${NC}\n"
python3 visualization.py

echo ""
auto_pause 2 "Checking outputs..."

echo -e "${BLUE}All visualizations:${NC}"
ls -lh CleanedDataPlt/

echo ""
echo -e "${YELLOW}>>> MANUAL ACTION: Open sponsor_distribution.png${NC}"
auto_pause 4 "Preparing summary..."

# ============================================
# SECTION 6: Summary
# ============================================
clear_screen
echo -e "${GREEN}=== SECTION 6: Project Summary ===${NC}\n"

echo -e "${BLUE}Total CSV datasets created:${NC}"
ls CleanedData/*.csv | wc -l
ls CleanedData/*.csv

echo ""
echo -e "${BLUE}Total visualizations created:${NC}"
ls CleanedDataPlt/*.png | wc -l
ls CleanedDataPlt/*.png

echo ""
echo -e "${GREEN}"
echo "============================================"
echo "  Analysis Complete!"
echo "  - 12 CSV datasets"
echo "  - 3 Visualizations"
echo "  - All RQs addressed"
echo "============================================"
echo -e "${NC}"

echo ""
echo -e "${YELLOW}>>> VIDEO RECORDING COMPLETE${NC}"
echo -e "${BLUE}Thank you for watching!${NC}"
