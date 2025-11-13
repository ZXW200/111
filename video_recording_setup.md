# 视频录制准备清单

## 推荐录屏工具（选一个）

### 方案A: OBS Studio (免费，推荐)
- 下载: https://obsproject.com/
- 优点: 专业、免费、可编辑
- 设置: 1920x1080, 30fps, MP4格式

### 方案B: Zoom (最简单)
- 打开Zoom → 新会议 → 共享屏幕 → 录制
- 优点: 一键录制，自动保存MP4
- 录制后从 文档/Zoom/ 找到文件

### 方案C: PowerPoint录制 (Windows)
- 打开PPT → 录制 → 屏幕录制
- 优点: Office自带，简单
- 缺点: 只有Windows

### 方案D: QuickTime (Mac)
- 打开QuickTime → 文件 → 新建屏幕录制
- 优点: Mac自带
- 快捷键: Command+Shift+5

## 录制前环境准备

```bash
# 1. 清理屏幕
cd /home/user/111
rm -rf CleanedData/ CleanedDataPlt/
mkdir -p CleanedData CleanedDataPlt

# 2. 测试代码运行
python3 CleanData.py

# 3. 准备终端
# - 字体放大到 16-18pt
# - 终端窗口全屏
# - 关闭其他程序
# - 关闭通知
```

## 录制设置

- 分辨率: 1920x1080 (或1280x720)
- 帧率: 30fps
- 格式: MP4
- 音频: 清晰（使用耳机麦克风）
- 背景: 安静环境
- 时长: 严格控制在3分钟内

## 彩排检查清单

□ 逐字稿已打印/显示在第二屏幕
□ 代码文件已在正确目录
□ 终端字体已放大
□ 音频测试清晰
□ 计时器准备好（手机秒表）
□ 完整演练1次，计时2分30秒-2分50秒

