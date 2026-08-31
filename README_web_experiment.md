# 音频实验 Web 版本

这是一个基于Web的音频实验系统，允许被试者通过浏览器参与实验，数据自动保存到本地。

## 功能特点

- 🌐 **Web界面**: 美观的现代化界面，支持移动端
- 🎵 **音频播放**: 自动同步音频播放，精确计时
- 📊 **实时数据**: 自动记录反应时间和选择结果
- 💾 **数据保存**: 多种格式保存（JSON、CSV、TXT）
- 🔄 **自动同步**: 实验进度自动同步到所有参与者
- ⌨️ **快捷键支持**: 支持键盘快捷键操作

## 安装依赖

```bash
pip install -r requirements.txt
```

## 快速开始

### 方法1: 使用启动脚本（推荐）
```bash
python start_web_experiment.py
```

### 方法2: 直接运行
```bash
python stilumi_web.py
```

## 使用方法

### 1. 启动服务器
运行启动脚本后，你会看到类似以下信息：
```
本机访问地址: http://localhost:5000
局域网访问地址: http://192.168.1.100:5000
```

### 2. 访问实验页面
- **本机访问**: 在浏览器中打开 `http://localhost:5000`
- **局域网访问**: 其他人可以通过 `http://你的IP:5000` 访问

### 3. 开始实验
1. 在浏览器中打开实验页面
2. 等待实验开始
3. 观察视频和聆听音频
4. 做出选择（点击按钮或使用键盘）
5. 实验自动进行，无需手动操作

## 键盘快捷键

- `←` 或 `A` - 选择左
- `→` 或 `D` - 选择右

## 实验流程

1. **准备阶段**: 显示实验说明，等待开始
2. **试验阶段**: 
   - 播放视频
   - 5秒静默
   - 10秒音频播放
   - 2秒静默
   - 显示选择界面
3. **数据收集**: 自动记录选择结果和反应时间
4. **自动进行**: 重复上述流程，直到所有试验完成

## 数据保存

实验数据会自动保存到 `experiment_output2` 目录：

- `experiment_results2.json` - JSON格式的完整数据
- `experiment_results.csv` - CSV格式，便于Excel打开
- `experiment_results.txt` - 文本格式的报表
- `audio_order.json` - 音频播放顺序信息

## 配置说明

在 `stilumi_web.py` 中修改以下路径：

```python
# 音频文件目录
AUDIO_DIR = r"D:\D\research\audioset下载\随机选择配对音频\new_audio"

# 视频文件路径
VIDEO_PATH = r"D:\D\research\audioset下载\RF.mp4"

# 输出目录
OUTPUT_DIR = r"D:\D\research\audioset下载\experiment_output2"
```

## 技术架构

- **后端**: Flask + Flask-SocketIO
- **前端**: HTML5 + JavaScript + CSS3
- **实时通信**: WebSocket
- **音频处理**: HTML5 Audio API
- **视频播放**: HTML5 Video API

## 故障排除

### 1. 端口被占用
如果5000端口被占用，修改 `stilumi_web.py` 中的端口号：
```python
socketio.run(app, host='0.0.0.0', port=5001, debug=True, allow_unsafe_werkzeug=True)
```

### 2. 音频文件无法播放
- 确保音频文件格式支持（WAV、MP3、OGG）
- 检查文件路径是否正确
- 确保音频文件没有损坏

### 3. 视频无法播放
- 确保视频文件格式支持（MP4推荐）
- 检查视频文件路径
- 确保浏览器支持视频格式

### 4. 网络访问问题
- 检查防火墙设置
- 确保端口未被阻止
- 检查IP地址是否正确

## 安全注意事项

- 仅在可信网络环境中使用
- 不要在生产环境中使用默认密钥
- 定期备份实验数据
- 注意保护参与者隐私

## 支持

如有问题，请检查：
1. 依赖包是否正确安装
2. 文件路径是否正确
3. 端口是否被占用
4. 防火墙设置是否正确

## 更新日志

### v1.0.0
- 初始版本发布
- 支持Web界面实验
- 自动数据收集和保存
- 实时同步功能
