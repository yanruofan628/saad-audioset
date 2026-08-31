import os
import random
import time
import json
import csv
from datetime import datetime
import pygame
import cv2
import numpy as np
from flask import Flask, render_template_string, request, jsonify
import threading

# 初始化pygame
pygame.init()
pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)

# 初始化Flask应用
app = Flask(__name__)

# 配置参数
AUDIO_DIR = r"D:\D\research\audioset下载\音频配对合成_新版\分组音频\folder2"
VIDEO_PATH = r"D:\BaiduNetdiskDownload\40.第 四十 回 庆中秋好梦一日游 历磨难客栈重聚首.mp4"
OUTPUT_DIR = r"D:\D\research\audioset下载\experiment_output2"
RESULTS_FILE = os.path.join(OUTPUT_DIR, "experiment_results2.json")

# 创建输出目录
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# 视频位置文件
VIDEO_POSITIONS_FILE = os.path.join(OUTPUT_DIR, "video_positions.json")

# 加载视频位置数据
def load_video_positions():
    """从文件加载视频位置数据"""
    if os.path.exists(VIDEO_POSITIONS_FILE):
        try:
            with open(VIDEO_POSITIONS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"加载视频位置文件失败: {e}")
    return {}

def save_video_positions():
    """保存视频位置数据到文件"""
    try:
        with open(VIDEO_POSITIONS_FILE, 'w', encoding='utf-8') as f:
            json.dump(experiment_data['video_position'], f, indent=2)
    except Exception as e:
        print(f"保存视频位置文件失败: {e}")

# 全局变量
experiment_data = {
    "trials": [],
    "start_time": None,
    "total_duration": 0,
    "current_trial": 0,
    "is_running": False,
    "participants": {},
    "trial_completed": {},
    "username": None,
    "video_position": load_video_positions()  # 保存每个用户的视频播放位置
}

audio_files = []
audio_order = []
original_order = []

# 获取所有音频文件
try:
    if not os.path.exists(AUDIO_DIR):
        print(f"错误: 音频目录不存在: {AUDIO_DIR}")
        exit(1)
    
    audio_files = [f for f in os.listdir(AUDIO_DIR) if f.endswith(('.wav', '.mp3', '.ogg'))]
    if len(audio_files) == 0:
        print(f"错误: 在目录 {AUDIO_DIR} 中没有找到音频文件")
        exit(1)
    
    print(f"找到 {len(audio_files)} 个音频文件")
except Exception as e:
    print(f"错误: 无法读取音频目录: {e}")
    exit(1)

# 设置随机种子
random_seed = int(time.time() * 1000) % 2**32
random.seed(random_seed)
print(f"使用随机种子: {random_seed}")

# 随机打乱音频顺序
audio_order = list(range(len(audio_files)))
random.shuffle(audio_order)
original_order = audio_files.copy()

# 保存顺序信息
order_info = {
    "random_seed": random_seed,
    "shuffled_indices": audio_order,
    "original_files": original_order,
    "timestamp": datetime.now().isoformat()
}

with open(os.path.join(OUTPUT_DIR, "audio_order.json"), "w") as f:
    json.dump(order_info, f, indent=4)

print("音频顺序已保存到 audio_order.json")

# 实验页面
@app.route('/')
def index():
    return render_template_string('''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>音频实验</title>
    <style>
        body {
            font-family: 'Microsoft YaHei', Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }
        .container {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            text-align: center;
            max-width: 800px;
            width: 90%;
        }
        h1 { font-size: 2.5em; margin-bottom: 20px; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3); }
        .status {
            font-size: 1.2em;
            margin: 20px 0;
            padding: 15px;
            background: rgba(255, 255, 255, 0.2);
            border-radius: 10px;
            border-left: 4px solid #4CAF50;
        }
        .progress {
            margin: 20px 0;
            background: rgba(255, 255, 255, 0.2);
            border-radius: 10px;
            padding: 15px;
        }
        .progress-bar {
            width: 100%;
            height: 20px;
            background: rgba(255, 255, 255, 0.3);
            border-radius: 10px;
            overflow: hidden;
            margin: 10px 0;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #4CAF50, #8BC34A);
            width: 0%;
            transition: width 0.3s ease;
            border-radius: 10px;
        }
        .choice-buttons {
            margin: 30px 0;
            display: flex;
            gap: 30px;
            justify-content: center;
            flex-wrap: wrap;
        }
        .choice-btn {
            padding: 20px 40px;
            font-size: 1.5em;
            font-weight: bold;
            border: none;
            border-radius: 15px;
            cursor: pointer;
            transition: all 0.3s ease;
            min-width: 150px;
            text-transform: uppercase;
            letter-spacing: 2px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        }
        .choice-btn:hover { transform: translateY(-3px); box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3); }
        .choice-btn:active { transform: translateY(0); }
        .left-btn { background: linear-gradient(135deg, #FF6B6B, #EE5A24); color: white; }
        .right-btn { background: linear-gradient(135deg, #4ECDC4, #44A08D); color: white; }
        .choice-btn:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }
        .instructions {
            margin: 20px 0;
            padding: 20px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            font-size: 1.1em;
            line-height: 1.6;
        }
        .timer {
            font-size: 1.5em;
            font-weight: bold;
            margin: 15px 0;
            color: #FFD700;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
        }
        .waiting { animation: pulse 2s infinite; }
        @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.5; } 100% { opacity: 1; } }
        .audio-info {
            margin: 20px 0;
            padding: 15px;
            background: rgba(255, 255, 255, 0.2);
            border-radius: 10px;
            font-size: 1.1em;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎵 音频实验 🎵</h1>
        
        <div class="instructions">
            <p><strong>实验说明：</strong></p>
            <p>1. 请仔细观察视频画面并聆听音频</p>
            <p>2. 音频播放后，请选择您认为更合适的选项</p>
            <p>3. 请尽快做出选择，反应时间会被记录</p>
            <p>4. 实验会自动进行，请耐心等待</p>
        </div>

        <!-- 用户名输入 -->
        <div id="usernameBlock" style="margin: 10px 0; padding: 15px; background: rgba(255,255,255,0.15); border-radius: 10px;">
            <label for="usernameInput">请输入您的用户名：</label>
            <input id="usernameInput" type="text" placeholder="用户名" style="padding: 8px 12px; border-radius: 6px; border: none; margin-left: 8px;">
            <button id="usernameConfirmBtn" onclick="confirmUsername()" class="choice-btn right-btn" style="padding: 8px 16px; font-size: 1em; margin-left: 10px;">确认</button>
            <div id="usernameTip" style="margin-top: 8px; font-size: 0.9em; opacity: 0.9;">确认后将用于命名CSV文件</div>
        </div>

        <div class="status" id="status" style="display: none;">等待实验开始...</div>
        
        <!-- 用户交互提示 -->
        <div id="userInteractionPrompt" style="display: block; background: #ff6b6b; color: white; padding: 15px; border-radius: 8px; margin: 20px 0; text-align: center;">
            <h3>⚠️ 请先点击页面任意位置来启用音频播放</h3>
            <p>现代浏览器要求用户交互后才能播放音频</p>
            <button onclick="enableAudio()" style="background: white; color: #ff6b6b; border: none; padding: 10px 20px; border-radius: 5px; font-size: 16px; cursor: pointer;">点击启用音频</button>
        </div>

        <div class="progress" id="progressContainer" style="display: none;">
            <div>实验进度</div>
            <div class="progress-bar">
                <div class="progress-fill" id="progressFill"></div>
            </div>
            <div id="progressText">0 / 0</div>
        </div>

        <div class="timer" id="timer" style="display: none;">
            <div id="timerText">准备中...</div>
        </div>

        <div class="video-container" id="videoContainer" style="display: none;">
            <video id="videoPlayer" controls muted loop style="width: 100%; max-width: 640px; margin: 20px auto; display: block;">
                您的浏览器不支持视频播放
            </video>
        </div>

        <div class="audio-info" id="audioInfo" style="display: none;">
            <div>当前音频: <span id="currentAudio">-</span></div>
            <audio id="audioPlayer" style="width: 100%; margin-top: 10px; display: none;">
                您的浏览器不支持音频播放
            </audio>
        </div>

        <div class="choice-buttons" id="choiceButtons" style="display: none;">
            <button class="choice-btn left-btn" id="leftBtn" onclick="makeChoice('left')">← 左</button>
            <button class="choice-btn right-btn" id="rightBtn" onclick="makeChoice('right')">右 →</button>
        </div>
    </div>

    <script>
        let currentTrial = 0;
        let totalTrials = 0;
        let trialStartTime = 0;
        let isWaitingForChoice = false;
        let participantId = 'participant_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
        let audioEnabled = false; // 标记音频是否已启用
        let lastTrialNumber = 0;  // 跟踪上一个试验编号，避免重复播放
        let audioPlayTimeout = null;  // 音频播放定时器
        let audioStopTimeout = null;  // 音频停止定时器
        let videoStartOffsetSeconds = 180; // 视频从3分钟（180秒）开始
        let videoStartSeeked = false; // 确保只在首次加载时定位到起始位置
        let username = '';

        function confirmUsername() {
            const input = document.getElementById('usernameInput');
            const val = (input.value || '').trim();
            if (!val) {
                alert('请先输入用户名');
                input.focus();
                return;
            }
            username = val;
            fetch('/set_username', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ username })
            }).then(r => r.json()).then(data => {
                console.log('用户名已设置', data);
                document.getElementById('usernameBlock').style.display = 'none';
                
                // 获取用户的视频播放位置
                loadVideoPosition();
            }).catch(e => {
                console.error('设置用户名失败', e);
            });
        }

        // 加载用户的视频播放位置
        function loadVideoPosition() {
            if (!username) return;
            
            fetch(`/get_video_position?username=${encodeURIComponent(username)}`)
                .then(response => response.json())
                .then(data => {
                    if (data.position) {
                        videoStartOffsetSeconds = data.position;
                        console.log(`加载用户 ${username} 的视频位置: ${videoStartOffsetSeconds}秒`);
                    }
                })
                .catch(error => {
                    console.error('获取视频位置失败:', error);
                });
        }
        
        // 保存用户的视频播放位置
        function saveVideoPosition() {
            if (!username) return;
            
            const video = document.getElementById('videoPlayer');
            if (video && video.currentTime) {
                fetch('/save_video_position', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        username: username,
                        position: video.currentTime
                    })
                }).then(response => response.json())
                .then(data => {
                    console.log(`保存用户 ${username} 的视频位置: ${video.currentTime.toFixed(2)}秒`);
                }).catch(error => {
                    console.error('保存视频位置失败:', error);
                });
            }
        }

        // 启用音频播放
        function enableAudio() {
            console.log('用户点击启用音频');
            audioEnabled = true;
            document.getElementById('userInteractionPrompt').style.display = 'none';
            
            // 尝试播放一个静音音频来解锁音频播放
            const audio = document.getElementById('audioPlayer');
            if (audio) {
                // 先清空音频源，避免播放已加载的音频文件
                const originalSrc = audio.src;
                audio.src = '';
                
                // 创建一个短暂的静音音频数据URL来解锁播放权限
                const silentAudioDataUrl = 'data:audio/wav;base64,UklGRigAAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YQAAAAA=';
                audio.src = silentAudioDataUrl;
                
                // 先设置为静音来解锁播放权限
                audio.muted = true;
                audio.play().then(() => {
                    console.log('音频播放已启用');
                    audio.pause();
                    // 恢复原始音频源（如果有的话）
                    if (originalSrc) {
                        audio.src = originalSrc;
                    } else {
                        audio.src = '';
                    }
                    // 立即恢复为有声音状态
                    audio.muted = false;
                    console.log('音频已设置为有声音模式');
                }).catch(e => {
                    console.log('音频启用失败:', e);
                    // 恢复原始音频源
                    if (originalSrc) {
                        audio.src = originalSrc;
                    } else {
                        audio.src = '';
                    }
                    // 即使失败也要确保音频不是静音状态
                    audio.muted = false;
                });
            }
        }

        // 定期检查实验状态
        setInterval(checkExperimentStatus, 1000);

        function checkExperimentStatus() {
            fetch('/status')
                .then(response => response.json())
                .then(data => {
                    updateStatus(data);
                })
                .catch(error => {
                    console.error('检查状态失败:', error);
                });
        }

        let videoStarted = false; // 标记视频是否已开始
        let videoPaused = false; // 标记视频是否被暂停

        function updateStatus(data) {
            const statusEl = document.getElementById('status');
            
            if (data.is_running) {
                // 实验开始时启动连续视频播放
                if (!videoStarted) {
                    startContinuousVideo();
                    videoStarted = true;
                }
                
                // 隐藏状态与进度，不对用户显示
                if (statusEl) statusEl.style.display = 'none';
                const progressContainer = document.getElementById('progressContainer');
                if (progressContainer) progressContainer.style.display = 'none';
                
        // 如果有当前试验信息，显示音频
        if (data.is_running && data.current_trial > 0 && data.current_trial <= data.total_trials && data.current_audio) {
            // 只有当试验编号变化时才重新开始
            if (data.current_trial !== lastTrialNumber) {
                lastTrialNumber = data.current_trial;
                console.log('开始新trial:', data.current_audio);
                startNewTrial(data.current_audio);
            }
        }
            } else {
                // 隐藏状态与进度
                if (statusEl) statusEl.style.display = 'none';
                const progressContainer = document.getElementById('progressContainer');
                if (progressContainer) progressContainer.style.display = 'none';
                document.getElementById('audioInfo').style.display = 'none';
                document.getElementById('choiceButtons').style.display = 'none';
                document.getElementById('timer').style.display = 'none';
                
                // 暂停视频但保持进度
                const video = document.getElementById('videoPlayer');
                if (video && !video.paused) {
                    video.pause();
                    videoPaused = true;
                }
            }
        }

        function startNewTrial(audioFile) {
            console.log('开始新试验:', audioFile);
            
            // 重置选择状态
            choiceSubmitted = false;
            
            // 清除之前的定时器
            if (audioPlayTimeout) {
                clearTimeout(audioPlayTimeout);
            }
            if (audioStopTimeout) {
                clearTimeout(audioStopTimeout);
            }
            
            // 停止当前音频
            const audio = document.getElementById('audioPlayer');
            audio.pause();
            audio.currentTime = 0;
            
            // 如果音频未启用，先设置为静音状态，避免意外播放
            if (!audioEnabled) {
                audio.muted = true;
                console.log('音频未启用，设置为静音状态');
            }
            
            // 显示试验信息
            document.getElementById('currentAudio').textContent = audioFile;
            document.getElementById('audioInfo').style.display = 'block';
            
            // 显示选择按钮（但先禁用）
            document.getElementById('choiceButtons').style.display = 'flex';
            // 隐藏倒计时UI，不对用户显示
            document.getElementById('leftBtn').disabled = true;
            document.getElementById('rightBtn').disabled = true;
            isWaitingForChoice = false; // 17秒内不能选择
            trialStartTime = Date.now();
            currentTrial = lastTrialNumber;
            
            // 立即开始计时器
            startTrialTimer();
            
            // 加载音频文件（但不立即播放）
            audio.src = `/audio/${audioFile}`;
            audio.load();
            
            // 确保音频不会自动播放
            audio.autoplay = false;
            // 如果音频未启用，保持静音状态
            if (!audioEnabled) {
                audio.muted = true;
                console.log('音频未启用，保持静音状态');
            } else {
                audio.muted = false; // 确保音频默认有声音
            }
            audio.pause();
            audio.currentTime = 0;
            
            // 固定时间安排：5秒静默 + 5秒音频 + 2秒静默 = 12秒
            console.log('设置音频播放定时器:', audioFile);
            
            // 5秒后开始播放音频
            audioPlayTimeout = setTimeout(() => {
                console.log('5秒到了，尝试播放音频:', audioFile);
                console.log('音频状态:', {
                    readyState: audio.readyState,
                    paused: audio.paused,
                    duration: audio.duration,
                    currentTime: audio.currentTime,
                    audioEnabled: audioEnabled,
                    muted: audio.muted
                });
                
                // 检查音频是否已启用
                if (!audioEnabled) {
                    console.log('音频未启用，跳过播放');
                    return;
                }
                
                // 双重检查：确保音频不是静音状态
                if (audio.muted) {
                    console.log('音频处于静音状态，跳过播放');
                    return;
                }
                
                // 确保音频从头开始播放
                audio.currentTime = 0;
                audio.play().then(() => {
                    console.log('音频播放成功:', audioFile);
                }).catch(e => {
                    console.log('音频播放失败:', e);
                });
            }, 5000);
            
            // 10秒后停止音频（5秒静默 + 5秒播放）
            audioStopTimeout = setTimeout(() => {
                console.log('10秒到了，停止音频:', audioFile);
                audio.pause();
                console.log('音频已停止');
            }, 10000);
        }
        
        function startContinuousVideo() {
            let video = document.getElementById('videoPlayer');
            let videoContainer = document.getElementById('videoContainer');
            
            if (video && videoContainer) {
                // 显示视频容器
                videoContainer.style.display = 'block';
                
                // 确保视频是静音的（浏览器自动播放要求）
                video.muted = true;
                video.loop = true; // 循环播放
                
                // 如果视频还没有加载，则设置源并加载
                if (!video.src) {
                    video.src = '/video';
                    video.load();
                    // 确保视频加载后保持静音
                    video.addEventListener('loadeddata', () => {
                        video.muted = true;
                    }, { once: true });
                }
                
                // 在元数据加载后，将播放位置跳到用户上次的位置
                if (!videoStartSeeked) {
                    const seekToUserPosition = () => {
                        try {
                            if (video.duration && video.duration > videoStartOffsetSeconds) {
                                video.currentTime = videoStartOffsetSeconds;
                                videoStartSeeked = true;
                                console.log(`视频已跳转到用户位置 ${videoStartOffsetSeconds} 秒`);
                            } else {
                                // 若时长未知，稍后重试
                                setTimeout(seekToUserPosition, 200);
                            }
                        } catch (e) {
                            console.log('设置视频位置失败，重试中:', e);
                            setTimeout(seekToUserPosition, 200);
                        }
                    };
                    
                    if (video.readyState >= 1) {
                        // 已有元数据
                        seekToUserPosition();
                    } else {
                        // 等待元数据加载
                        video.addEventListener('loadedmetadata', seekToUserPosition, { once: true });
                    }
                }
                
                // 定期保存视频位置（每10秒）
                const savePositionInterval = setInterval(() => {
                    saveVideoPosition();
                }, 10000);
                
                // 页面卸载时保存位置
                window.addEventListener('beforeunload', saveVideoPosition);
                
                // 结束时从设定起点重新播放
                const onVideoEnded = () => {
                    try {
                        video.currentTime = Math.min(videoStartOffsetSeconds, Math.max(0, (video.duration || 0) - 0.1));
                        // 确保视频保持静音状态
                        video.muted = true;
                    } catch (e) {}
                    const p = video.play(); if (p) p.catch(()=>{});
                };
                video.removeEventListener('ended', onVideoEnded); // 避免重复绑定
                video.addEventListener('ended', onVideoEnded);
                
                // 恢复视频播放（如果之前被暂停了）
                if (videoPaused || video.paused) {
                    // 确保视频保持静音状态
                    video.muted = true;
                    const playPromise = video.play();
                    if (playPromise !== undefined) {
                        playPromise.then(() => {
                            console.log('视频恢复连续播放（静音）');
                            videoPaused = false;
                        }).catch(e => {
                            console.log('视频播放失败:', e);
                        });
                    }
                }
            } else {
                console.log('视频元素未找到:', {video, videoContainer});
            }
        }

        function startTrialTimer() {
            const timerEl = document.getElementById('timer');
            const timerTextEl = document.getElementById('timerText');
            
            // 不显示倒计时UI（仅在结束时展示提示）
            if (timerEl) timerEl.style.display = 'none';
            
            // 固定时间安排：12秒总时间（5s静默 + 5s音频 + 2s静默）
            let timeLeft = 12;
            let currentPhase = '静默期';
            // 不更新可见文本
            
            console.log('开始trial计时器，总共12秒');
            
            const countdown = setInterval(() => {
                timeLeft--;
                
                // 根据时间确定当前阶段
                if (timeLeft > 7) {
                    currentPhase = '静默期'; // 前5秒
                } else if (timeLeft > 2) {
                    currentPhase = '音频播放'; // 中间5秒
                } else {
                    currentPhase = '静默期'; // 最后2秒
                }
                
                console.log(`计时器: ${timeLeft}秒 (${currentPhase})`);
                
                if (timeLeft <= 0) {
                    clearInterval(countdown);
                    // 仅在结束时显示提示文字
                    if (timerEl) timerEl.style.display = 'block';
                    if (timerTextEl) timerTextEl.textContent = '时间到！请做出选择';
                    console.log('12秒计时结束，启用选择按钮');
                    
                    // 启用选择按钮
                    document.getElementById('leftBtn').disabled = false;
                    document.getElementById('rightBtn').disabled = false;
                    isWaitingForChoice = true;
                }
            }, 1000);
        }

        let choiceSubmitted = false; // 防止重复提交

        function makeChoice(choice) {
            if (!isWaitingForChoice) {
                console.log('当前不在等待选择状态');
                return;
            }
            
            // 防止重复提交
            if (choiceSubmitted) {
                console.log('选择已提交，忽略重复点击');
                return;
            }
            
            choiceSubmitted = true;
            
            // 立即禁用按钮，防止重复点击
            document.getElementById('leftBtn').disabled = true;
            document.getElementById('rightBtn').disabled = true;
            
            const reactionTime = Date.now() - trialStartTime;
            
            const choiceData = {
                participant_id: participantId,
                trial_number: currentTrial,
                audio_file: document.getElementById('currentAudio').textContent,
                choice: choice,
                reaction_time: reactionTime
            };
            
            console.log('发送选择:', choiceData);
            
            // 发送选择到服务器
            fetch('/participant_choice', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(choiceData)
            })
            .then(response => response.json())
            .then(data => {
                console.log('选择已记录:', data);
                
                // 隐藏选择按钮，等待下一个试验
                document.getElementById('choiceButtons').style.display = 'none';
                isWaitingForChoice = false;
            })
            .catch(error => {
                console.error('发送选择失败:', error);
                // 如果发送失败，重新启用按钮
                choiceSubmitted = false;
                document.getElementById('leftBtn').disabled = false;
                document.getElementById('rightBtn').disabled = false;
            });
            
            console.log(`选择: ${choice}, 反应时间: ${reactionTime}ms`);
        }

        // 键盘快捷键
        document.addEventListener('keydown', function(event) {
            if (!isWaitingForChoice) return;
            
            if (event.key === 'ArrowLeft' || event.key === 'a' || event.key === 'A') {
                makeChoice('left');
                event.preventDefault();
            } else if (event.key === 'ArrowRight' || event.key === 'd' || event.key === 'D') {
                makeChoice('right');
                event.preventDefault();
            }
        });
    </script>
</body>
</html>
    ''')

# 管理控制台
@app.route('/admin')
def admin():
    return render_template_string('''
<!DOCTYPE html>
<html>
<head>
    <title>实验控制台</title>
    <style>
        body { font-family: Arial, sans-serif; background: #f5f5f5; margin: 0; padding: 20px; }
        .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .btn { padding: 15px 30px; font-size: 1.2em; margin: 10px; border: none; border-radius: 5px; cursor: pointer; transition: background 0.3s; }
        .btn-start { background: #4CAF50; color: white; }
        .btn-stop { background: #f44336; color: white; }
        .btn-reset { background: #ff9800; color: white; }
        .btn:hover { opacity: 0.8; }
        .status { padding: 15px; margin: 20px 0; border-radius: 5px; background: #e3f2fd; border-left: 4px solid #2196F3; }
        h1 { color: #333; text-align: center; }
        .info { background: #f9f9f9; padding: 15px; border-radius: 5px; margin: 20px 0; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎵 音频实验控制台 🎵</h1>
        
        <div class="info">
            <h3>实验信息</h3>
            <p><strong>音频文件数量:</strong> {{ audio_count }}个</p>
            <p><strong>视频文件:</strong> RF.mp4</p>
            <p><strong>输出目录:</strong> experiment_output2</p>
        </div>
        
        <div class="status" id="status">实验状态: 未开始</div>
        
        <div style="text-align: center;">
            <button class="btn btn-start" onclick="startExperiment()">开始实验</button>
            <button class="btn btn-stop" onclick="stopExperiment()">停止实验</button>
            <button class="btn btn-reset" onclick="resetExperiment()">重置实验</button>
        </div>
        
        <div class="info">
            <h3>访问地址</h3>
            <p><strong>被试者访问:</strong> <a href="http://localhost:5000" target="_blank">http://localhost:5000</a></p>
        </div>
    </div>
    
    <script>
        function startExperiment() {
            fetch('/start_experiment', {method: 'POST'})
                .then(response => response.json())
                .then(data => {
                    alert(data.message);
                    updateStatus('实验进行中...');
                })
                .catch(error => alert('启动失败: ' + error));
        }
        
        function stopExperiment() {
            fetch('/stop_experiment', {method: 'POST'})
                .then(response => response.json())
                .then(data => {
                    alert(data.message);
                    updateStatus('实验已停止');
                });
        }
        
        function resetExperiment() {
            if (confirm('确定要重置实验吗？这将清空所有数据！')) {
                fetch('/reset_experiment', {method: 'POST'})
                    .then(response => response.json())
                    .then(data => {
                        alert(data.message);
                        updateStatus('实验已重置');
                    });
            }
        }
        
        function updateStatus(status) {
            document.getElementById('status').innerHTML = '实验状态: ' + status;
        }
        
        // 定期更新状态
        setInterval(() => {
            fetch('/status')
                .then(response => response.json())
                .then(data => {
                    if (data.is_running) {
                        updateStatus(`实验进行中 - 当前试验: ${data.current_trial} / ${data.total_trials}`);
                    } else {
                        updateStatus('实验未开始');
                    }
                });
        }, 2000);
    </script>
</body>
</html>
    ''', audio_count=len(audio_files))

# API路由
@app.route('/status')
def get_status():
    # 获取当前试验的音频文件
    current_audio = ""
    if experiment_data['is_running'] and experiment_data['current_trial'] > 0:
        # 根据断点续做逻辑，需要从remaining_order中获取音频
        trial_offset = experiment_data.get('trial_offset', 0)
        current_trial_index = experiment_data['current_trial'] - trial_offset - 1
        
        # 获取remaining_order（在run_experiment中设置）
        remaining_order = experiment_data.get('remaining_order', audio_order)
        
        if 0 <= current_trial_index < len(remaining_order):
            audio_index = remaining_order[current_trial_index]
            current_audio = audio_files[audio_index]
    
    return jsonify({
        'is_running': experiment_data['is_running'],
        'current_trial': experiment_data['current_trial'],
        'total_trials': len(audio_order),
        'current_audio': current_audio
    })

@app.route('/audio/<filename>')
def serve_audio(filename):
    from flask import send_file
    audio_path = os.path.join(AUDIO_DIR, filename)
    if os.path.exists(audio_path):
        return send_file(audio_path)
    else:
        return "Audio file not found", 404

@app.route('/video')
def serve_video():
    from flask import send_file
    if os.path.exists(VIDEO_PATH):
        return send_file(VIDEO_PATH)
    else:
        return "Video file not found", 404

@app.route('/participant_choice', methods=['POST'])
def handle_participant_choice():
    data = request.json
    print(f'收到选择: {data}')
    
    trial_number = data['trial_number']
    
    # 防重复提交：检查该trial是否已经完成
    if experiment_data['trial_completed'].get(trial_number, False):
        print(f'Trial {trial_number} 已经完成，忽略重复提交')
        return jsonify({'message': '该试验已完成，忽略重复提交'})
    
    # 保存选择数据
    choice_data = {
        'participant_id': data.get('participant_id', 'unknown'),
        'trial_number': trial_number,
        'audio_file': data['audio_file'],
        'choice': data['choice'],
        'reaction_time': data['reaction_time'],
        'timestamp': datetime.now().isoformat()
    }
    
    # 添加到实验结果
    experiment_data['trials'].append(choice_data)
    
    # 标记当前trial完成
    experiment_data['trial_completed'][trial_number] = True
    print(f'Trial {trial_number} 标记为完成')
    
    # 追加写入到CSV（不中断累计）
    append_trial_to_csv(choice_data, experiment_data.get('username') or 'unknown')
    
    return jsonify({'message': '选择已记录'})

@app.route('/set_username', methods=['POST'])
def set_username():
    data = request.json or {}
    username = (data.get('username') or '').strip()
    if not username:
        return jsonify({'error': '用户名不能为空'}), 400
    experiment_data['username'] = username
    # 不在此处保存，避免覆盖旧数据；开始实验时会做断点续做并保存
    return jsonify({'message': '用户名已设置', 'username': username})

@app.route('/save_video_position', methods=['POST'])
def save_video_position():
    """保存用户的视频播放位置"""
    data = request.json or {}
    username = (data.get('username') or '').strip()
    position = float(data.get('position', 0))  # 视频播放位置（秒）
    
    if not username:
        return jsonify({'error': '用户名不能为空'}), 400
    
    # 保存视频位置
    experiment_data['video_position'][username] = position
    save_video_positions()  # 同时保存到文件
    print(f"保存用户 {username} 的视频位置: {position:.2f}秒")
    
    return jsonify({'message': '视频位置已保存', 'position': position})

@app.route('/get_video_position', methods=['GET'])
def get_video_position():
    """获取用户的视频播放位置"""
    username = request.args.get('username', '').strip()
    
    if not username:
        return jsonify({'error': '用户名不能为空'}), 400
    
    position = experiment_data['video_position'].get(username, 180)  # 默认从3分钟开始
    print(f"获取用户 {username} 的视频位置: {position:.2f}秒")
    
    return jsonify({'position': position})

@app.route('/start_experiment', methods=['POST'])
def start_experiment():
    if experiment_data['is_running']:
        return jsonify({'error': '实验已在运行中'}), 400
    
    # 在新线程中运行实验
    experiment_thread = threading.Thread(target=run_experiment)
    experiment_thread.daemon = True
    experiment_thread.start()
    
    return jsonify({'message': '实验已开始'})

@app.route('/stop_experiment', methods=['POST'])
def stop_experiment():
    experiment_data['is_running'] = False
    return jsonify({'message': '实验已停止'})

@app.route('/reset_experiment', methods=['POST'])
def reset_experiment():
    global experiment_data
    experiment_data = {
        "trials": [],
        "start_time": None,
        "total_duration": 0,
        "current_trial": 0,
        "is_running": False,
        "participants": {},
    "trial_completed": {},
    "username": None,
    "trial_offset": 0
    }
    return jsonify({'message': '实验已重置'})

# 保存结果函数
def _safe_username(name: str) -> str:
    name = (name or 'unknown').strip()
    return ''.join(c for c in name if c not in '\\/:*?"<>|').strip() or 'unknown'

def _get_csv_path_for_username(username: str) -> str:
    safe_username = _safe_username(username)
    return os.path.join(OUTPUT_DIR, f"{safe_username}.csv")

def _ensure_csv_header(csv_path: str):
    need_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    if need_header:
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['试验编号', '音频文件', '选择', '反应时间(ms)', '被试者ID', '时间戳'])

def append_trial_to_csv(trial: dict, username: str):
    csv_path = _get_csv_path_for_username(username or 'unknown')
    _ensure_csv_header(csv_path)
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            trial['trial_number'],
            trial['audio_file'],
            trial['choice'],
            f"{trial['reaction_time']:.2f}",
            trial['participant_id'],
            trial['timestamp']
        ])

# 实验主循环
def run_experiment():
    global experiment_data
    
    experiment_data['is_running'] = True
    experiment_data['current_trial'] = 0
    experiment_data['start_time'] = datetime.now().isoformat()
    experiment_data['trial_completed'] = {}  # 记录每个trial是否完成
    
    # 断点续做：若已设置用户名，则从对应CSV恢复历史并跳过已完成
    username = (experiment_data.get('username') or '').strip()
    completed_files = set()
    restored_trials = []
    if username:
        safe_username = ''.join(c for c in username if c not in '\\/:*?"<>|').strip() or 'unknown'
        csv_path = os.path.join(OUTPUT_DIR, f"{safe_username}.csv")
        print(f"检查用户 {username} 的CSV文件: {csv_path}")
        
        if os.path.exists(csv_path):
            try:
                with open(csv_path, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    header_skipped = False
                    for row in reader:
                        if not header_skipped:
                            header_skipped = True
                            continue
                        if not row:
                            continue
                        try:
                            trial_number = int(row[0]) if row[0] else 0
                            audio_file = row[1]
                            choice = row[2]
                            reaction_ms = float(row[3]) if row[3] else 0.0
                            participant_id = row[4]
                            timestamp = row[5]
                            restored_trials.append({
                                'trial_number': trial_number,
                                'audio_file': audio_file,
                                'choice': choice,
                                'reaction_time': reaction_ms,
                                'participant_id': participant_id,
                                'timestamp': timestamp
                            })
                            completed_files.add(audio_file)
                        except Exception:
                            continue
                print(f"恢复用户 {username} 已有记录 {len(restored_trials)} 条，跳过这些音频")
                print(f"用户 {username} 已完成音频文件: {sorted(completed_files)}")
            except Exception as e:
                print(f"恢复CSV失败: {e}")
        else:
            print(f"用户 {username} 的CSV文件不存在，从头开始")
    else:
        print("未设置用户名，从头开始实验")
    
    # 根据completed_files过滤本次待做的audio_order
    # 修复：正确比较音频文件名
    remaining_order = []
    for idx in audio_order:
        audio_file = audio_files[idx]
        if audio_file not in completed_files:
            remaining_order.append(idx)
    
    print(f"\n=== 断点续做状态 ===")
    print(f"用户名: {username}")
    print(f"原始音频文件数: {len(audio_files)}")
    print(f"用户已完成音频文件数: {len(completed_files)}")
    print(f"用户剩余待做音频文件数: {len(remaining_order)}")
    print(f"用户已完成试验数: {len(restored_trials)}")
    
    if len(completed_files) > 0:
        print(f"用户已完成音频文件示例: {sorted(list(completed_files))[:5]}")
    if len(remaining_order) > 0:
        remaining_files = [audio_files[i] for i in remaining_order[:5]]
        print(f"用户剩余音频文件示例: {remaining_files}")
    
    # 计算正确的试验偏移量
    # 如果用户已有试验，应该从最大试验编号+1开始
    if restored_trials:
        max_trial_number = max([t['trial_number'] for t in restored_trials])
        experiment_data['trial_offset'] = max_trial_number
    else:
        experiment_data['trial_offset'] = 0
    
    experiment_data['remaining_order'] = remaining_order  # 保存剩余音频顺序
    
    print(f"试验偏移量设置为: {experiment_data['trial_offset']}")
    
    # 将历史试验并入内存（用于覆盖写出）
    if restored_trials:
        experiment_data['trials'] = restored_trials.copy()
        for t in restored_trials:
            experiment_data['trial_completed'][t['trial_number']] = True
    
    print(f"开始Web实验，共 {len(remaining_order)} 个试验")
    if len(remaining_order) == 0:
        print("警告：没有剩余的试验需要完成！")
        print("可能的原因：")
        print("1. 所有音频文件都已完成")
        print("2. 断点续做逻辑有问题")
        experiment_data['is_running'] = False
        return
    
    # 简单循环：等待用户完成每个trial
    for i, audio_idx in enumerate(remaining_order):
        if not experiment_data['is_running']:
            break
        
        # 断点续做时，试验编号应该从已完成的最后一个试验编号+1开始
        trial_number = experiment_data.get('trial_offset', 0) + i + 1
        experiment_data['current_trial'] = trial_number
        experiment_data['trial_completed'][trial_number] = False
        
        # 获取当前音频文件
        current_audio = audio_files[audio_idx]
        
        print(f"开始试验 {trial_number}/{len(remaining_order)}: {current_audio}")
        print("等待用户进入页面并完成trial...")
        
        # 等待用户完成当前trial
        while not experiment_data['trial_completed'].get(trial_number, False) and experiment_data['is_running']:
            time.sleep(0.1)  # 短暂等待，避免占用过多CPU
        
        print(f"试验 {trial_number} 完成")
        
        # 选择后立即开始下一个trial，不需要等待
    
    # 实验结束
    experiment_data['is_running'] = False
    experiment_data['end_time'] = datetime.now().isoformat()
    
    print("实验完成！")
    print(f"结果保存在: {OUTPUT_DIR}")

if __name__ == "__main__":
    print("=" * 60)
    print("音频实验 Web 服务器 (简化版)")
    print("=" * 60)
    print(f"音频目录: {AUDIO_DIR}")
    print(f"视频文件: {VIDEO_PATH}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"找到 {len(audio_files)} 个音频文件")
    print("=" * 60)
    print("服务器启动中...")
    print("访问地址: http://localhost:5000")
    print("管理界面: http://localhost:5000/admin")
    print("=" * 60)
    print("开始启动Flask服务器...")
    
    try:
        app.run(host='0.0.0.0', port=5000, debug=True)
    except KeyboardInterrupt:
        print("\n服务器已停止")
    except Exception as e:
        print(f"服务器启动失败: {e}")
    finally:
        pygame.quit()
        print("程序结束")
