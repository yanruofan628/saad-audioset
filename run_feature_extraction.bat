@echo off
echo 开始运行特征提取，防止电脑休眠...
echo 按 Ctrl+C 可以停止程序

REM 防止电脑休眠
powercfg -change -standby-timeout-ac 0
powercfg -change -standby-timeout-dc 0
powercfg -change -hibernate-timeout-ac 0
powercfg -change -hibernate-timeout-dc 0

REM 运行Python脚本
python extract_all_categories_features.py

REM 恢复休眠设置
echo 恢复休眠设置...
powercfg -change -standby-timeout-ac 20
powercfg -change -standby-timeout-dc 10
powercfg -change -hibernate-timeout-ac 0
powercfg -change -hibernate-timeout-dc 0

echo 程序完成！
pause

