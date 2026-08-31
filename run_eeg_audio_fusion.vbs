' EEG音频融合分类 - 单人处理
' 双击此文件即可运行，无需打开命令行窗口

Dim shell
Set shell = CreateObject("WScript.Shell")

' 检查Python是否安装
Dim result
result = shell.Run("cmd /c python --version", 0, True)

If result <> 0 Then
    MsgBox "错误：未找到Python，请确保Python已安装并添加到PATH环境变量中。" & vbCrLf & vbCrLf & "您可以从 https://www.python.org 下载并安装Python", vbCritical, "Python未找到"
    WScript.Quit
End If

' 运行Python脚本
shell.Run "cmd /c python eeg_audio_fusion_individual.py & pause", 1, True