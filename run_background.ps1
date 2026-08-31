# 防止电脑休眠的脚本
Write-Host "开始运行特征提取，防止电脑休眠..."

# 创建一个后台作业
$job = Start-Job -ScriptBlock {
    Set-Location "D:\D\research\audioset下载\download_audioset-master\download_audioset-master"
    python extract_all_categories_features.py
}

Write-Host "作业已启动，ID: $($job.Id)"
Write-Host "使用 'Get-Job' 查看作业状态"
Write-Host "使用 'Receive-Job -Id $($job.Id)' 查看输出"
Write-Host "使用 'Stop-Job -Id $($job.Id)' 停止作业"

# 保持脚本运行，防止休眠
while ($job.State -eq "Running") {
    Start-Sleep -Seconds 60
    Write-Host "作业仍在运行... $(Get-Date)"
}

Write-Host "作业完成！"
Receive-Job -Id $job.Id
Remove-Job -Id $job.Id

