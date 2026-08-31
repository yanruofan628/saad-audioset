# 增量保存和断点续传功能说明

## 新功能特性

### 1. 增量保存
- **每处理完一个类别就保存一次**
- **即使中途中断，已处理的类别数据不会丢失**
- **自动保存完整的pickle文件和元数据JSON文件**

### 2. 断点续传
- **自动检测已处理的类别**
- **跳过已完成的类别，继续处理未完成的**
- **支持重新运行，不会重复处理**

### 3. 进度显示
- **实时显示处理进度**
- **显示剩余类别数量**
- **保存状态反馈**

## 使用方法

### 基本使用（推荐）
```python
result = extract_all_categories_features(
    parent_folder=r"D:\D\research\audioset下载\classified_audio",
    output_path=r"D:\D\research\audioset下载\特征",
    resume=True,  # 启用断点续传
    skip_existing=True  # 跳过已存在的类别
)
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `resume` | `True` | 是否启用断点续传 |
| `skip_existing` | `True` | 是否跳过已存在的类别 |
| `max_files_per_category` | `None` | 每个类别最大处理文件数 |

### 使用场景

#### 1. 首次运行
```python
# 第一次运行，处理所有类别
result = extract_all_categories_features(
    parent_folder=r"D:\D\research\audioset下载\classified_audio",
    output_path=r"D:\D\research\audioset下载\特征"
)
```

#### 2. 中断后继续
```python
# 程序中断后，重新运行会自动继续
result = extract_all_categories_features(
    parent_folder=r"D:\D\research\audioset下载\classified_audio",
    output_path=r"D:\D\research\audioset下载\特征"
)
# 输出: "发现已存在的特征数据，包含 5 个已处理的类别"
# 输出: "跳过已处理的类别 (5 个): Category1, Category2, ..."
# 输出: "需要处理的类别 (3 个): Category6, Category7, Category8"
```

#### 3. 重新处理所有类别
```python
# 强制重新处理所有类别
result = extract_all_categories_features(
    parent_folder=r"D:\D\research\audioset下载\classified_audio",
    output_path=r"D:\D\research\audioset下载\特征",
    resume=False  # 禁用断点续传
)
```

#### 4. 重新处理特定类别
```python
# 重新处理已存在的类别
result = extract_all_categories_features(
    parent_folder=r"D:\D\research\audioset下载\classified_audio",
    output_path=r"D:\D\research\audioset下载\特征",
    resume=True,
    skip_existing=False  # 不跳过已存在的类别
)
```

## 输出示例

### 首次运行
```
=== 批量提取所有类别特征（支持断点续传）===
找到 8 个类别文件夹: Category1, Category2, Category3, Category4, Category5, Category6, Category7, Category8
需要处理的类别 (8 个): Category1, Category2, Category3, Category4, Category5, Category6, Category7, Category8

=== 处理类别 1/8: Category1 ===
找到 150 个音频文件
开始提取 Category1 的特征...
提取Category1特征: 100%|██████████| 150/150 [05:23<00:00,  2.15s/it]
Category1 处理完成: 成功 148 个，失败 2 个
✓ Category1 类别特征已保存
进度: 1/8 类别完成，剩余 7 个类别

=== 处理类别 2/8: Category2 ===
...
```

### 中断后继续
```
=== 批量提取所有类别特征（支持断点续传）===
发现已存在的特征数据，包含 3 个已处理的类别
跳过已处理的类别 (3 个): Category1, Category2, Category3
需要处理的类别 (5 个): Category4, Category5, Category6, Category7, Category8

=== 处理类别 1/5: Category4 ===
找到 120 个音频文件
开始提取 Category4 的特征...
...
```

## 文件结构

### 输出文件
```
D:\D\research\audioset下载\特征\
├── all_categories_features.pkl          # 完整特征数据（增量更新）
├── all_categories_metadata.json         # 元数据（增量更新）
├── Category1_features_summary.json      # 类别1特征摘要
├── Category2_features_summary.json      # 类别2特征摘要
└── ...
```

### 数据完整性
- **每次保存都包含所有已处理的类别**
- **即使中途中断，已处理的数据完全可用**
- **重新运行会自动检测并继续处理**

## 优势

### 1. 可靠性
- **不会因为中断而丢失数据**
- **每个类别处理完立即保存**
- **支持随时中断和继续**

### 2. 效率
- **避免重复处理已完成的类别**
- **可以分批处理大量数据**
- **支持测试模式（限制文件数量）**

### 3. 灵活性
- **可以选择重新处理特定类别**
- **支持不同的处理策略**
- **便于调试和优化**

## 注意事项

1. **存储空间**：每次保存都会更新完整文件，确保有足够存储空间
2. **文件权限**：确保输出目录有写入权限
3. **数据一致性**：建议不要手动修改输出文件
4. **备份**：重要数据建议定期备份

## 故障排除

### 问题1：保存失败
```
✗ Category1 类别特征已保存失败: Permission denied
```
**解决方案**：检查输出目录权限，确保有写入权限

### 问题2：加载失败
```
加载已存在数据失败: UnpicklingError
```
**解决方案**：删除损坏的pickle文件，重新开始处理

### 问题3：内存不足
```
MemoryError: Unable to allocate array
```
**解决方案**：减少 `max_files_per_category` 参数，分批处理

