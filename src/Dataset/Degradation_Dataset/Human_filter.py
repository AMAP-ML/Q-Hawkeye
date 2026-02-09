import json
import base64
import os
import argparse
from pathlib import Path
from typing import Optional

def image_to_base64(image_path):
    """将图片转换为base64编码"""
    try:
        with open(image_path, "rb") as img_file:
            encoded = base64.b64encode(img_file.read()).decode('utf-8')
            # 判断图片类型
            ext = os.path.splitext(image_path)[1].lower()
            mime_type = "image/jpeg" if ext in ['.jpg', '.jpeg'] else "image/png"
            return f"data:{mime_type};base64,{encoded}"
    except Exception as e:
        print(f"警告: 无法读取图片 {image_path}: {e}")
        return None

def process_json_data(json_path: str, max_samples: Optional[int] = None):
    """处理JSON数据并编码图片"""
    print(f"正在读取JSON文件: {json_path}")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    if max_samples:
        data = data[:max_samples]
        print(f"限制处理前 {max_samples} 个样本")
    
    print(f"开始处理 {len(data)} 个样本...")
    
    processed_data = []
    for idx, item in enumerate(data):
        print(f"处理样本 {idx+1}/{len(data)}: {item.get('id', 'unknown')}")
        
        # 创建新的数据项
        processed_item = {
            'id': item.get('id', ''),
            'gt_score': item.get('gt_score', 0),
            'degradation_type': item.get('degradation_type', ''),
            'std': item.get('std', 0),
            'formatted_score': item.get('formatted_score', ''),
            'degraded_image': None,
            'original_image': None,
            'degraded_path': '',
            'original_path': ''
        }
        
        # 编码degraded图片
        if item.get('images') and item['images'][0]:
            processed_item['degraded_path'] = item['images'][0]
            processed_item['degraded_image'] = image_to_base64(item['images'][0])
        
        # 编码original图片
        if item.get('original_images') and item['original_images'][0]:
            processed_item['original_path'] = item['original_images'][0]
            processed_item['original_image'] = image_to_base64(item['original_images'][0])
        
        processed_data.append(processed_item)
    
    return processed_data

def generate_html(data):
    """生成包含所有数据的HTML文件"""
    
    html_template = '''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>图像对比工具 - 独立版</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1600px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        
        .header {
            text-align: center;
            margin-bottom: 30px;
            padding-bottom: 20px;
            border-bottom: 2px solid #e0e0e0;
        }
        
        .header h1 {
            color: #333;
            margin-bottom: 10px;
            font-size: 2.5em;
        }
        
        .header p {
            color: #666;
            font-size: 1.1em;
        }
        
        .progress-section {
            margin-bottom: 25px;
        }
        
        .progress {
            background: #e0e0e0;
            height: 35px;
            border-radius: 20px;
            overflow: hidden;
            box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .progress-bar {
            background: linear-gradient(90deg, #4CAF50, #8BC34A);
            height: 100%;
            transition: width 0.5s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
            font-size: 14px;
        }
        
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 20px;
            margin-bottom: 25px;
        }
        
        .stat-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            color: white;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        }
        
        .stat-value {
            font-size: 2.5em;
            font-weight: bold;
            margin-bottom: 5px;
        }
        
        .stat-label {
            font-size: 0.9em;
            opacity: 0.9;
        }
        
        .info-panel {
            background: #f5f7fa;
            padding: 20px;
            border-radius: 12px;
            margin-bottom: 25px;
            border-left: 4px solid #667eea;
        }
        
        .info-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 15px;
        }
        
        .info-item {
            display: flex;
            align-items: center;
        }
        
        .info-label {
            font-weight: bold;
            color: #555;
            margin-right: 10px;
            min-width: 100px;
        }
        
        .info-value {
            color: #333;
            background: white;
            padding: 5px 10px;
            border-radius: 5px;
            flex: 1;
            font-family: 'Courier New', monospace;
            font-size: 13px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        
        .images-section {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .image-card {
            background: white;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        
        .image-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            text-align: center;
            font-weight: bold;
            font-size: 1.1em;
        }
        
        .image-container {
            padding: 20px;
            background: #f8f9fa;
            min-height: 400px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .image-container img {
            max-width: 100%;
            max-height: 450px;
            border-radius: 8px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            display: block;
        }
        
        .no-image {
            color: #999;
            font-size: 1.2em;
            text-align: center;
        }
        
        .controls {
            display: flex;
            gap: 20px;
            justify-content: center;
            margin-top: 30px;
        }
        
        button {
            padding: 15px 40px;
            font-size: 18px;
            border: none;
            border-radius: 30px;
            cursor: pointer;
            transition: all 0.3s;
            font-weight: bold;
            text-transform: uppercase;
            letter-spacing: 1px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }
        
        button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        
        .btn-yes {
            background: linear-gradient(45deg, #4CAF50, #8BC34A);
            color: white;
        }
        
        .btn-yes:hover:not(:disabled) {
            transform: translateY(-3px);
            box-shadow: 0 6px 20px rgba(76, 175, 80, 0.4);
        }
        
        .btn-no {
            background: linear-gradient(45deg, #f44336, #FF6B6B);
            color: white;
        }
        
        .btn-no:hover:not(:disabled) {
            transform: translateY(-3px);
            box-shadow: 0 6px 20px rgba(244, 67, 54, 0.4);
        }
        
        .btn-download {
            background: linear-gradient(45deg, #FF6B6B, #FFD93D);
            color: white;
        }
        
        .btn-download:hover {
            transform: translateY(-3px);
            box-shadow: 0 6px 20px rgba(255, 107, 107, 0.4);
        }
        
        .btn-nav {
            background: linear-gradient(45deg, #2196F3, #21CBF3);
            color: white;
            padding: 10px 20px;
            font-size: 14px;
        }
        
        .btn-nav:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(33, 150, 243, 0.4);
        }
        
        .modal {
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0,0,0,0.7);
        }
        
        .modal-content {
            background: white;
            margin: 10% auto;
            padding: 40px;
            width: 400px;
            border-radius: 20px;
            text-align: center;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        
        .modal-content h3 {
            color: #333;
            margin-bottom: 20px;
            font-size: 1.8em;
        }
        
        .modal-content p {
            color: #666;
            margin-bottom: 15px;
            font-size: 1.1em;
        }
        
        .shortcuts {
            background: #fff3cd;
            padding: 15px;
            border-radius: 8px;
            margin-top: 20px;
            text-align: center;
            border: 2px solid #ffc107;
        }
        
        .shortcuts-title {
            font-weight: bold;
            margin-bottom: 10px;
            color: #856404;
        }
        
        .shortcut-item {
            display: inline-block;
            margin: 0 15px;
            color: #856404;
        }
        
        kbd {
            background: #333;
            color: white;
            padding: 3px 8px;
            border-radius: 4px;
            font-family: monospace;
            font-size: 14px;
        }
        
        .loading-overlay {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0,0,0,0.7);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 2000;
        }
        
        .loading-content {
            background: white;
            padding: 30px;
            border-radius: 10px;
            text-align: center;
        }
        
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="loading-overlay" id="loadingOverlay">
        <div class="loading-content">
            <div class="spinner"></div>
            <p>正在加载数据...</p>
        </div>
    </div>
    
    <div class="container">
        <div class="header">
            <h1>🖼️ 图像对比工具 - 独立版</h1>
            <p>所有数据已嵌入，无需外部文件</p>
            <p style="margin-top: 10px; color: #999; font-size: 0.9em;">
                共 <span id="totalSamples">0</span> 个样本
            </p>
        </div>
        
        <div id="mainContent">
            <div class="progress-section">
                <div class="progress">
                    <div class="progress-bar" id="progressBar">0%</div>
                </div>
            </div>
            
            <div class="stats">
                <div class="stat-card">
                    <div class="stat-value" id="currentIndex">0</div>
                    <div class="stat-label">当前样本</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="totalCount">0</div>
                    <div class="stat-label">总样本数</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="keptCount">0</div>
                    <div class="stat-label">已保留</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="removedCount">0</div>
                    <div class="stat-label">已删除</div>
                </div>
            </div>
            
            <div class="info-panel">
                <div class="info-grid">
                    <div class="info-item">
                        <span class="info-label">样本ID:</span>
                        <span class="info-value" id="sampleId">-</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">GT分数:</span>
                        <span class="info-value" id="gtScore">-</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">降质类型:</span>
                        <span class="info-value" id="degradationType">-</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">标准差:</span>
                        <span class="info-value" id="stdDev">-</span>
                    </div>
                </div>
            </div>
            
            <div class="images-section">
                <div class="image-card">
                    <div class="image-header">🔍 Degraded Image</div>
                    <div class="image-container" id="degradedContainer">
                        <div class="no-image">加载中...</div>
                    </div>
                </div>
                <div class="image-card">
                    <div class="image-header">🎯 Original Image</div>
                    <div class="image-container" id="originalContainer">
                        <div class="no-image">加载中...</div>
                    </div>
                </div>
            </div>
            
            <div class="controls">
                <button class="btn-nav" onclick="previousSample()" id="btnPrev">
                    ⬅ 上一个
                </button>
                <button class="btn-yes" onclick="handleYes()">
                    ✅ 相同 (Y)
                </button>
                <button class="btn-no" onclick="handleNo()">
                    ❌ 不同 (N)
                </button>
                <button class="btn-nav" onclick="nextSample()" id="btnNext">
                    下一个 ➡
                </button>
            </div>
            
            <div class="controls" style="margin-top: 20px;">
                <button class="btn-download" onclick="downloadJSON()">
                    💾 下载处理后的JSON
                </button>
                <button class="btn-download" onclick="exportResults()">
                    📊 导出结果统计
                </button>
            </div>
            
            <div class="shortcuts">
                <div class="shortcuts-title">⌨️ 键盘快捷键</div>
                <span class="shortcut-item"><kbd>Y</kbd> 相同(保留)</span>
                <span class="shortcut-item"><kbd>N</kbd> 不同(删除)</span>
                <span class="shortcut-item"><kbd>←</kbd> 上一个</span>
                <span class="shortcut-item"><kbd>→</kbd> 下一个</span>
                <span class="shortcut-item"><kbd>Ctrl+Z</kbd> 撤销</span>
            </div>
        </div>
    </div>
    
    <div id="modal" class="modal">
        <div class="modal-content">
            <h3>🎉 处理完成！</h3>
            <p>所有样本已处理完毕</p>
            <p>保留: <span id="finalKept"></span> 个样本</p>
            <p>删除: <span id="finalRemoved"></span> 个样本</p>
            <button class="btn-download" onclick="downloadJSON()">💾 下载处理后的JSON</button>
            <button class="btn-yes" onclick="closeModal()">关闭</button>
        </div>
    </div>

    <script>
        // 嵌入的数据
        const embeddedData = ''' + json.dumps(data, ensure_ascii=False) + ''';
        
        // 全局变量
        let jsonData = [];
        let originalData = [];
        let currentIndex = 0;
        let removedCount = 0;
        let keptCount = 0;
        let history = [];
        let decisions = {}; // 记录每个样本的决策
        
        // 初始化
        window.onload = function() {
            originalData = embeddedData;
            jsonData = [...originalData];
            
            document.getElementById('totalSamples').textContent = jsonData.length;
            document.getElementById('totalCount').textContent = jsonData.length;
            
            // 初始化决策记录
            jsonData.forEach(item => {
                decisions[item.id] = null; // null: 未处理, true: 保留, false: 删除
            });
            
            updateDisplay();
            
            // 隐藏加载层
            document.getElementById('loadingOverlay').style.display = 'none';
        };
        
        function updateDisplay() {
            if (currentIndex >= jsonData.length) {
                showCompletionModal();
                return;
            }
            
            const sample = jsonData[currentIndex];
            
            // 更新信息
            document.getElementById('sampleId').textContent = sample.id || 'N/A';
            document.getElementById('gtScore').textContent = sample.gt_score ? sample.gt_score.toFixed(4) : 'N/A';
            document.getElementById('degradationType').textContent = sample.degradation_type || 'N/A';
            document.getElementById('stdDev').textContent = sample.std ? sample.std.toFixed(4) : 'N/A';
            
            // 更新degraded图片
            const degradedContainer = document.getElementById('degradedContainer');
            if (sample.degraded_image) {
                degradedContainer.innerHTML = `<img src="${sample.degraded_image}" alt="Degraded Image">`;
            } else {
                degradedContainer.innerHTML = `
                    <div class="no-image">
                        ❌ 图片不可用<br>
                        <small>${sample.degraded_path}</small>
                    </div>`;
            }
            
            // 更新original图片
            const originalContainer = document.getElementById('originalContainer');
            if (sample.original_image) {
                originalContainer.innerHTML = `<img src="${sample.original_image}" alt="Original Image">`;
            } else {
                originalContainer.innerHTML = `
                    <div class="no-image">
                        ❌ 图片不可用<br>
                        <small>${sample.original_path}</small>
                    </div>`;
            }
            
            // 更新按钮状态
            document.getElementById('btnPrev').disabled = currentIndex === 0;
            document.getElementById('btnNext').disabled = currentIndex === jsonData.length - 1;
            
            updateProgress();
        }
        
        function updateProgress() {
            const total = originalData.length;
            const processed = keptCount + removedCount;
            const progress = total > 0 ? (processed / total * 100).toFixed(1) : 0;
            
            document.getElementById('progressBar').style.width = progress + '%';
            document.getElementById('progressBar').textContent = progress + '%';
            
            document.getElementById('currentIndex').textContent = currentIndex + 1;
            document.getElementById('removedCount').textContent = removedCount;
            document.getElementById('keptCount').textContent = keptCount;
        }
        
        function handleYes() {
            if (currentIndex >= jsonData.length) return;
            
            const sample = jsonData[currentIndex];
            
            // 如果之前已经处理过，更新计数
            if (decisions[sample.id] === false) {
                removedCount--;
            } else if (decisions[sample.id] === null) {
                keptCount++;
            }
            
            decisions[sample.id] = true;
            
            history.push({
                action: 'keep',
                index: currentIndex,
                sample: sample
            });
            
            currentIndex++;
            updateDisplay();
        }
        
        function handleNo() {
            if (currentIndex >= jsonData.length) return;
            
            const sample = jsonData[currentIndex];
            
            // 如果之前已经处理过，更新计数
            if (decisions[sample.id] === true) {
                keptCount--;
            } else if (decisions[sample.id] === null) {
                removedCount++;
            }
            
            decisions[sample.id] = false;
            
            history.push({
                action: 'remove',
                index: currentIndex,
                sample: sample
            });
            
            // 从数组中删除
            jsonData.splice(currentIndex, 1);
            
            updateDisplay();
        }
        
        function previousSample() {
            if (currentIndex > 0) {
                currentIndex--;
                updateDisplay();
            }
        }
        
        function nextSample() {
            if (currentIndex < jsonData.length - 1) {
                currentIndex++;
                updateDisplay();
            }
        }
        
        function undo() {
            if (history.length === 0) return;
            
            const lastAction = history.pop();
            if (lastAction.action === 'keep') {
                decisions[lastAction.sample.id] = null;
                keptCount--;
                currentIndex--;
            } else if (lastAction.action === 'remove') {
                decisions[lastAction.sample.id] = null;
                jsonData.splice(lastAction.index, 0, lastAction.sample);
                removedCount--;
                currentIndex = lastAction.index;
            }
            updateDisplay();
        }
        
        function downloadJSON() {
            // 创建只包含必要字段的简化数据
            const exportData = jsonData.map(item => {
                const original = originalData.find(o => o.id === item.id);
                return {
                    id: item.id,
                    messages: original.messages || [],
                    images: [item.degraded_path],
                    gt_score: item.gt_score,
                    formatted_score: item.formatted_score,
                    degradation_type: item.degradation_type,
                    original_images: [item.original_path]
                };
            });
            
            const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
            const filename = `filtered_koniq_${timestamp}.json`;
            
            const dataStr = JSON.stringify(exportData, null, 2);
            const dataBlob = new Blob([dataStr], { type: 'application/json' });
            const url = URL.createObjectURL(dataBlob);
            const link = document.createElement('a');
            link.href = url;
            link.download = filename;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            URL.revokeObjectURL(url);
        }
        
        function exportResults() {
            const results = {
                total_samples: originalData.length,
                kept_samples: keptCount,
                removed_samples: removedCount,
                kept_ids: [],
                removed_ids: []
            };
            
            for (let id in decisions) {
                if (decisions[id] === true) {
                    results.kept_ids.push(id);
                } else if (decisions[id] === false) {
                    results.removed_ids.push(id);
                }
            }
            
            const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
            const filename = `results_${timestamp}.json`;
            
            const dataStr = JSON.stringify(results, null, 2);
            const dataBlob = new Blob([dataStr], { type: 'application/json' });
            const url = URL.createObjectURL(dataBlob);
            const link = document.createElement('a');
            link.href = url;
            link.download = filename;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            URL.revokeObjectURL(url);
        }
        
        function showCompletionModal() {
            document.getElementById('finalKept').textContent = keptCount;
            document.getElementById('finalRemoved').textContent = removedCount;
            document.getElementById('modal').style.display = 'block';
        }
        
        function closeModal() {
            document.getElementById('modal').style.display = 'none';
        }
        
        // 键盘快捷键
        document.addEventListener('keydown', function(e) {
            if (jsonData.length > 0 && currentIndex < jsonData.length) {
                switch(e.key.toLowerCase()) {
                    case 'y':
                        if (!e.ctrlKey && !e.metaKey) {
                            handleYes();
                        }
                        break;
                    case 'n':
                        if (!e.ctrlKey && !e.metaKey) {
                            handleNo();
                        }
                        break;
                    case 'z':
                        if (e.ctrlKey || e.metaKey) {
                            e.preventDefault();
                            undo();
                        }
                        break;
                    case 'arrowleft':
                        e.preventDefault();
                        previousSample();
                        break;
                    case 'arrowright':
                        e.preventDefault();
                        nextSample();
                        break;
                }
            }
        });
        
        // 点击模态框外部关闭
        window.onclick = function(event) {
            const modal = document.getElementById('modal');
            if (event.target == modal) {
                closeModal();
            }
        };
    </script>
</body>
</html>
'''
    
    return html_template

# 主函数
def main():
    parser = argparse.ArgumentParser(description="Generate an offline HTML page for human filtering of image pairs.")
    parser.add_argument("--json_path", required=True, help="Input JSON containing images[0] and original_images[0].")
    parser.add_argument("--output_html", required=True, help="Output HTML path.")
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit number of samples included in the HTML (default: all).",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("图像对比工具 - HTML生成器")
    print("=" * 60)

    processed_data = process_json_data(args.json_path, args.max_samples)
    
    # 生成HTML
    print("\n生成HTML文件...")
    html_content = generate_html(processed_data)
    
    # 保存HTML文件
    with open(args.output_html, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    # 获取文件大小
    file_size = os.path.getsize(args.output_html) / (1024 * 1024)  # MB
    
    print(f"\n✅ HTML文件生成成功！")
    print(f"文件名: {args.output_html}")
    print(f"文件大小: {file_size:.2f} MB")
    print(f"样本数量: {len(processed_data)}")
    print(f"\n在VSCode中:")
    print(f"1. 打开 {args.output_html}")
    print(f"2. 右键选择 'Open with Live Server' 或 'Show Preview'")
    print(f"3. 开始处理图像对比任务")

if __name__ == "__main__":
    main()
