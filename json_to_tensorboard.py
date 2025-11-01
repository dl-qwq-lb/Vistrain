"""
将 MMEngine 的 JSON 日志转换为 TensorBoard 格式
使用方法: python json_to_tensorboard.py
"""
import json
from pathlib import Path
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    try:
        from tensorboardX import SummaryWriter
    except ImportError:
        print("错误: 请安装 tensorboard 或 tensorboardX")
        print("运行: pip install tensorboard")
        exit(1)

def convert_json_to_tensorboard(json_file, output_dir):
    """将 scalars.json 转换为 TensorBoard events 文件"""
    json_path = Path(json_file)
    if not json_path.exists():
        print(f"错误: 找不到文件 {json_path}")
        return
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 创建 TensorBoard writer
    writer = SummaryWriter(str(output_path))
    
    print(f"正在读取 {json_path}...")
    line_count = 0
    scalar_count = 0
    skipped_keys = set()
    
    with open(json_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                step = data.get('step', data.get('iter', 0))
                
                # 记录所有标量值
                for key, value in data.items():
                    if key in ['step', 'iter', 'epoch']:
                        continue
                    if isinstance(value, (int, float)):
                        # 保持标签名称不变（TensorBoard 支持斜杠作为命名空间分隔符）
                        writer.add_scalar(key, value, step)
                        scalar_count += 1
                    else:
                        skipped_keys.add(key)
                
                line_count += 1
                
            except json.JSONDecodeError as e:
                print(f"警告: 跳过无效的 JSON 行: {e}")
                continue
    
    writer.flush()  # 确保数据写入磁盘
    writer.close()
    
    print(f"\n转换统计:")
    print(f"  - 处理行数: {line_count}")
    print(f"  - 记录标量数: {scalar_count}")
    if skipped_keys:
        print(f"  - 跳过的键（非数值）: {skipped_keys}")
    print(f"\n转换完成! TensorBoard 文件已保存到: {output_path}")
    print(f"现在可以运行: tensorboard --logdir {output_path}")

if __name__ == '__main__':
    # 自动查找最新的训练目录
    work_dir = Path('work_dirs/rtmdet_tiny_visdrone')
    if not work_dir.exists():
        print(f"错误: 找不到工作目录 {work_dir}")
        exit(1)
    
    # 查找所有包含 vis_data 的目录
    vis_data_dirs = list(work_dir.glob('*/vis_data/scalars.json'))
    if not vis_data_dirs:
        print(f"错误: 在 {work_dir} 中找不到 scalars.json 文件")
        exit(1)
    
    # 使用最新的文件
    latest_file = max(vis_data_dirs, key=lambda p: p.stat().st_mtime)
    vis_data_dir = latest_file.parent
    
    print(f"找到最新的训练数据: {latest_file}")
    print(f"正在转换...")
    
    # 在同一目录下创建 tensorboard 子目录
    tb_dir = vis_data_dir / 'tensorboard'
    convert_json_to_tensorboard(latest_file, tb_dir)
    
    print("\n" + "="*60)
    print("转换完成! 请运行以下命令启动 TensorBoard:")
    print(f"tensorboard --logdir {tb_dir}")
    print("="*60)

