import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
import torch
from matplotlib.font_manager import FontProperties
import matplotlib.font_manager as fm

# 设置中文字体
def set_chinese_font():
    try:
        # 尝试使用项目根目录下的SimHei字体
        # 获取当前工作目录，然后回退到项目根目录
        current_dir = os.getcwd()
        # 尝试多个可能的路径
        possible_paths = [
            os.path.join(current_dir, 'simhei.ttf'),  # 当前目录
            os.path.join(os.path.dirname(current_dir), 'simhei.ttf'),  # 上一级目录
            '/share/home/ncu_418000230048/project/Email_filtering_detection/simhei.ttf',  # 绝对路径
            'simhei.ttf'  # 相对路径
        ]
        
        font_path = None
        for path in possible_paths:
            if os.path.exists(path):
                font_path = path
                break
        
        if font_path:
            # 添加字体文件
            fm.fontManager.addfont(font_path)
            # 设置全局字体
            import matplotlib as mpl
            mpl.rcParams['font.family'] = 'SimHei'
            # 同时返回字体属性对象，用于单独设置
            font = fm.FontProperties(fname=font_path)
            return font
        else:
            print("警告：找不到SimHei字体文件，将使用系统默认字体")
            # 尝试使用系统中可能存在的中文字体
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
            return None
    except Exception as e:
        print(f"设置中文字体时出错：{e}")
        return None

def plot_training_history(train_losses, val_losses, train_parent_accs, val_parent_accs, 
                          train_child_accs, val_child_accs, save_path=None):
    """
    绘制训练历史
    
    Args:
        train_losses: 训练损失列表
        val_losses: 验证损失列表
        train_parent_accs: 训练父类准确率列表
        val_parent_accs: 验证父类准确率列表
        train_child_accs: 训练子类准确率列表
        val_child_accs: 验证子类准确率列表
        save_path: 保存路径
    """
    font = set_chinese_font()
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # 绘制损失曲线
    axes[0].plot(train_losses, label='训练损失', marker='o')
    axes[0].plot(val_losses, label='验证损失', marker='s')
    
    # 设置字体
    if font is not None:
        axes[0].set_xlabel('Epoch', fontproperties=font)
        axes[0].set_ylabel('损失', fontproperties=font)
        axes[0].set_title('训练和验证损失', fontproperties=font)
        axes[0].legend(prop=font)
    else:
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('损失')
        axes[0].set_title('训练和验证损失')
        axes[0].legend()
        
    axes[0].grid(True)
    
    # 绘制准确率曲线
    axes[1].plot(train_parent_accs, label='训练父类准确率', marker='o')
    axes[1].plot(val_parent_accs, label='验证父类准确率', marker='s')
    axes[1].plot(train_child_accs, label='训练子类准确率', marker='^')
    axes[1].plot(val_child_accs, label='验证子类准确率', marker='d')
    
    # 设置字体
    if font is not None:
        axes[1].set_xlabel('Epoch', fontproperties=font)
        axes[1].set_ylabel('准确率', fontproperties=font)
        axes[1].set_title('训练和验证准确率', fontproperties=font)
        axes[1].legend(prop=font)
    else:
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('准确率')
        axes[1].set_title('训练和验证准确率')
        axes[1].legend()
        
    axes[1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    plt.close()  # 关闭图形，避免显示


def plot_confusion_matrix(y_true, y_pred, class_names, title='混淆矩阵', save_path=None):
    """
    绘制混淆矩阵
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        class_names: 类别名称
        title: 图表标题
        save_path: 保存路径
    """
    font = set_chinese_font()
    
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    
    # 处理可能的零行，避免除零错误
    row_sums = cm.sum(axis=1)
    # 将零行替换为1，避免除零
    row_sums[row_sums == 0] = 1
    cm_norm = cm.astype('float') / row_sums[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    
    # 使用seaborn绘制热力图
    ax = sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    
    # 设置标签字体
    if font is not None:
        plt.xlabel('预测标签', fontproperties=font)
        plt.ylabel('真实标签', fontproperties=font)
        plt.title(title, fontproperties=font)
        
        # 设置刻度标签字体
        plt.setp(ax.get_xticklabels(), fontproperties=font)
        plt.setp(ax.get_yticklabels(), fontproperties=font)
    else:
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.title(title)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    plt.close()  # 关闭图形，避免显示


def visualize_attention_weights(model, data_loader, device, num_samples=5, save_dir=None):
    """
    可视化注意力权重
    
    Args:
        model: 模型
        data_loader: 数据加载器
        device: 设备
        num_samples: 样本数量
        save_dir: 保存目录
    """
    font = set_chinese_font()
    
    model.eval()
    samples_visualized = 0
    
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    with torch.no_grad():
        for batch in data_loader:
            if samples_visualized >= num_samples:
                break
            
            # 将数据移动到设备
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            tfidf_features = batch['tfidf_features'].to(device)
            
            # 获取BERT注意力权重
            outputs = model.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                output_attentions=True
            )
            
            # 获取最后一层的注意力权重
            attentions = outputs.attentions[-1]  # [batch_size, num_heads, seq_len, seq_len]
            
            # 对每个样本可视化注意力
            for i in range(min(len(input_ids), num_samples - samples_visualized)):
                # 获取样本的注意力权重
                sample_attention = attentions[i].mean(dim=0).cpu().numpy()  # 平均所有注意力头
                
                # 绘制热力图
                plt.figure(figsize=(10, 8))
                sns.heatmap(sample_attention, cmap='viridis')
                
                # 设置标题和标签
                title = f"样本 {samples_visualized + 1} 的BERT注意力权重"
                if font is not None:
                    plt.title(title, fontproperties=font)
                    plt.xlabel('Token位置', fontproperties=font)
                    plt.ylabel('Token位置', fontproperties=font)
                else:
                    plt.title(title)
                    plt.xlabel('Token位置')
                    plt.ylabel('Token位置')
                
                if save_dir:
                    plt.savefig(os.path.join(save_dir, f'attention_sample_{samples_visualized + 1}.png'), 
                               bbox_inches='tight')
                
                plt.close()  # 关闭图形，避免显示
                samples_visualized += 1
                
                if samples_visualized >= num_samples:
                    break


def plot_hierarchical_classification_results(parent_true, parent_pred, child_true, child_pred, 
                                            parent_class_names, child_class_names, save_dir=None):
    """
    绘制层次分类结果
    
    Args:
        parent_true: 父类真实标签
        parent_pred: 父类预测标签
        child_true: 子类真实标签
        child_pred: 子类预测标签
        parent_class_names: 父类名称
        child_class_names: 子类名称
        save_dir: 保存目录
    """
    font = set_chinese_font()
    
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 绘制父类混淆矩阵
    plot_confusion_matrix(
        parent_true, parent_pred, parent_class_names,
        title='父类分类混淆矩阵',
        save_path=os.path.join(save_dir, 'parent_confusion_matrix.png') if save_dir else None
    )
    
    # 绘制子类混淆矩阵（仅对有效样本）
    valid_indices = np.where(child_true != -1)[0]
    if len(valid_indices) > 0:
        plot_confusion_matrix(
            child_true[valid_indices], child_pred[valid_indices], child_class_names,
            title='子类分类混淆矩阵',
            save_path=os.path.join(save_dir, 'child_confusion_matrix.png') if save_dir else None
        )
    
    # 绘制垃圾邮件下的子类分布
    # 获取垃圾邮件样本（父类索引为1）
    spam_samples = np.where(parent_true == 1)[0]
    if len(spam_samples) > 0:
        # 获取垃圾邮件中有效的子类样本
        valid_child_samples = np.intersect1d(spam_samples, valid_indices)
        if len(valid_child_samples) > 0:
            # 计算子类分布
            child_labels = child_true[valid_child_samples]
            child_preds = child_pred[valid_child_samples]
            
            # 绘制垃圾邮件下的子类混淆矩阵
            plot_confusion_matrix(
                child_labels, child_preds, child_class_names,
                title='垃圾邮件下的子类分类混淆矩阵',
                save_path=os.path.join(save_dir, 'spam_child_confusion_matrix.png') if save_dir else None
            ) 