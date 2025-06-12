import os
import argparse
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random
import time
from tqdm import tqdm
import torch.nn.functional as F

# 导入自定义模块
from data.data_processor import EmailProcessor, create_data_loaders
from models.bert_cnn_model import BertCNNHierarchicalModel
from utils.train_utils import HierarchicalLoss, train_epoch, evaluate, predict
from utils.visualization import plot_training_history, plot_hierarchical_classification_results, plot_confusion_matrix
from utils.word_cloud import generate_word_clouds, plot_top_keywords
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def set_seed(seed):
    """设置随机种子以确保可重现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='训练BERT-CNN层次分类模型')
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, default='trec06c', help='数据集目录')
    parser.add_argument('--bert_model_path', type=str, default='bert-base-chinese', help='BERT模型路径')
    parser.add_argument('--output_dir', type=str, default='outputs', help='输出目录')
    parser.add_argument('--max_length', type=int, default=256, help='BERT输入最大长度')
    parser.add_argument('--tfidf_max_features', type=int, default=20000, help='TF-IDF最大特征数量')
    parser.add_argument('--use_augmentation', action='store_true', help='是否使用数据增强')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--lr', type=float, default=2e-5, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='权重衰减')
    parser.add_argument('--parent_weight', type=float, default=1.0, help='父类损失权重')
    parser.add_argument('--child_weight', type=float, default=1.0, help='子类损失权重')
    parser.add_argument('--routing_weight', type=float, default=0.5, help='路由损失权重')
    parser.add_argument('--patience', type=int, default=3, help='早停耐心值')
    parser.add_argument('--freeze_bert', action='store_true', help='是否冻结BERT')
    parser.add_argument('--unfreeze_epoch', type=int, default=3, help='解冻BERT的轮次')
    parser.add_argument('--unfreeze_layers', type=int, default=3, help='解冻BERT的层数')
    parser.add_argument('--malicious_weight', type=float, default=1.5, help='恶意邮件子类损失权重')
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--use_gpu', action='store_true', help='是否使用GPU')
    parser.add_argument('--log_interval', type=int, default=100, help='日志间隔')
    
    args = parser.parse_args()
    return args

def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建TensorBoard日志
    writer = SummaryWriter(os.path.join(args.output_dir, 'logs'))
    
    # 初始化数据处理器
    print("初始化数据处理器...")
    processor = EmailProcessor(
        data_dir=args.data_dir,
        bert_model_path=args.bert_model_path,
        max_length=args.max_length,
        tfidf_max_features=args.tfidf_max_features,
        use_augmentation=args.use_augmentation
    )
    
    # 加载数据
    print("加载数据...")
    df = processor.load_data()
    print(f"加载了 {len(df)} 条邮件数据")
    
    # 创建数据加载器
    print("创建数据加载器...")
    train_loader, val_loader, test_loader = create_data_loaders(
        dataframe=df,
        processor=processor,
        batch_size=args.batch_size
    )
    
    # 初始化模型
    print("初始化模型...")
    model = BertCNNHierarchicalModel(
        bert_model_path=args.bert_model_path,
        tfidf_dim=args.tfidf_max_features,
        freeze_bert=args.freeze_bert,
        malicious_feature_dim=14,  # 恶意邮件特征维度
        malicious_hidden_dim=32    # 恶意邮件特征隐藏层维度
    )
    model.to(device)
    
    # 初始化损失函数
    criterion = HierarchicalLoss(
        parent_weight=args.parent_weight,
        child_weight=args.child_weight,
        routing_weight=args.routing_weight
    )
    
    # 为恶意邮件子类设置更高的权重
    def weighted_loss_fn(outputs, targets):
        # 获取预测和标签
        parent_logits, child_logits, routed_child_logits = outputs
        parent_labels, child_labels = targets
        
        # 确保标签是适当的形状
        if isinstance(parent_labels, tuple):
            parent_labels = parent_labels[0]
        if isinstance(child_labels, tuple):
            child_labels = child_labels[0]
            
        # 计算基本损失
        loss, parent_loss, child_loss, routing_loss = criterion(
            parent_logits, child_logits, routed_child_logits,
            parent_labels, child_labels
        )
        
        # 找出恶意邮件样本（子类标签为0）
        # 首先确保child_labels的维度正确
        if len(child_labels.shape) > 1:
            # 如果有额外的维度，进行压缩
            child_labels = child_labels.view(-1)
            
        malicious_mask = (child_labels == 0)
        
        # 如果有恶意邮件样本，增加其损失权重
        if malicious_mask.sum() > 0:
            # 获取恶意邮件样本的索引
            malicious_indices = torch.nonzero(malicious_mask).squeeze(-1)
            
            # 确保有足够的样本
            if malicious_indices.numel() > 0:
                # 计算恶意邮件样本的子类损失，确保维度匹配
                selected_logits = routed_child_logits[malicious_indices]
                selected_labels = child_labels[malicious_indices]
                
                # 打印一下维度，便于调试
                print(f"Selected logits shape: {selected_logits.shape}")
                print(f"Selected labels shape: {selected_labels.shape}")
                
                # 确保标签是一维的
                if len(selected_labels.shape) > 1:
                    selected_labels = selected_labels.view(-1)
                
                try:
                    malicious_loss = F.cross_entropy(
                        selected_logits, 
                        selected_labels,
                        reduction='mean'
                    )
                    
                    # 增加恶意邮件损失权重
                    loss = loss + (args.malicious_weight - 1.0) * malicious_loss
                except Exception as e:
                    print(f"Error in computing malicious loss: {e}")
                    # 出错时不增加额外损失
                    pass
        
        return loss, parent_loss, child_loss, routing_loss
    
    # 初始化优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # 初始化学习率调度器
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=2
    )
    
    # 训练历史记录
    train_losses = []
    val_losses = []
    train_parent_accs = []
    val_parent_accs = []
    train_child_accs = []
    val_child_accs = []
    
    # 早停
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    
    # 开始训练
    print("开始训练...")
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        
        # 如果达到解冻轮次，解冻BERT
        if args.freeze_bert and epoch >= args.unfreeze_epoch:
            print(f"解冻BERT顶层 {args.unfreeze_layers} 层...")
            model.unfreeze_bert_layers(args.unfreeze_layers)
            
            # 重新初始化优化器
            optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.lr / 10,  # 降低学习率
                weight_decay=args.weight_decay
            )
        
        # 训练一个epoch
        train_loss, train_parent_loss, train_child_loss, train_routing_loss = train_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            criterion=weighted_loss_fn,  # 使用带有恶意邮件权重的损失函数
            device=device
        )
        
        # 评估
        train_eval = evaluate(model, train_loader, weighted_loss_fn, device)
        val_eval = evaluate(model, val_loader, weighted_loss_fn, device)
        
        # 更新学习率
        scheduler.step(val_eval['loss'])
        
        # 打印当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        print(f"当前学习率: {current_lr:.6f}")
        
        # 记录训练历史
        train_losses.append(train_loss)
        val_losses.append(val_eval['loss'])
        train_parent_accs.append(train_eval['parent_acc'])
        val_parent_accs.append(val_eval['parent_acc'])
        train_child_accs.append(train_eval['child_acc'])
        val_child_accs.append(val_eval['child_acc'])
        
        # 记录到TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_eval['loss'], epoch)
        writer.add_scalar('Accuracy/train_parent', train_eval['parent_acc'], epoch)
        writer.add_scalar('Accuracy/val_parent', val_eval['parent_acc'], epoch)
        writer.add_scalar('Accuracy/train_child', train_eval['child_acc'], epoch)
        writer.add_scalar('Accuracy/val_child', val_eval['child_acc'], epoch)
        
        # 打印训练信息
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_eval['loss']:.4f}")
        print(f"Train Parent Acc: {train_eval['parent_acc']:.4f}, Val Parent Acc: {val_eval['parent_acc']:.4f}")
        print(f"Train Child Acc: {train_eval['child_acc']:.4f}, Val Child Acc: {val_eval['child_acc']:.4f}")
        
        # 检查是否是最佳模型
        if val_eval['loss'] < best_val_loss:
            best_val_loss = val_eval['loss']
            best_epoch = epoch
            patience_counter = 0
            
            # 保存最佳模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_eval['loss'],
            }, os.path.join(args.output_dir, 'best_model.pth'))
            
            print(f"保存最佳模型，验证损失: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"早停！没有改善 {args.patience} 轮")
                break
    
    # 训练结束
    total_time = time.time() - start_time
    print(f"训练完成！总时间: {total_time:.2f} 秒")
    print(f"最佳模型在第 {best_epoch} 轮，验证损失: {best_val_loss:.4f}")
    
    # 加载最佳模型
    checkpoint = torch.load(os.path.join(args.output_dir, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 在测试集上评估
    print("在测试集上评估...")
    test_eval = evaluate(model, test_loader, weighted_loss_fn, device)
    
    print(f"测试损失: {test_eval['loss']:.4f}")
    print(f"测试父类准确率: {test_eval['parent_acc']:.4f}")
    print(f"测试父类F1分数: {test_eval['parent_f1']:.4f}")
    print(f"测试子类准确率: {test_eval['child_acc']:.4f}")
    print(f"测试子类F1分数: {test_eval['child_f1']:.4f}")
    
    # 获取测试集的预测结果和真实标签
    # 获取预测结果
    parent_preds, child_preds = predict(model, test_loader, device)
    
    # 获取真实标签
    parent_true = []
    child_true = []
    for batch in test_loader:
        parent_true.extend(batch['parent_label'].cpu().numpy())
        child_true.extend(batch['child_label'].cpu().numpy())
    
    parent_true = np.array(parent_true).flatten()
    child_true = np.array(child_true).flatten()
    
    # 定义类别名称
    parent_class_names = ['正常邮件', '垃圾邮件']
    child_class_names = ['恶意邮件', '广告邮件', '欺诈邮件']
    
    # 输出父类分类报告
    print("\n父类分类报告:")
    print(classification_report(parent_true, parent_preds, target_names=parent_class_names))
    
    # 输出子类分类报告（仅对有效样本）
    valid_indices = np.where(child_true != -1)[0]
    if len(valid_indices) > 0:
        print("\n子类分类报告:")
        print(classification_report(
            child_true[valid_indices], 
            child_preds[valid_indices], 
            target_names=child_class_names
        ))
    
    # 绘制混淆矩阵和层次分类结果
    print("\n绘制混淆矩阵和层次分类结果...")
    # 确保字体文件路径正确
    simhei_path = os.path.join(os.getcwd(), 'simhei.ttf')
    if os.path.exists(simhei_path):
        print(f"找到SimHei字体文件: {simhei_path}")
    else:
        print(f"警告: SimHei字体文件不存在于 {simhei_path}")
        # 尝试其他可能的路径
        alt_paths = [
            '/share/home/ncu_418000230048/project/Email_filtering_detection/simhei.ttf',
            os.path.join(os.path.dirname(os.getcwd()), 'simhei.ttf')
        ]
        for path in alt_paths:
            if os.path.exists(path):
                print(f"在替代路径找到SimHei字体文件: {path}")
                simhei_path = path
                break
    
    plot_hierarchical_classification_results(
        parent_true=parent_true,
        parent_pred=parent_preds,
        child_true=child_true,
        child_pred=child_preds,
        parent_class_names=parent_class_names,
        child_class_names=child_class_names,
        save_dir=args.output_dir
    )
    
    # 绘制训练历史
    plot_training_history(
        train_losses=train_losses,
        val_losses=val_losses,
        train_parent_accs=train_parent_accs,
        val_parent_accs=val_parent_accs,
        train_child_accs=train_child_accs,
        val_child_accs=val_child_accs,
        save_path=os.path.join(args.output_dir, 'training_history.png')
    )
    
    # 生成词云图
    print("\n生成邮件类型词云图...")
    generate_word_clouds(df, args.output_dir, simhei_path)
    
    # 生成高频词柱状图
    print("\n生成邮件类型高频词柱状图...")
    plot_top_keywords(df, args.output_dir, simhei_path)
    
    # 将最佳模型保存为pkl格式
    print("\n将最佳模型打包为pkl格式...")
    import pickle
    
    # 创建包含所有必要信息的模型包
    model_package = {
        'model': model,
        'model_state_dict': model.state_dict(),
        'epoch': best_epoch,
        'best_loss': best_val_loss,
        'parent_class_names': parent_class_names,
        'child_class_names': child_class_names,
        'config': vars(args),
        'tfidf_vectorizer': processor.tfidf_vectorizer,
        'bert_tokenizer': processor.bert_tokenizer,
        'metadata': {
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'training_time': total_time,
            'metrics': {
                'test_loss': test_eval['loss'],
                'parent_acc': test_eval['parent_acc'],
                'parent_f1': test_eval['parent_f1'],
                'child_acc': test_eval['child_acc'],
                'child_f1': test_eval['child_f1']
            }
        }
    }
    
    # 保存模型包
    model_package_path = os.path.join(args.output_dir, 'best_model_package.pkl')
    with open(model_package_path, 'wb') as f:
        pickle.dump(model_package, f)
    
    print(f"模型包已保存到: {model_package_path}")
    
    # 关闭TensorBoard写入器
    writer.close()

if __name__ == '__main__':
    main() 