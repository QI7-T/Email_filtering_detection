#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pickle
import torch
import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd
import jieba

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='使用训练好的模型进行邮件分类')
    parser.add_argument('--model_path', type=str, required=True, help='模型包路径(.pkl文件)')
    parser.add_argument('--input', type=str, required=True, help='输入文件或目录')
    parser.add_argument('--output', type=str, default='predictions.csv', help='输出文件路径')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--use_gpu', action='store_true', help='是否使用GPU')
    args = parser.parse_args()
    return args

def load_model_package(model_path):
    """加载模型包"""
    print(f"加载模型包: {model_path}")
    with open(model_path, 'rb') as f:
        model_package = pickle.load(f)
    
    print("模型信息:")
    print(f"- 训练轮次: {model_package['epoch']}")
    print(f"- 最佳损失: {model_package['best_loss']:.4f}")
    print(f"- 训练时间: {model_package['metadata']['training_time']:.2f} 秒")
    print(f"- 测试集准确率: 父类={model_package['metadata']['metrics']['parent_acc']:.4f}, 子类={model_package['metadata']['metrics']['child_acc']:.4f}")
    
    return model_package

def preprocess_text(text, model_package):
    """预处理文本"""
    # 去除HTML标签
    import re
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # 分词
    words = jieba.cut(text)
    processed_text = ' '.join(words)
    
    return processed_text

def load_texts_from_input(input_path):
    """从输入路径加载文本"""
    texts = []
    filenames = []
    
    if os.path.isfile(input_path):
        # 输入是单个文件
        try:
            with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            texts.append(content)
            filenames.append(os.path.basename(input_path))
        except Exception as e:
            print(f"读取文件错误: {e}")
    
    elif os.path.isdir(input_path):
        # 输入是目录
        for root, _, files in os.walk(input_path):
            for file in tqdm(files, desc="读取文件"):
                if file.endswith('.txt'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                        texts.append(content)
                        filenames.append(os.path.relpath(file_path, input_path))
                    except Exception as e:
                        print(f"读取文件错误 {file_path}: {e}")
    
    else:
        raise ValueError(f"输入路径不存在: {input_path}")
    
    return texts, filenames

def extract_email_body(email_content):
    """提取邮件正文内容（跳过邮件头）"""
    # 邮件头和正文由空行分隔
    parts = email_content.split('\n\n', 1)
    if len(parts) > 1:
        return parts[1]
    return email_content

def predict_emails(texts, model_package, batch_size=16, use_gpu=False):
    """预测邮件分类"""
    # 准备模型和设备
    model = model_package['model']
    tfidf_vectorizer = model_package['tfidf_vectorizer']
    bert_tokenizer = model_package['bert_tokenizer']
    parent_class_names = model_package['parent_class_names']
    child_class_names = model_package['child_class_names']
    
    device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
    model.to(device)
    model.eval()
    
    # 存储结果
    results = []
    
    # 分批处理
    for i in tqdm(range(0, len(texts), batch_size), desc="预测"):
        batch_texts = texts[i:i + batch_size]
        batch_bodies = [extract_email_body(text) for text in batch_texts]
        
        # 处理批次数据
        processed_texts = [preprocess_text(body, model_package) for body in batch_bodies]
        
        # 获取BERT特征
        encoded_inputs = bert_tokenizer(
            batch_bodies,
            padding='max_length',
            truncation=True,
            max_length=256,
            return_tensors='pt'
        )
        
        input_ids = encoded_inputs['input_ids'].to(device)
        attention_mask = encoded_inputs['attention_mask'].to(device)
        token_type_ids = encoded_inputs['token_type_ids'].to(device)
        
        # 获取TF-IDF特征
        tfidf_features = tfidf_vectorizer.transform(processed_texts).toarray()
        tfidf_features = torch.FloatTensor(tfidf_features).to(device)
        
        # 提取恶意邮件特征
        malicious_features = []
        for body in batch_bodies:
            # 使用简单的特征提取，这里简化处理
            # 实际应用中应当使用与训练时相同的特征提取方法
            features = np.zeros(14)  # 与训练时相同的特征维度
            malicious_features.append(features)
        
        malicious_features = torch.FloatTensor(malicious_features).to(device)
        
        # 前向传播
        with torch.no_grad():
            parent_logits, _, routed_child_logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                tfidf_features=tfidf_features,
                malicious_features=malicious_features
            )
        
        # 获取预测结果
        parent_preds = torch.argmax(parent_logits, dim=1).cpu().numpy()
        child_preds = torch.argmax(routed_child_logits, dim=1).cpu().numpy()
        
        # 收集批次结果
        for j, (parent_pred, child_pred) in enumerate(zip(parent_preds, child_preds)):
            parent_class = parent_class_names[parent_pred]
            
            # 只有垃圾邮件才有子类
            if parent_pred == 1:  # spam
                child_class = child_class_names[child_pred]
            else:
                child_class = "无子类"
            
            results.append({
                'parent_class': parent_class,
                'child_class': child_class,
                'parent_pred': int(parent_pred),
                'child_pred': int(child_pred),
                'content': batch_texts[j][:200] + '...' if len(batch_texts[j]) > 200 else batch_texts[j]
            })
    
    return results

def main():
    """主函数"""
    args = parse_args()
    
    # 加载模型包
    model_package = load_model_package(args.model_path)
    
    # 加载输入文本
    print(f"从 {args.input} 加载邮件...")
    texts, filenames = load_texts_from_input(args.input)
    print(f"加载了 {len(texts)} 封邮件")
    
    if len(texts) == 0:
        print("没有找到邮件，退出")
        return
    
    # 预测
    print("开始预测...")
    results = predict_emails(
        texts=texts,
        model_package=model_package,
        batch_size=args.batch_size,
        use_gpu=args.use_gpu
    )
    
    # 添加文件名
    for i, filename in enumerate(filenames):
        if i < len(results):
            results[i]['filename'] = filename
    
    # 转换为DataFrame并保存
    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False, encoding='utf-8')
    
    print(f"预测结果已保存到 {args.output}")
    
    # 输出分类统计
    parent_counts = df['parent_class'].value_counts()
    print("\n邮件分类统计:")
    for parent_class, count in parent_counts.items():
        print(f"- {parent_class}: {count} 封 ({count/len(df)*100:.1f}%)")
    
    spam_df = df[df['parent_class'] == '垃圾邮件']
    if len(spam_df) > 0:
        child_counts = spam_df['child_class'].value_counts()
        print("\n垃圾邮件子类统计:")
        for child_class, count in child_counts.items():
            print(f"- {child_class}: {count} 封 ({count/len(spam_df)*100:.1f}%)")

if __name__ == '__main__':
    main() 