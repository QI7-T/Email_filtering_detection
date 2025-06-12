import os
import re
import jieba
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader
import torch

# 导入数据增强和特征提取模块
from data.data_augmentation import EmailAugmenter
from data.malicious_feature_extractor import MaliciousFeatureExtractor

# 邮件层次类别定义
PARENT_CLASSES = {
    'normal': 0,  # 正常邮件
    'spam': 1     # 垃圾邮件
}

CHILD_CLASSES = {
    # 垃圾邮件子类
    'malicious': 0,    # 恶意邮件
    'advertising': 1,  # 广告邮件
    'scam': 2          # 欺诈邮件
}

# 子类到父类的映射
CHILD_TO_PARENT = {
    'malicious': 'spam',
    'advertising': 'spam',
    'scam': 'spam'
}

class EmailProcessor:
    def __init__(self, data_dir, bert_model_path, max_length=256, tfidf_max_features=20000, use_augmentation=True):
        """
        初始化邮件处理器
        
        Args:
            data_dir: 数据集根目录
            bert_model_path: BERT模型路径
            max_length: BERT输入最大长度
            tfidf_max_features: TF-IDF特征数量
            use_augmentation: 是否使用数据增强
        """
        self.data_dir = data_dir
        self.max_length = max_length
        self.tfidf_max_features = tfidf_max_features
        self.use_augmentation = use_augmentation
        
        # 加载BERT分词器
        self.bert_tokenizer = BertTokenizer.from_pretrained(bert_model_path)
        
        # 加载停用词
        self.stopwords = self._load_stopwords(os.path.join(data_dir, 'stopwords_hit.txt'))
        
        # TF-IDF向量化器
        self.tfidf_vectorizer = None
        
        # 初始化数据增强器
        self.augmenter = EmailAugmenter()
        
        # 初始化恶意邮件特征提取器
        self.malicious_feature_extractor = MaliciousFeatureExtractor()
    
    def _load_stopwords(self, stopwords_file):
        """加载停用词"""
        with open(stopwords_file, 'r', encoding='utf-8', errors='ignore') as f:
            stopwords = [line.strip() for line in f.readlines()]
        return set(stopwords)
    
    def _extract_email_body(self, email_content):
        """提取邮件正文内容（跳过邮件头）"""
        # 邮件头和正文由空行分隔
        parts = email_content.split('\n\n', 1)
        if len(parts) > 1:
            return parts[1]
        return email_content
    
    def _preprocess_text(self, text):
        """文本预处理：去除特殊字符、分词、去停用词"""
        # 去除HTML标签
        text = re.sub(r'<[^>]+>', ' ', text)
        
        # 去除特殊字符和数字
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', ' ', text)
        
        # 分词并去除停用词
        words = jieba.cut(text)
        words = [word for word in words if word not in self.stopwords and len(word.strip()) > 0]
        
        return ' '.join(words)
    
    def _get_class_from_path(self, file_path):
        """从文件路径中提取类别标签"""
        path_parts = file_path.split(os.sep)
        
        # 正常邮件
        if 'ham' in path_parts:
            return 'normal', None
        
        # 垃圾邮件及其子类
        parent_class = 'spam'
        
        # 恶意邮件
        if 'Malicious emails' in path_parts:
            child_class = 'malicious'
        
        # 广告邮件
        elif 'Advertising emails' in path_parts:
            child_class = 'advertising'
        
        # 欺诈邮件
        elif 'Scam emails' in path_parts:
            child_class = 'scam'
        
        else:
            return None, None
        
        return parent_class, child_class
    
    def load_data(self):
        """加载所有邮件数据"""
        data = []
        
        # 遍历数据目录下的所有文件
        for root, _, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith('.txt') and file != 'stopWords.txt':
                    file_path = os.path.join(root, file)
                    
                    # 获取类别标签
                    parent_class, child_class = self._get_class_from_path(file_path)
                    if parent_class is None:
                        continue
                    
                    # 读取邮件内容
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")
                        continue
                    
                    # 提取邮件正文
                    body = self._extract_email_body(content)
                    
                    # 预处理文本
                    processed_text = self._preprocess_text(body)
                    
                    # 添加到数据集
                    data.append({
                        'file_path': file_path,
                        'content': body,
                        'processed_text': processed_text,
                        'parent_class': parent_class,
                        'child_class': child_class,
                        'parent_label': PARENT_CLASSES.get(parent_class, -1),
                        'child_label': CHILD_CLASSES.get(child_class, -1) if child_class else -1
                    })
        
        # 转换为DataFrame
        df = pd.DataFrame(data)
        
        # 应用数据增强（如果启用）
        if self.use_augmentation:
            print("应用数据增强...")
            # 对恶意邮件进行过采样
            df = self.augmenter.oversample_malicious_emails(df)
            
            # 对恶意邮件进行文本增强
            df = self.augmenter.augment_malicious_emails(df)
        
        # 训练TF-IDF向量化器
        print(f"训练TF-IDF向量化器，最大特征数量: {self.tfidf_max_features}")
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=self.tfidf_max_features,
            min_df=2,
            max_df=0.95
        )
        self.tfidf_vectorizer.fit(df['processed_text'])
        
        # 打印实际的特征数量
        actual_features = len(self.tfidf_vectorizer.get_feature_names_out())
        print(f"TF-IDF实际特征数量: {actual_features}")
        
        # 提取恶意邮件特征
        print("提取恶意邮件特征...")
        df['malicious_features'] = df['content'].apply(
            lambda x: self.malicious_feature_extractor.extract_features(x)
        )
        
        return df
    
    def get_bert_features(self, texts):
        """使用BERT分词器处理文本"""
        encoded_inputs = self.bert_tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoded_inputs['input_ids'],
            'attention_mask': encoded_inputs['attention_mask'],
            'token_type_ids': encoded_inputs['token_type_ids']
        }
    
    def get_tfidf_features(self, texts):
        """使用TF-IDF提取文本特征"""
        if self.tfidf_vectorizer is None:
            raise ValueError("TF-IDF vectorizer not trained. Call load_data() first.")
        
        return self.tfidf_vectorizer.transform(texts).toarray()
    
    def get_malicious_features(self, texts):
        """提取恶意邮件特征"""
        return np.array([self.malicious_feature_extractor.extract_features(text) for text in texts])


class EmailDataset(Dataset):
    def __init__(self, dataframe, processor, is_training=True):
        """
        邮件数据集
        
        Args:
            dataframe: 包含邮件数据的DataFrame
            processor: EmailProcessor实例
            is_training: 是否为训练模式
        """
        self.data = dataframe
        self.processor = processor
        self.is_training = is_training
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        
        # 获取BERT特征
        bert_features = self.processor.get_bert_features([item['content']])
        
        # 获取TF-IDF特征
        tfidf_features = self.processor.get_tfidf_features([item['processed_text']])
        
        # 获取恶意邮件特征
        malicious_features = torch.FloatTensor(item['malicious_features'])
        
        # 构建样本
        sample = {
            'input_ids': bert_features['input_ids'][0],
            'attention_mask': bert_features['attention_mask'][0],
            'token_type_ids': bert_features['token_type_ids'][0],
            'tfidf_features': torch.FloatTensor(tfidf_features[0]),
            'malicious_features': malicious_features,
            'parent_label': torch.LongTensor([item['parent_label']]),
        }
        
        # 如果有子类标签，也加入样本
        if item['child_label'] != -1:
            sample['child_label'] = torch.LongTensor([item['child_label']])
        else:
            sample['child_label'] = torch.LongTensor([-1])  # 无效标签
        
        return sample


def create_data_loaders(dataframe, processor, batch_size=16, train_ratio=0.8, val_ratio=0.1):
    """
    创建训练、验证和测试数据加载器
    
    Args:
        dataframe: 包含邮件数据的DataFrame
        processor: EmailProcessor实例
        batch_size: 批次大小
        train_ratio: 训练集比例
        val_ratio: 验证集比例
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # 随机打乱数据
    df_shuffled = dataframe.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # 划分数据集
    train_size = int(len(df_shuffled) * train_ratio)
    val_size = int(len(df_shuffled) * val_ratio)
    
    train_df = df_shuffled[:train_size]
    val_df = df_shuffled[train_size:train_size + val_size]
    test_df = df_shuffled[train_size + val_size:]
    
    # 创建数据集
    train_dataset = EmailDataset(train_df, processor, is_training=True)
    val_dataset = EmailDataset(val_df, processor, is_training=False)
    test_dataset = EmailDataset(test_df, processor, is_training=False)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader 