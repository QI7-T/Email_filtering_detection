import random
import re
import numpy as np
import pandas as pd
from sklearn.utils import resample

class EmailAugmenter:
    """邮件数据增强器，专注于恶意邮件增强"""
    
    def __init__(self, random_state=42):
        """
        初始化邮件数据增强器
        
        Args:
            random_state: 随机种子
        """
        self.random_state = random_state
        random.seed(random_state)
        np.random.seed(random_state)
        
        # 恶意邮件常见特征词汇
        self.malicious_keywords = [
            '账号', '密码', '验证', '银行', '登录', '安全', '紧急', '确认', 
            '点击', '链接', '更新', '信息', '异常', '风险', '冻结', '解冻', 
            '中奖', '奖金', '支付', '转账', '认证', '警告', '重要通知'
        ]
        
        # 常见钓鱼域名特征
        self.phishing_domains = [
            'secure-', 'verify-', 'account-', 'service-', 'update-',
            'login-', 'signin-', 'auth-', 'confirm-', 'validate-'
        ]
        
    def oversample_malicious_emails(self, df, target_ratio=0.8):
        """
        对恶意邮件进行过采样，使其数量接近其他类别
        
        Args:
            df: 包含邮件数据的DataFrame
            target_ratio: 目标采样比例（相对于最多的类别）
            
        Returns:
            过采样后的DataFrame
        """
        # 获取各子类数量
        class_counts = df['child_class'].value_counts()
        print(f"原始类别分布: {class_counts}")
        
        # 找出最多的类别数量
        max_class_count = class_counts.max()
        target_count = int(max_class_count * target_ratio)
        
        # 获取恶意邮件
        malicious_emails = df[df['child_class'] == 'malicious']
        
        # 如果恶意邮件数量少于目标数量，进行过采样
        if len(malicious_emails) < target_count:
            # 过采样
            oversampled_malicious = resample(
                malicious_emails,
                replace=True,
                n_samples=target_count,
                random_state=self.random_state
            )
            
            # 合并过采样数据
            df_oversampled = pd.concat([df[df['child_class'] != 'malicious'], oversampled_malicious])
            
            print(f"过采样后类别分布: {df_oversampled['child_class'].value_counts()}")
            return df_oversampled
        
        return df
    
    def augment_text(self, text, augmentation_type='synonym'):
        """
        对文本进行数据增强
        
        Args:
            text: 原始文本
            augmentation_type: 增强类型，可选 'synonym', 'insert', 'swap', 'delete'
            
        Returns:
            增强后的文本
        """
        # 确保text是字符串类型
        if isinstance(text, pd.Series):
            text = text.iloc[0] if not text.empty else ""
        
        # 如果text不是字符串，尝试转换为字符串
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
            
        if augmentation_type == 'synonym':
            # 简单的同义词替换
            for keyword in random.sample(self.malicious_keywords, min(5, len(self.malicious_keywords))):
                if keyword in text and random.random() > 0.5:
                    idx = random.randint(0, len(self.malicious_keywords) - 1)
                    text = text.replace(keyword, self.malicious_keywords[idx], 1)
        
        elif augmentation_type == 'insert':
            # 随机插入恶意关键词
            words = text.split()
            for _ in range(min(3, len(words) // 10)):
                keyword = random.choice(self.malicious_keywords)
                position = random.randint(0, len(words))
                words.insert(position, keyword)
            text = ' '.join(words)
        
        elif augmentation_type == 'swap':
            # 随机交换相邻词
            words = text.split()
            for _ in range(min(3, len(words) // 10)):
                if len(words) > 2:
                    idx = random.randint(0, len(words) - 2)
                    words[idx], words[idx + 1] = words[idx + 1], words[idx]
            text = ' '.join(words)
        
        elif augmentation_type == 'delete':
            # 随机删除一些词
            words = text.split()
            words = [word for word in words if random.random() > 0.1]
            text = ' '.join(words)
        
        return text
    
    def augment_malicious_emails(self, df):
        """
        对恶意邮件进行文本增强
        
        Args:
            df: 包含邮件数据的DataFrame
            
        Returns:
            增强后的DataFrame
        """
        # 复制DataFrame
        df_augmented = df.copy()
        
        # 对恶意邮件进行增强
        malicious_indices = df_augmented[df_augmented['child_class'] == 'malicious'].index
        
        for idx in malicious_indices:
            # 随机选择一种增强方法
            aug_type = random.choice(['synonym', 'insert', 'swap', 'delete'])
            
            # 增强文本 - 确保获取的是字符串
            original_text = df_augmented.loc[idx, 'content']
            if not isinstance(original_text, str):
                original_text = str(original_text) if original_text is not None else ""
                
            augmented_text = self.augment_text(original_text, aug_type)
            
            # 更新文本
            df_augmented.loc[idx, 'content'] = augmented_text
            
            # 标记为增强样本
            if 'is_augmented' not in df_augmented.columns:
                df_augmented['is_augmented'] = False
            df_augmented.loc[idx, 'is_augmented'] = True
        
        return df_augmented
    
    def extract_malicious_features(self, text):
        """
        提取恶意邮件特征
        
        Args:
            text: 邮件文本
            
        Returns:
            特征字典
        """
        # 确保text是字符串类型
        if isinstance(text, pd.Series):
            text = text.iloc[0] if not text.empty else ""
        
        # 如果text不是字符串，尝试转换为字符串
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
            
        features = {}
        
        # 1. 检测URL数量
        url_pattern = r'https?://\S+|www\.\S+'
        urls = re.findall(url_pattern, text)
        features['url_count'] = len(urls)
        
        # 2. 检测可疑域名
        suspicious_domain_count = 0
        for url in urls:
            for domain in self.phishing_domains:
                if domain in url:
                    suspicious_domain_count += 1
                    break
        features['suspicious_domain_count'] = suspicious_domain_count
        
        # 3. 恶意关键词频率
        keyword_count = 0
        for keyword in self.malicious_keywords:
            keyword_count += text.lower().count(keyword.lower())
        features['malicious_keyword_count'] = keyword_count
        
        # 4. 紧急程度指标 (感叹号数量)
        features['urgency_score'] = text.count('!')
        
        # 5. 敏感信息请求指标
        sensitive_patterns = [
            r'密码', r'账号', r'验证码', r'银行卡', r'身份证',
            r'登录', r'点击', r'链接', r'确认', r'紧急'
        ]
        sensitive_count = 0
        for pattern in sensitive_patterns:
            sensitive_count += len(re.findall(pattern, text))
        features['sensitive_info_request'] = sensitive_count
        
        return features

def apply_data_augmentation(df):
    """
    应用数据增强
    
    Args:
        df: 原始DataFrame
        
    Returns:
        增强后的DataFrame
    """
    augmenter = EmailAugmenter()
    
    # 1. 对恶意邮件进行过采样
    df_oversampled = augmenter.oversample_malicious_emails(df)
    
    # 2. 对恶意邮件进行文本增强
    df_augmented = augmenter.augment_malicious_emails(df_oversampled)
    
    return df_augmented 