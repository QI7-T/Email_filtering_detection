import re
import numpy as np
import torch
import torch.nn as nn
import tldextract
from urllib.parse import urlparse
import pandas as pd

class MaliciousFeatureExtractor:
    """恶意邮件特征提取器"""
    
    def __init__(self):
        """初始化恶意邮件特征提取器"""
        # 恶意邮件常见特征词汇
        self.malicious_keywords = [
            '账号', '密码', '验证', '银行', '登录', '安全', '紧急', '确认', 
            '点击', '链接', '更新', '信息', '异常', '风险', '冻结', '解冻', 
            '中奖', '奖金', '支付', '转账', '认证', '警告', '重要通知'
        ]
        
        # 常见钓鱼域名特征
        self.phishing_domains = [
            'secure-', 'verify-', 'account-', 'service-', 'update-',
            'login-', 'signin-', 'auth-', 'confirm-', 'validate-',
            'security', 'banking', 'payment', 'wallet', 'verify',
            'authenticate', 'recover', 'unlock', 'alert'
        ]
        
        # 可疑脚本文件扩展名
        self.suspicious_extensions = [
            '.php', '.asp', '.aspx', '.jsp', '.cgi', '.exe', '.bat',
            '.sh', '.pl', '.py', '.js', '.vbs'
        ]
        
        # 可疑域名后缀（非标准或常被滥用的TLD）
        self.suspicious_tlds = [
            'xyz', 'top', 'club', 'online', 'site', 'work', 'tech',
            'gq', 'ml', 'cf', 'ga', 'tk', 'buzz', 'icu', 'monster',
            'rest', 'fit', 'loan', 'download', 'racing', 'date', 'cn', 'net'
        ]
        
        # 常见品牌名称（用于检测品牌仿冒）
        self.common_brands = [
            'paypal', 'apple', 'microsoft', 'amazon', 'google', 'facebook',
            'netflix', 'twitter', 'instagram', 'alibaba', 'taobao', 'jd',
            'wechat', 'alipay', 'weibo', 'baidu', 'tencent', 'qq',
            'bank', 'chase', 'wellsfargo', 'citi', 'hsbc', 'visa', 'mastercard'
        ]
        
        # 敏感信息模式
        self.sensitive_patterns = [
            r'密码', r'账号', r'验证码', r'银行卡', r'身份证',
            r'登录', r'点击', r'链接', r'确认', r'紧急'
        ]
    
    def _extract_urls(self, text):
        """提取文本中的URL"""
        url_pattern = r'https?://[^\s<>"]+|www\.[^\s<>"]+|[a-zA-Z0-9][a-zA-Z0-9-]{1,61}[a-zA-Z0-9]\.[a-zA-Z]{2,}'
        urls = re.findall(url_pattern, text)
        return urls
    
    def _analyze_url(self, url):
        """分析URL的可疑特征"""
        features = {
            'has_suspicious_extension': False,
            'has_suspicious_tld': False,
            'has_phishing_domain': False,
            'has_brand_name': False,
            'has_numeric_domain': False,
            'has_long_subdomain': False,
            'has_ip_address': False,
            'has_unusual_port': False,
            'has_many_dots': False
        }
        
        # 标准化URL
        if not url.startswith('http'):
            url = 'http://' + url
        
        try:
            # 解析URL
            parsed_url = urlparse(url)
            path = parsed_url.path.lower()
            
            # 提取域名信息
            extracted = tldextract.extract(url)
            domain = extracted.domain
            subdomain = extracted.subdomain
            suffix = extracted.suffix
            
            # 检查可疑脚本扩展名
            for ext in self.suspicious_extensions:
                if path.endswith(ext):
                    features['has_suspicious_extension'] = True
                    break
            
            # 检查可疑TLD
            if suffix in self.suspicious_tlds:
                features['has_suspicious_tld'] = True
            
            # 检查钓鱼域名特征
            for phish in self.phishing_domains:
                if phish in domain or phish in subdomain:
                    features['has_phishing_domain'] = True
                    break
            
            # 检查品牌名称
            for brand in self.common_brands:
                if brand in domain and brand != domain:  # 品牌名在域名中但不完全匹配
                    features['has_brand_name'] = True
                    break
            
            # 检查数字域名
            if re.search(r'\d{3,}', domain):
                features['has_numeric_domain'] = True
            
            # 检查子域名长度
            if len(subdomain) > 20:
                features['has_long_subdomain'] = True
            
            # 检查IP地址
            if re.match(r'\d+\.\d+\.\d+\.\d+', parsed_url.netloc):
                features['has_ip_address'] = True
            
            # 检查非标准端口
            if parsed_url.port and parsed_url.port not in [80, 443]:
                features['has_unusual_port'] = True
            
            # 检查域名中的点数量
            if parsed_url.netloc.count('.') > 3:
                features['has_many_dots'] = True
                
        except Exception:
            # URL解析错误，可能是恶意构造的URL
            return {'parsing_error': True}
        
        return features
    
    def extract_features(self, text):
        """
        提取恶意邮件特征
        
        Args:
            text: 邮件文本
            
        Returns:
            特征向量 (numpy array)
        """
        # 确保text是字符串类型
        if isinstance(text, pd.Series):
            text = text.iloc[0] if not text.empty else ""
        
        # 如果text不是字符串，尝试转换为字符串
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
            
        features = []
        
        # 1. 提取所有URL
        urls = self._extract_urls(text)
        features.append(min(len(urls), 10) / 10.0)  # URL数量（归一化）
        
        # 2. URL特征分析
        suspicious_extension_count = 0
        suspicious_tld_count = 0
        phishing_domain_count = 0
        brand_impersonation_count = 0
        suspicious_url_count = 0
        
        for url in urls:
            url_features = self._analyze_url(url)
            
            # 计数各种可疑特征
            if url_features.get('has_suspicious_extension', False):
                suspicious_extension_count += 1
            
            if url_features.get('has_suspicious_tld', False):
                suspicious_tld_count += 1
            
            if url_features.get('has_phishing_domain', False):
                phishing_domain_count += 1
            
            if url_features.get('has_brand_name', False):
                brand_impersonation_count += 1
            
            # 如果URL有多个可疑特征，认为它是高度可疑的
            suspicious_score = sum(1 for v in url_features.values() if v)
            if suspicious_score >= 2:
                suspicious_url_count += 1
        
        # 添加URL分析特征（归一化）
        features.append(min(suspicious_extension_count, 5) / 5.0)
        features.append(min(suspicious_tld_count, 5) / 5.0)
        features.append(min(phishing_domain_count, 5) / 5.0)
        features.append(min(brand_impersonation_count, 5) / 5.0)
        features.append(min(suspicious_url_count, 5) / 5.0)
        
        # 3. 恶意关键词频率
        keyword_count = 0
        for keyword in self.malicious_keywords:
            keyword_count += text.lower().count(keyword.lower())
        features.append(min(keyword_count, 20) / 20.0)  # 归一化
        
        # 4. 紧急程度指标 (感叹号数量)
        urgency_score = text.count('!')
        features.append(min(urgency_score, 10) / 10.0)  # 归一化
        
        # 5. 敏感信息请求指标
        sensitive_count = 0
        for pattern in self.sensitive_patterns:
            sensitive_count += len(re.findall(pattern, text))
        features.append(min(sensitive_count, 15) / 15.0)  # 归一化
        
        # 6. 文本长度
        text_length = len(text)
        features.append(min(text_length, 5000) / 5000.0)  # 归一化
        
        # 7. 大写字母比例
        uppercase_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        features.append(uppercase_ratio)
        
        # 8. 特殊字符比例
        special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
        special_char_ratio = special_chars / max(len(text), 1)
        features.append(special_char_ratio)
        
        # 9. HTML标签数量
        html_tags = len(re.findall(r'<[^>]+>', text))
        features.append(min(html_tags, 50) / 50.0)  # 归一化
        
        # 10. 隐藏链接特征（HTML中的链接文本与href不匹配）
        hidden_links = len(re.findall(r'<a\s+[^>]*href\s*=\s*["\']([^"\']+)["\'][^>]*>(?!.*\1).*?</a>', text, re.IGNORECASE | re.DOTALL))
        features.append(min(hidden_links, 5) / 5.0)  # 归一化
        
        return np.array(features)


class MaliciousFeatureNetwork(nn.Module):
    """恶意邮件特征网络"""
    
    def __init__(self, input_dim=14, hidden_dim=32):
        """
        初始化恶意邮件特征网络
        
        Args:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度
        """
        super(MaliciousFeatureNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入特征 [batch_size, input_dim]
            
        Returns:
            输出特征 [batch_size, hidden_dim]
        """
        return self.network(x) 