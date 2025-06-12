import os
import jieba
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from collections import Counter
from PIL import Image
import matplotlib.font_manager as fm

def generate_word_clouds(df, output_dir, font_path):
    """
    为不同类型的邮件生成词云图
    
    Args:
        df: 包含邮件数据的DataFrame
        output_dir: 输出目录
        font_path: 字体路径
    """
    # 创建输出目录
    word_cloud_dir = os.path.join(output_dir, 'word_clouds')
    os.makedirs(word_cloud_dir, exist_ok=True)
    
    # 加载自定义字体
    if os.path.exists(font_path):
        print(f"使用字体: {font_path}")
        font_prop = fm.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = [font_prop.get_name()]
    else:
        print(f"警告: 字体文件 {font_path} 不存在")
    
    # 生成各类邮件的词云图
    generate_class_word_cloud(df, parent_class='normal', child_class=None, 
                              title='正常邮件词云图', 
                              output_path=os.path.join(word_cloud_dir, 'normal_wordcloud.png'),
                              font_path=font_path)
    
    generate_class_word_cloud(df, parent_class='spam', child_class='advertising', 
                              title='广告邮件词云图', 
                              output_path=os.path.join(word_cloud_dir, 'advertising_wordcloud.png'),
                              font_path=font_path)
    
    generate_class_word_cloud(df, parent_class='spam', child_class='malicious', 
                              title='恶意邮件词云图', 
                              output_path=os.path.join(word_cloud_dir, 'malicious_wordcloud.png'),
                              font_path=font_path)
    
    generate_class_word_cloud(df, parent_class='spam', child_class='scam', 
                              title='欺诈邮件词云图', 
                              output_path=os.path.join(word_cloud_dir, 'scam_wordcloud.png'),
                              font_path=font_path)
    
    # 生成所有垃圾邮件的词云图
    generate_class_word_cloud(df, parent_class='spam', child_class=None, 
                              title='所有垃圾邮件词云图', 
                              output_path=os.path.join(word_cloud_dir, 'all_spam_wordcloud.png'),
                              font_path=font_path)
    
    print(f"词云图已保存到 {word_cloud_dir}")

def generate_class_word_cloud(df, parent_class, child_class=None, title='词云图', 
                             output_path='wordcloud.png', font_path=None, max_words=200):
    """
    为特定类别的邮件生成词云图
    
    Args:
        df: 包含邮件数据的DataFrame
        parent_class: 父类名称 ('normal' 或 'spam')
        child_class: 子类名称 ('malicious', 'advertising', 'scam' 或 None)
        title: 图表标题
        output_path: 输出路径
        font_path: 字体路径
        max_words: 最大词数
    """
    # 筛选指定类别的邮件
    if child_class:
        filtered_df = df[(df['parent_class'] == parent_class) & (df['child_class'] == child_class)]
    else:
        filtered_df = df[df['parent_class'] == parent_class]
    
    if len(filtered_df) == 0:
        print(f"没有找到类别为 {parent_class} {child_class} 的邮件")
        return
    
    print(f"生成 {parent_class} {child_class} 类别的词云图，共 {len(filtered_df)} 封邮件")
    
    # 合并所有文本
    all_text = ' '.join(filtered_df['processed_text'].fillna('').astype(str))
    
    # 如果processed_text为空，尝试使用content
    if not all_text or all_text.isspace():
        all_text = ' '.join(filtered_df['content'].fillna('').astype(str))
        # 进行简单的文本处理
        all_text = ' '.join(jieba.cut(all_text))
    
    # 生成词云
    generate_word_cloud(all_text, title, output_path, font_path, max_words)

def generate_word_cloud(text, title, output_path, font_path=None, max_words=200):
    """
    生成词云图
    
    Args:
        text: 文本
        title: 图表标题
        output_path: 输出路径
        font_path: 字体路径
        max_words: 最大词数
    """
    # 创建词云对象
    wc = WordCloud(
        font_path=font_path,
        background_color='white',
        max_words=max_words,
        width=800,
        height=600,
        max_font_size=100,
        random_state=42
    )
    
    # 生成词云
    try:
        wc.generate(text)
        
        # 绘制词云图
        plt.figure(figsize=(10, 7))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.title(title, fontsize=20)
        plt.tight_layout()
        
        # 保存图片
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"已保存词云图到 {output_path}")
    except Exception as e:
        print(f"生成词云图时出错: {e}")

def get_top_keywords(df, parent_class, child_class=None, top_n=20):
    """
    获取特定类别邮件的高频关键词
    
    Args:
        df: 包含邮件数据的DataFrame
        parent_class: 父类名称
        child_class: 子类名称
        top_n: 返回前N个关键词
        
    Returns:
        top_keywords: 前N个关键词及其频率
    """
    # 筛选指定类别的邮件
    if child_class:
        filtered_df = df[(df['parent_class'] == parent_class) & (df['child_class'] == child_class)]
    else:
        filtered_df = df[df['parent_class'] == parent_class]
    
    if len(filtered_df) == 0:
        return []
    
    # 合并所有文本
    all_text = ' '.join(filtered_df['processed_text'].fillna('').astype(str))
    words = all_text.split()
    
    # 计算词频
    word_counts = Counter(words)
    
    # 返回前N个高频词
    return word_counts.most_common(top_n)

def plot_top_keywords(df, output_dir, font_path):
    """
    绘制各类邮件的高频关键词柱状图
    
    Args:
        df: 包含邮件数据的DataFrame
        output_dir: 输出目录
        font_path: 字体路径
    """
    # 创建输出目录
    keywords_dir = os.path.join(output_dir, 'keywords')
    os.makedirs(keywords_dir, exist_ok=True)
    
    # 加载自定义字体
    if os.path.exists(font_path):
        font_prop = fm.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = [font_prop.get_name()]
    
    # 获取各类邮件的高频词
    normal_keywords = get_top_keywords(df, 'normal', None, 15)
    advertising_keywords = get_top_keywords(df, 'spam', 'advertising', 15)
    malicious_keywords = get_top_keywords(df, 'spam', 'malicious', 15)
    scam_keywords = get_top_keywords(df, 'spam', 'scam', 15)
    
    # 绘制高频词柱状图
    plot_keywords(normal_keywords, '正常邮件高频词', 
                 os.path.join(keywords_dir, 'normal_keywords.png'), font_path)
    plot_keywords(advertising_keywords, '广告邮件高频词', 
                 os.path.join(keywords_dir, 'advertising_keywords.png'), font_path)
    plot_keywords(malicious_keywords, '恶意邮件高频词', 
                 os.path.join(keywords_dir, 'malicious_keywords.png'), font_path)
    plot_keywords(scam_keywords, '欺诈邮件高频词', 
                 os.path.join(keywords_dir, 'scam_keywords.png'), font_path)
    
    print(f"关键词图表已保存到 {keywords_dir}")

def plot_keywords(keywords, title, output_path, font_path=None):
    """
    绘制关键词柱状图
    
    Args:
        keywords: 关键词列表 [(word, count), ...]
        title: 图表标题
        output_path: 输出路径
        font_path: 字体路径
    """
    if not keywords:
        print(f"没有关键词数据用于绘制 {title}")
        return
    
    # 提取词和频率
    words = [item[0] for item in keywords]
    counts = [item[1] for item in keywords]
    
    # 创建水平柱状图
    plt.figure(figsize=(12, 8))
    y_pos = np.arange(len(words))
    
    # 横向柱状图，按频率降序排列
    plt.barh(y_pos, counts, align='center', alpha=0.8, color='skyblue')
    plt.yticks(y_pos, words)
    plt.xlabel('频率')
    plt.title(title)
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"已保存关键词图表到 {output_path}") 