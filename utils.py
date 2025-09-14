import re
import ftfy
from langdetect import detect
from datasketch import MinHash, MinHashLSH
import jieba
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
import torch
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import os
from scipy.stats import gaussian_kde
import plotly.express as px




class dataClean():
    """
    清洗
    """
    def __init__(self,min_length=5):
        self.min_length = min_length
        
    def clean_text(self,text):
        # 修复编码问题
        text = ftfy.fix_text(text)

        # 语言过滤
        # 文本太短或无法判断语言时会报错
        try:
            if detect(text) not in ("zh", "zh-cn", "en"):
                return
        except:
            print(f"报错文本{text}")

        # 特殊字符处理
        text = re.sub(r"[\x00-\x1F\x7F-\x9F]", "", text)  # 控制字符
        text = re.sub(r"[【】■◆●▲©®™]", "", text)  # 广告符号

        # URL/邮箱过滤
        text = re.sub(r"http\S+|www\.\S+", "", text)
        text = re.sub(r"\S+@\S+", "", text)

        # 冗余空格处理
        text = re.sub(r"\s+", " ", text).strip()

        # 归一化
        # 统一全角/半角
        text = re.sub(r"[\s\u3000]+", " ", text.strip())
        # 统一标点
        text = re.sub(r"[！!]+", "!", text)
        text = re.sub(r"[？?]+", "?", text)
        text = re.sub(r"[。，,]+", "，", text)
        # 小写化（对中英文混合有效）
        text = text.lower()
        
        return text
    
    def is_short(self,text):
        """
        过滤长度
        """
        text = str(text)
        return len(text) < self.min_length


class Deduplicator():
    """
    去重
    """
    def __init__(self,threshold=0.85, max_words=200,num_perm=128):
        self.threshold = threshold
        self.num_perm = num_perm
        self.max_words = max_words
        self.lsh = MinHashLSH(self.threshold, self.num_perm)

    def chinese_tokenize(self,text):
        """中文分词"""
        return list(jieba.cut(text))[:self.max_words]

    def deduplicate_text(self,text):
        words = self.chinese_tokenize(text)
        
        if len(words) == 0: return

        m = MinHash(self.num_perm)
        for word in words:
            word = word.strip()
            if not word:
                continue
            m.update(word.encode("utf-8"))
            
        # 查询相似文档
        return bool(self.lsh.query(m))
    

class toxicityFilter():
    """
    过滤-毒性检测
    """
    def __init__(self, model_path,):
        # 加载 tokenizer 和 模型
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        # 设置为评估模式
        self.model.eval()
    # 先用 0.5 作为起点，在你的数据集上评估效果，再根据业务需求微调。
    def is_toxic(self,text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.sigmoid(logits).numpy()[0]
        return float(probabilities[0])



class qualityFilter():
    """
    过滤-困惑度检测（和信息熵关系密切）
    """
    def __init__(self, model_path):
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
    
    def is_low_quality(self, text, min_ppl=25, max_ppl=150):
        """
        批量计算困惑度（Perplexity）
        """
        if len(text.strip()) == 0:
            return np.nan  # 空文本视为低质量
        
        # 在推理阶段，不需要反向传播，所以关闭梯度计算。减少显存占用，大幅提升速度
        with torch.no_grad():
            try:
                inputs = self.tokenizer(
                    text, return_tensors="pt", truncation=True, max_length=512
                )
                outputs = self.model(**inputs, labels=inputs["input_ids"])
                ppl = torch.exp(outputs.loss).cpu().item()
            except:
                ppl = np.nan
        return ppl

def cttr(text):
    # 词汇多样性检测
    words = jieba.lcut(text.strip())
    words = [w for w in words if w.strip() and w not in "，。！？；：“”‘’（）【】《》、"]
    n_tokens = len(words)
    n_types = len(set(words))
    if n_tokens == 0:
        return 0.0
    return n_types / sqrt(2 * n_tokens)

class qualityEval():
    """
    质量评估模块
    """
    def __init__(self, max_toxic,max_ppl,detail_path,stats_path):
        self.max_toxic = max_toxic
        self.max_ppl = max_ppl
        self.detail_path = detail_path
        self.stats_path = stats_path
        
        # 从 json 文件加载 detail，detail的形式为jsonl
        self.df = pd.read_json(self.detail_path, lines=True)

        # 从 json 文件加载 stats
        with open(self.stats_path, 'r', encoding='utf-8') as f:
            self.stats = json.load(f)


    def get_eval_result(self):
        """
        结果统计
        """
        # 重复度
        duplica_before_ratio = np.round(self.stats.get('duplicates') / self.stats.get('total'),4)
        duplica_after_ratio = 0
        # 毒性分布
        mean_toxic_before = self.df.get('toxic', pd.Series()).mean()
        max_toxic_before = self.df.get('toxic', pd.Series()).max()
        min_toxic_before = self.df.get('toxic', pd.Series()).min()
        mean_toxic_after = self.df.get('toxic', pd.Series()).loc[lambda x: x < self.max_toxic].mean()
        max_toxic_after = self.df.get('toxic', pd.Series()).loc[lambda x: x < self.max_toxic].max()
        min_toxic_after = self.df.get('toxic', pd.Series()).loc[lambda x: x < self.max_toxic].min()
        # 平均困惑度
        mean_ppl_before = self.df.get('ppl', pd.Series()).mean()
        max_ppl_before = self.df.get('ppl', pd.Series()).max()
        min_ppl_before = self.df.get('ppl', pd.Series()).min()
        mean_ppl_after = self.df.get('ppl', pd.Series()).loc[lambda x: x < self.max_ppl].mean()
        max_ppl_after = self.df.get('ppl', pd.Series()).loc[lambda x: x < self.max_ppl].max()
        min_ppl_after = self.df.get('ppl', pd.Series()).loc[lambda x: x < self.max_ppl].min()
        # mean_ppl_after = self.df.get('ppl_after', pd.Series()).mean()
        # max_ppl_after = self.df.get('ppl_after', pd.Series()).max()
        # min_ppl_after = self.df.get('ppl_after', pd.Series()).min()
        # 词汇多样性
        mean_cttr_before = self.df.get('cttr_before', pd.Series()).mean()
        max_cttr_before = self.df.get('cttr_before', pd.Series()).max()
        min_cttr_before = self.df.get('cttr_before', pd.Series()).min()
        mean_cttr_after = self.df.get('cttr_after', pd.Series()).mean()
        max_cttr_after = self.df.get('cttr_after', pd.Series()).max()
        min_cttr_after = self.df.get('cttr_after', pd.Series()).min()

        return {
            "total": self.stats["total"],
            "cleaned": self.stats["cleaned"],
            "too_short": self.stats["too_short"],
            "duplicates": self.stats["duplicates"],
            "toxic": self.stats["toxic"],
            "low_quality": self.stats["low_quality"],
            "errors": self.stats["errors"],
            "duplica_before_ratio": duplica_before_ratio,
            "duplica_after_ratio": duplica_after_ratio,
            "mean_toxic_before":mean_toxic_before,
            "max_toxic_before":max_toxic_before,
            "min_toxic_before":min_toxic_before,
            "mean_toxic_after":mean_toxic_after,
            "max_toxic_after":max_toxic_after,
            "min_toxic_after":min_toxic_after,
            "mean_ppl_before":mean_ppl_before,
            "max_ppl_before":max_ppl_before,
            "min_ppl_before":min_ppl_before,
            "mean_ppl_after":mean_ppl_after,
            "max_ppl_after":max_ppl_after,
            "min_ppl_after":min_ppl_after,
            "mean_cttr_before":mean_cttr_before,
            "max_cttr_before":max_cttr_before,
            "min_cttr_before":min_cttr_before,
            "mean_cttr_after":mean_cttr_after,
            "max_cttr_after":max_cttr_after,
            "min_cttr_after":min_cttr_after,
        }
    

    def plot_histogram(self, title="Distribution",threshold=None, xlabel="Value", ylabel="Frequency", bins=50, color='skyblue', kde=True):
        """
        绘制数值数组的直方图（带 KDE）
        
        Args:
            title: 图表标题
            xlabel: x 轴标签
            ylabel: y 轴标签
            bins: 直方图柱子数量
            color: 颜色
            kde: 是否绘制核密度估计曲线
        """

        data = self.df[title].tolist()
        if not data or len(data) == 0:
            print("数据为空，无法绘图")
            return

        # 转为 numpy 数组
        data = np.array(data)
        data = data[~np.isnan(data)]  # 剔除 NaN
        
        # 过滤阈值、修改标题
        if threshold:
            data = data[data <= threshold]
            title += '_after'
        else:
            title += '_before'

        if len(data) == 0:
            print("数据全为 NaN，无法绘图")
            return

        # 绘图
        plt.figure(figsize=(10, 6))
        sns.histplot(data, bins=bins, kde=kde, color=color, alpha=0.7)

        # 添加均值和中位数线
        mean_val = data.mean()
        median_val = np.median(data)
        plt.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
        plt.axvline(median_val, color='orange', linestyle='-', label=f'Median: {median_val:.2f}')

        # 添加峰值
        # 使用核密度估计 (KDE) 找到峰值（密度最大处）
        kde = gaussian_kde(data)
        x_range = np.linspace(data.min(), data.max(), 1000)
        y_kde = kde(x_range)
        peak_val = x_range[np.argmax(y_kde)]  # 密度最高的 x 值

        # 添加峰值线
        peak_x = x_range[np.argmax(y_kde)]
        plt.axvline(peak_val, color='blue', linestyle='-', linewidth=2, label=f'Peak (KDE): {peak_val:.2f}')

        # 添加文本标注（可选）
        plt.annotate(f'Peak\n{peak_x:.2f}', xy=(peak_x, 0), xytext=(peak_x, 0.5),
        arrowprops=dict(arrowstyle='->', color='blue', lw=1.5),
        fontsize=9, ha='center', va='bottom', color='blue')

        # 标签和标题
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        # plt.show()
        plt.savefig(title, dpi=150, bbox_inches='tight')
        plt.close()


    def plot_funnel(self,x='values', y='steps', title="Data Filtering Funnel Chart"):
        """
        构建漏斗数据
        """
        funnel_data = {
            "steps":[],
            "values":[]
        }
        # 储存上一轮的value
        tmp = self.stats['total']

        for key,value in self.stats.items():
            funnel_data['steps'].append(key)
            if key == 'total':
                funnel_data['values'].append(value)
            
            elif key == 'cleaned': 
                funnel_data['values'].append(value)
                break
            else:
                funnel_data['values'].append(tmp-value)
                tmp -= value

        # 使用 Plotly 绘制漏斗图
        fig = px.funnel(
            funnel_data,
            x=x,           # 动态字段名，默认 'values'
            y=y,           # 动态字段名，默认 'steps'
            title=title,
            labels={x: "quantity", y: "steps"},
            template="plotly_white"
        )

        # 设置文字显示在内部
        fig.update_traces(textposition='inside')
        fig.update_layout(font=dict(size=14), title_font_size=18, showlegend=False)

        # fig.show()
        fig.write_html("chart.html") 
        print("✅ 图表已保存为 chart.html，可用浏览器打开")

        # # 保存为图片（需要 kaleido）
        # try:
        #     fig.write_image(f"{title}.png", scale=2, width=800, height=500)
        #     print(f"✅ 图表已保存为: {title}.png")
        # except Exception as e:
        #     print(f"❌ 无法保存图片，请安装: pip install kaleido\n错误: {e}")

        # return fig


    def save_samples_by_indices(self, target, input_file, output_file, threshold=None):
        """
        抽样验证：支持点阈值和区间阈值

        Args:
            target (str): 指标名，如 'toxic_after'
            input_file (str): 原始数据文件路径（jsonl）
            output_file (str): 输出文件路径
            threshold (float or tuple or None):
                - float: x > threshold
                - tuple of two elements: (low, high), 表示 low < x <= high
                - None: 使用 self.max_toxic 作为下限（> self.max_toxic）
        """
        if not hasattr(self, 'df') or self.df is None:
            raise AttributeError("请确保 self.df 已初始化")

        # 确定过滤条件
        series = self.df[target]

        if isinstance(threshold, (int, float)):
            # 单一阈值：x > threshold
            target_indices = set(self.df[series > threshold]['idx'])

        elif isinstance(threshold, (list, tuple)) and len(threshold) == 2:
            low, high = threshold
            if low is None and high is not None:
                # x <= high
                target_indices = set(self.df[series <= high]['idx'])
            elif high is None and low is not None:
                # x > low
                target_indices = set(self.df[series > low]['idx'])
            else:
                # low < x <= high
                target_indices = set(self.df[(series > low) & (series <= high)]['idx'])
        else:
            raise ValueError("threshold 必须是 float、(low, high) 元组，或 None")

        if not target_indices:
            print(f"没有满足条件的样本（{target} 区间过滤）")
            return

        # 保存抽样内容
        count = 0
        with open(input_file, 'r', encoding='utf-8') as fin, \
             open(output_file, 'w', encoding='utf-8') as fout:

            for line_no, line in enumerate(fin):
                if line_no in target_indices:
                    try:                        
                        sample = json.loads(line.strip())
                        sample = {"idx": line_no, **sample}  # 将 idx 放在最前面
                        fout.write(json.dumps(sample, ensure_ascii=False) + '\n')
                        count += 1
                    except Exception as e:
                        print(f"跳过第 {line_no} 行（解析错误）: {e}")

        print(f"已将 {count} 个的样本保存到: {output_file}")



def save_json(data, save_path):
    """
    保存处理结果，自动适配 data 是 list 或 dict 的情况
    
    Args:
        data (list or dict): 要保存的数据
        save_path (str): 保存路径，推荐使用 .jsonl（多条）或 .json（单条）
    """
    # 判断输出目录是否存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    if isinstance(data, list):
        # 情况 1: data 是列表，保存为 .jsonl 格式（每行一个 JSON）
        with open(save_path, 'w', encoding='utf-8') as f:
            for item in data:
                if isinstance(item, dict):
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
                else:
                    raise ValueError(f"List 中的元素必须是 dict，但得到: {type(item)}")
        print(f"列表数据已保存为 JSON Lines 格式：{save_path}")

    elif isinstance(data, dict):
        # 情况 2: data 是单个字典
        _, ext = os.path.splitext(save_path)
        if ext.lower() == '.json':
            # 保存为标准 JSON 文件
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"字典数据已保存为 JSON 文件：{save_path}")
        else:
            # 保存为单行 .jsonl（兼容）
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(json.dumps(data, ensure_ascii=False) + '\n')
            print(f"字典数据已保存为单行 JSONL 格式：{save_path}")

    else:
        raise TypeError(f"data 必须是 list 或 dict，但得到: {type(data)}")