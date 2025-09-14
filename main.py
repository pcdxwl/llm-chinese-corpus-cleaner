import json
from utils import dataClean,Deduplicator,toxicityFilter,qualityFilter,qualityEval,cttr, save_json
from config import max_toxic,min_ppl,max_ppl,threshold,max_words,num_perm,sample

def data_pipeline(input_path: str, output_path: str,detail_path,stats_path):
    """"
    数据集处理pipeline
    """

    # 初始化模型
    dedup = Deduplicator(threshold,max_words,num_perm)
    toxicity_filter = toxicityFilter('./multilingual-toxic-xlm-roberta')
    quality_filter = qualityFilter('./gpt2-chinese-cluecorpussmall')
    
    # 清洗结果统计
    stats = {
        "total": 0,
        "too_short": 0,
        "duplicates": 0,
        "toxic": 0,
        "low_quality": 0,
        "errors": 0,
        "cleaned": 0
    }

    detail = []
    tmp = {}

    # 加载jsonl 文件
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:

        for idx, line in enumerate(fin):
            if idx < 20000:
                # 保存标识
                flag = 1
                print(f'正在处理第{idx}条数据')
                tmp['idx'] = idx
                stats["total"] += 1
                try:
                    item = json.loads(line.strip())
                    tmp.update(item)

                    # 0. 处理前质量评估
                    score = cttr(" \n ".join([item.get('instruction'), item.get('input'), item.get('output')]))
                    tmp['cttr_before'] = score

                    # 1. 清理：不符合条件会返回空
                    print('clean')
                    data_clean = dataClean()
                    item["instruction"] = data_clean.clean_text(item.get("instruction", ""))
                    item["input"] = data_clean.clean_text(item.get("input", ""))
                    item["output"] = data_clean.clean_text(item.get("output", ""))
                    if data_clean.is_short(item):
                        stats["too_short"] += 1
                        # continue
                        flag = 0
                    
                    instruction = item["instruction"] if item["instruction"] else ''
                    input = item["input"] if item["input"] else ''
                    output = item["output"] if item["output"] else ''


                    # 2. 去重（基于 instruction + output）
                    print('去重')
                    if dedup.deduplicate_text(" \n ".join([instruction, output])):
                        stats["duplicates"] += 1
                        # continue
                        flag = 0

                    # 3. 过滤
                    # 3.1 毒性检测（仅 instruction + output）
                    print('毒性检测')
                    score = toxicity_filter.is_toxic(" \n ".join([instruction, output]))
                    tmp["toxic"]=score 
                    if score >= max_toxic:
                        stats["toxic"] += 1
                        # continue
                        flag = 0


                    # 3.2 困惑度（仅 output）
                    print('困惑度检测')
                    score = quality_filter.is_low_quality(" \n ".join([instruction, output]))
                    tmp["ppl"]= score
                    if score >= max_ppl:
                        stats["low_quality"] += 1
                        # continue
                        flag = 0

                    # 4 处理后质量评估
                    print('质量评估')
                    score = cttr(" \n ".join([instruction, input, output]))
                    tmp['cttr_after']=score

                    # 通过所有检查，保存
                    if flag == 1:
                        fout.write(json.dumps(item, ensure_ascii=False) + "\n")
                        stats["cleaned"] += 1
                    detail.append(tmp.copy())
                                        
                    
                except Exception as e:
                    stats["errors"] += 1
                    print(f"处理第 {stats['total']} 行失败: {e}")  

    # 保存统计结果
    save_json(detail,detail_path)
    save_json(stats,stats_path)
    return detail,stats

def result_check(detail_path,stats_path,input_path):
    """
    从文件中加载 detail 和 stats，验证处理结果，判断阈值是否需要修改
    
    Args:
        detail_path (str): detail.jsonl 文件路径
        stats_path (str): stats.json 文件路径
    """

    quality_eval = qualityEval(max_toxic,max_ppl,detail_path,stats_path)  
    # 统计结果
    stats = quality_eval.get_eval_result()
    save_json(stats,stats_path)
    # 抽查样本
    for line in sample:
        quality_eval.save_samples_by_indices(line['target'], input_path, line['path'],line['threshold'])
    # # 画图
    for line in sample:
        # before的图
        quality_eval.plot_histogram(line['target'])
        # after的图
        quality_eval.plot_histogram(line['target'],line['threshold'])

if __name__ == '__main__':
    input_path = "./train_0.5M_CN/Belle_open_source_0.5M.json"
    output_path = "./train_0.5M_CN/Belle_open_source_0.5M_clean.jsonl"
    detail_path = './detail.json'
    stats_path = './stats.json'
    # # # 数据处理
    # detail,stats = data_pipeline(input_path, output_path,detail_path,stats_path)
    # # # 结果验证
    # # result_check(detail_path,stats_path,input_path)


    
    quality_eval = qualityEval(max_toxic,max_ppl,detail_path,stats_path)
    quality_eval.plot_funnel()