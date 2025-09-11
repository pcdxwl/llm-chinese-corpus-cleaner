# 输入参数
max_toxic = 0.8
min_ppl = 5
max_ppl = 110
threshold = 0.80
max_words = 200
num_perm = 128

sample = [
    {"target":"toxic","path":'./toxic_auto_sample.jsonl',"threshold":max_toxic},
    {"target":"ppl","path":'./ppl_auto_sample.jsonl',"threshold":max_ppl}
]


# sample = [
#     {"target":"toxic","path":'./toxic_manual_sample.jsonl',"threshold":(0.80,0.85)},
#     {"target":"ppl","path":'./ppl_manual_sample.jsonl',"threshold":(15,16)}
# ]




# cd  llm-chinese-corpus-cleaner     //根据自己的远程仓库名输入
# git init
# git add .
# git commit -m “新建”
# git push