import os
import sys
# import logging
# from accelerate import dispatch_model
# from accelerate import infer_auto_device_map
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer,pipeline,AutoModel
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import HuggingFacePipeline
# from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from transformers import TextStreamer
from langchain.chains import LLMChain
import fitz, copy
import re
import numpy as np
from langchain.vectorstores import Milvus
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.docstore.document import Document
import whisper

#--------------Load whisper model----------------------

video_model=whisper.load_model('/home/username/kechengbaogao/model/large-v1.pt')
language = 'Chinese'
video_path="/home/username/kechengbaogao/video.mp4"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
#--------------生成结果txt文件---------------------
dir_path=os.path.dirname(os.path.abspath(__file__))
file_path = "/home/username/kechengbaogao/ssci47803_2020_9308468.pdf"
video_summary_result_path='video_summary.txt'
video_translate_result_path='video_translate.txt'
video_rag_result_path='video_rag.txt'

pdf_summary_result_path='pdf_summary.txt'
pdf_translate_result_path='pdf_translate.txt'
pdf_rag_result_path='pdf_rag.txt'

video_summary_file = open(os.path.join(dir_path,video_summary_result_path),'w')
video_translate_file = open(os.path.join(dir_path,video_translate_result_path),'w')
video_rag_file = open(os.path.join(dir_path,video_rag_result_path),'w')
#---------------------------------------------------------------
pdf_summary_file = open(os.path.join(dir_path,pdf_summary_result_path),'w')
pdf_translate_file = open(os.path.join(dir_path,pdf_translate_result_path),'w')
pdf_rag_file = open(os.path.join(dir_path,pdf_rag_result_path),'w')

#----------------Load model------------------------


#------------------chatglm3----------------------------
role = "user"
model_path = "/home/username/kechengbaogao/model/chatglm3-6b"

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
print("---------------Tokenizer loaded--------------")
if 'cuda' in DEVICE: # AMD, NVIDIA GPU can use Half Precision
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(DEVICE).eval()
else: # CPU, Intel GPU and other GPU can use Float16 Precision Only
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True).float().to(DEVICE).eval()

print("---------------LLM loaded-------------")
#----------------------load  embedding model-------------------
#为实现中英文embedding，加载不同的模型
device_type = "cuda"
#加载embedding模型
EMBEDDING_MODEL_NAME_EN = "/home/username/kechengbaogao/model/bge-base-en"
EMBEDDING_MODEL_NAME_ZH = "/home/username/kechengbaogao/model/bge-base-zh"

embeddings_en = HuggingFaceInstructEmbeddings(
    model_name=EMBEDDING_MODEL_NAME_EN,
    model_kwargs={"device": device_type},
)
embeddings_zh = HuggingFaceInstructEmbeddings(
    model_name=EMBEDDING_MODEL_NAME_ZH,
    model_kwargs={"device": device_type},
)

#--------------------------------------------------------
def read_and_clean_pdf_text(fp):
    """
    这个函数用于分割pdf，用了很多trick，逻辑较乱，效果奇好

    **输入参数说明**
    - `fp`：需要读取和清理文本的pdf文件路径

    **输出参数说明**
    - `meta_txt`：清理后的文本内容字符串
    - `page_one_meta`：第一页清理后的文本内容列表

    **函数功能**
    读取pdf文件并清理其中的文本内容，清理规则包括：
    - 提取所有块元的文本信息，并合并为一个字符串
    - 去除短块（字符数小于100）并替换为回车符
    - 清理多余的空行
    - 合并小写字母开头的段落块并替换为空格
    - 清除重复的换行
    - 将每个换行符替换为两个换行符，使每个段落之间有两个换行符分隔
    """
    fc = 0  # Index 0 文本
    fs = 1  # Index 1 字体
    fb = 2  # Index 2 框框
    REMOVE_FOOT_NOTE = True # 是否丢弃掉 不是正文的内容 （比正文字体小，如参考文献、脚注、图注等）
    REMOVE_FOOT_FFSIZE_PERCENT = 0.95 # 小于正文的？时，判定为不是正文（有些文章的正文部分字体大小不是100%统一的，有肉眼不可见的小变化）
    def primary_ffsize(l):
        """
        提取文本块主字体
        """
        fsize_statiscs = {}
        for wtf in l['spans']:
            if wtf['size'] not in fsize_statiscs: fsize_statiscs[wtf['size']] = 0
            fsize_statiscs[wtf['size']] += len(wtf['text'])
        return max(fsize_statiscs, key=fsize_statiscs.get)
        
    def ffsize_same(a,b):
        """
        提取字体大小是否近似相等
        """
        return abs((a-b)/max(a,b)) < 0.02

    with fitz.open(fp) as doc:
        meta_txt = []
        meta_font = []

        meta_line = []
        meta_span = []
        ############################## <第 1 步，搜集初始信息> ##################################
        for index, page in enumerate(doc):
            # file_content += page.get_text()
            text_areas = page.get_text("dict")  # 获取页面上的文本信息
            for t in text_areas['blocks']:
                if 'lines' in t:
                    pf = 998
                    for l in t['lines']:
                        txt_line = "".join([wtf['text'] for wtf in l['spans']])
                        if len(txt_line) == 0: continue
                        pf = primary_ffsize(l)
                        meta_line.append([txt_line, pf, l['bbox'], l])
                        for wtf in l['spans']: # for l in t['lines']:
                            meta_span.append([wtf['text'], wtf['size'], len(wtf['text'])])
                    # meta_line.append(["NEW_BLOCK", pf])
            # 块元提取                           for each word segment with in line                       for each line         cross-line words                          for each block
            meta_txt.extend([" ".join(["".join([wtf['text'] for wtf in l['spans']]) for l in t['lines']]).replace(
                '- ', '') for t in text_areas['blocks'] if 'lines' in t])
            meta_font.extend([np.mean([np.mean([wtf['size'] for wtf in l['spans']])
                             for l in t['lines']]) for t in text_areas['blocks'] if 'lines' in t])
            if index == 0:
                page_one_meta = [" ".join(["".join([wtf['text'] for wtf in l['spans']]) for l in t['lines']]).replace(
                    '- ', '') for t in text_areas['blocks'] if 'lines' in t]
                
        ############################## <第 2 步，获取正文主字体> ##################################
        try:
            fsize_statiscs = {}
            for span in meta_span:
                if span[1] not in fsize_statiscs: fsize_statiscs[span[1]] = 0
                fsize_statiscs[span[1]] += span[2]
            main_fsize = max(fsize_statiscs, key=fsize_statiscs.get)
            if REMOVE_FOOT_NOTE:
                give_up_fize_threshold = main_fsize * REMOVE_FOOT_FFSIZE_PERCENT
        except:
            raise RuntimeError(f'抱歉, 我们暂时无法解析此PDF文档: {fp}。')
        ############################## <第 3 步，切分和重新整合> ##################################
        mega_sec = []
        sec = []
        for index, line in enumerate(meta_line):
            if index == 0: 
                sec.append(line[fc])
                continue
            if REMOVE_FOOT_NOTE:
                if meta_line[index][fs] <= give_up_fize_threshold:
                    continue
            if ffsize_same(meta_line[index][fs], meta_line[index-1][fs]):
                # 尝试识别段落
                if meta_line[index][fc].endswith('.') and\
                    (meta_line[index-1][fc] != 'NEW_BLOCK') and \
                    (meta_line[index][fb][2] - meta_line[index][fb][0]) < (meta_line[index-1][fb][2] - meta_line[index-1][fb][0]) * 0.7:
                    sec[-1] += line[fc]
                    sec[-1] += "\n\n"
                else:
                    sec[-1] += " "
                    sec[-1] += line[fc]
            else:
                if (index+1 < len(meta_line)) and \
                    meta_line[index][fs] > main_fsize:
                    # 单行 + 字体大
                    mega_sec.append(copy.deepcopy(sec))
                    sec = []
                    sec.append("# " + line[fc])
                else:
                    # 尝试识别section
                    if meta_line[index-1][fs] > meta_line[index][fs]:
                        sec.append("\n" + line[fc])
                    else:
                        sec.append(line[fc])
        mega_sec.append(copy.deepcopy(sec))

        finals = []
        for ms in mega_sec:
            final = " ".join(ms)
            final = final.replace('- ', ' ')
            finals.append(final)
        meta_txt = finals

        ############################## <第 4 步，乱七八糟的后处理> ##################################
        def 把字符太少的块清除为回车(meta_txt):
            for index, block_txt in enumerate(meta_txt):
                if len(block_txt) < 100:
                    meta_txt[index] = '\n'
            return meta_txt
        meta_txt = 把字符太少的块清除为回车(meta_txt)

        def 清理多余的空行(meta_txt):
            for index in reversed(range(1, len(meta_txt))):
                if meta_txt[index] == '\n' and meta_txt[index-1] == '\n':
                    meta_txt.pop(index)
            return meta_txt
        meta_txt = 清理多余的空行(meta_txt)

        def 合并小写开头的段落块(meta_txt):
            def starts_with_lowercase_word(s):
                pattern = r"^[a-z]+"
                match = re.match(pattern, s)
                if match:
                    return True
                else:
                    return False
            for _ in range(100):
                for index, block_txt in enumerate(meta_txt):
                    if starts_with_lowercase_word(block_txt):
                        if meta_txt[index-1] != '\n':
                            meta_txt[index-1] += ' '
                        else:
                            meta_txt[index-1] = ''
                        meta_txt[index-1] += meta_txt[index]
                        meta_txt[index] = '\n'
            return meta_txt
        meta_txt = 合并小写开头的段落块(meta_txt)
        meta_txt = 清理多余的空行(meta_txt)

        meta_txt = '\n'.join(meta_txt)
        # 清除重复的换行
        for _ in range(5):
            meta_txt = meta_txt.replace('\n\n', '\n')

        # 换行 -> 双换行
        meta_txt = meta_txt.replace('\n', '\n\n')

    return meta_txt, page_one_meta

def force_breakdown(txt, limit, get_token_fn):
    """
    当无法用标点、空行分割时，我们用最暴力的方法切割
    """
    for i in reversed(range(len(txt))):
        if get_token_fn(txt[:i]) < limit:
            return txt[:i], txt[i:]
    return "Tiktoken未知错误", "Tiktoken未知错误"

def breakdown_txt_to_satisfy_token_limit_for_pdf(txt, get_token_fn, limit):
    # 递归
    def cut(txt_tocut, must_break_at_empty_line, break_anyway=False):  
        if get_token_fn(txt_tocut) <= limit:
            return [txt_tocut]
        else:
            lines = txt_tocut.split('\n')
            estimated_line_cut = limit / get_token_fn(txt_tocut) * len(lines)
            estimated_line_cut = int(estimated_line_cut)
            cnt = 0
            for cnt in reversed(range(estimated_line_cut)):
                if must_break_at_empty_line:
                    if lines[cnt] != "":
                        continue
                prev = "\n".join(lines[:cnt])
                post = "\n".join(lines[cnt:])
                if get_token_fn(prev) < limit:
                    break
            if cnt == 0:
                if break_anyway:
                    prev, post = force_breakdown(txt_tocut, limit, get_token_fn)
                else:
                    raise RuntimeError(f"存在一行极长的文本！{txt_tocut}")
            # print(len(post))
            # 列表递归接龙
            result = [prev]
            result.extend(cut(post, must_break_at_empty_line, break_anyway=break_anyway))
            return result
    try:
        # 第1次尝试，将双空行（\n\n）作为切分点
        return cut(txt, must_break_at_empty_line=True)
    except RuntimeError:
        try:
            # 第2次尝试，将单空行（\n）作为切分点
            return cut(txt, must_break_at_empty_line=False)
        except RuntimeError:
            try:
                # 第3次尝试，将英文句号（.）作为切分点
                res = cut(txt.replace('.', '。\n'), must_break_at_empty_line=False) # 这个中文的句号是故意的，作为一个标识而存在
                return [r.replace('。\n', '.') for r in res]
            except RuntimeError as e:
                try:
                    # 第4次尝试，将中文句号（。）作为切分点
                    res = cut(txt.replace('。', '。。\n'), must_break_at_empty_line=False)
                    return [r.replace('。。\n', '。') for r in res]
                except RuntimeError as e:
                    # 第5次尝试，没办法了，随便切一下敷衍吧
                    return cut(txt, must_break_at_empty_line=False, break_anyway=True)

def get_token_num(txt): return len(tokenizer.encode(txt))


#-----------------按章节切割pdf----------------------------
file_content, page_one = read_and_clean_pdf_text(file_path) # （尝试）按照章节切割PDF
file_content = file_content.encode('utf-8', 'ignore').decode()   # avoid reading non-utf8 chars
#-----------------章节内分段以满足token限制-------------------------
TOKEN_LIMIT_PER_FRAGMENT = 1500

paper_fragments = breakdown_txt_to_satisfy_token_limit_for_pdf(
    txt=file_content,  get_token_fn=get_token_num, limit=TOKEN_LIMIT_PER_FRAGMENT)

summary_history=[]
translate_history=[]
translate_sum=""
docs_en=[]
docs_zh=[]
# def build_prompt(history):
#     prompt = "欢迎使用 ChatGLM3-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序"
#     for query, response in history:
#         prompt += f"\n\n用户：{query}"
#         prompt += f"\n\nChatGLM3-6B：{response}"
#     return prompt
for index in range(len(paper_fragments)):
    #翻译一般不需要上下文信息，每次都清空，减小占用
    translate_history=[]
    print(str(index)+"/"+str(len(paper_fragments)))
    prefix = "接下来请你逐段分析下面的论文，概括其内容" if index == 0 else ""
    prefix_translate="接下来请你逐段翻译下面的论文，将英文翻译成中文" if index == 0 else ""
    summary_template = (
    prefix+
    "请对下面的文章片段用中文做一个概述，文章内容为:\n{text}"
    )
    summary_query=summary_template.format(text=paper_fragments[index])
    summary_response,summary_history = model.chat(tokenizer, query=summary_query, history=summary_history, role=role)
    # print(summary_response)
    #生成document存入向量数据库
    new_doc = Document(page_content=paper_fragments[index],metadata={})
    docs_en.append(new_doc)
    #----------------pdf翻译------------
    translate_template = (
    prefix_translate+"\n"
    "请将下面的文章片段翻译成中文，注意仅对文章内容做翻译，文章内容为:\n{text}"
    )
    translate_query=translate_template.format(text=paper_fragments[index])
    translate_response,translate_history = model.chat(tokenizer, query=translate_query, history=translate_history, role=role)
    # print(translate_response)
    #生成document存入向量数据库
    translate_doc=Document(page_content=translate_response,metadata={})
    translate_sum+=translate_response
    docs_zh.append(translate_doc)
print("----------------------summary----------------")
pdf_translate_file.write(translate_sum)
query_summary="根据以上你自己的分析，对全文进行概括，用学术性语言写一段中文摘要"
sum_resp,history_=model.chat(tokenizer, query=query_summary, history=summary_history, role=role)
pdf_summary_file.write(sum_resp)

#----------------PDF-RAG--------------------------

#----------------相似度查询------------------------

text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100, separators=[" ", ",", "\n"])
pdf_rag_history=[]
pdf_query = "零式转移是什么"
query_type="zh"          #默认询问问题为中文，可以把参数修改为en则为英文数据库
if query_type=="zh":
    db_zh = Milvus.from_documents(
        docs_zh,
        embeddings_zh,
        collection_name="pdf_zh_rag",
        connection_args={"host": "10.23.15.111", "port": "19530"},
    )
    db_zh.add_documents(docs_zh)
    docs_sim = db_zh.similarity_search(pdf_query,k=3)
    docs_sim_ = [doc.page_content for doc in docs_sim]
    # format the prompt
    docs_sim_ = "\n\n".join(docs_sim_)
    prompt_rag="根据以下的上下文信息:{docs_sim}使用中文回答以下中文问题。问题: {query} 用中文回答:"
    prompt_formatted = prompt_rag.format(docs_sim=docs_sim_, query=pdf_query)
    # generate answer
    pdf_rag_resp,_=model.chat(tokenizer, query=prompt_formatted, history=pdf_rag_history, role=role)
    pdf_rag_file.write("问题:"+pdf_query+"\n"+"答案:"+pdf_rag_resp)
elif query_type=="en":
    db_en = Milvus.from_documents(
        docs_en,
        embeddings_en,
        collection_name="pdf_en_rag",
        connection_args={"host": "10.23.15.111", "port": "19530"},
    )
    db_en.add_documents(docs_en)
    docs_sim = db_en.similarity_search(pdf_query,k=3)
    docs_sim_ = [doc.page_content for doc in docs_sim]
    # format the prompt
    docs_sim_ = "\n\n".join(docs_sim_)
    prompt_rag="Based on the following context information:{docs_sim}Answer the following English question in English。Question: {query} Answer in English:"
    prompt_formatted = prompt_rag.format(docs_sim=docs_sim_, query=pdf_query)
    # generate answer
    pdf_rag_resp,_=model.chat(tokenizer, query=prompt_formatted, history=pdf_rag_history, role=role)
    pdf_rag_file.write("query:"+pdf_query+"\n"+"answer:"+pdf_rag_resp)

#-----------------------视频转录----------------------
result = video_model.transcribe(video_path, language=language)
video_translate_file.write(result['text'])
#-------------------------视频总结--------------------------
video_sum_history=[]

prompt_video="请对下面的文本内容用中文做一个概述，文本内容为:{text}"
prompt_video_sum = prompt_video.format(text=result['text'])
# generate answer
video_sum_resp,_=model.chat(tokenizer, query=prompt_video_sum, history=pdf_rag_history, role=role)
video_summary_file.write(video_sum_resp)

#-------------------------视频RAG---------------------------

#-----------------------视频文本分割----------------------
#分割后存入向量数据库
texts = text_splitter.split_text(result['text'])

docs_video = [Document(page_content=t,metadata={}) for t in texts]
#视频文本默认为中文，且问题为中文
video_rag_history=[]
video_query = "熬夜造成的黑眼圈如何淡化?"
if query_type=="zh":
    db_video_zh = Milvus.from_documents(
        docs_video,
        embeddings_zh,
        collection_name="video_zh",
        connection_args={"host": "10.23.15.111", "port": "19530"},
    )
    db_video_zh.add_documents(docs_video)
    docs_video_sim = db_video_zh.similarity_search(video_query,k=3)

    docs_video_sim_ = [doc.page_content for doc in docs_video_sim]

    docs_video_simi = "\n\n".join(docs_video_sim_)
    prompt_rag_="根据以下的上下文信息:{docs_sim}使用中文回答以下中文问题。问题: {query} 用中文回答:"
    prompt_video_rag = prompt_rag_.format(docs_sim=docs_sim_, query=video_query)
    # generate answer
    video_rag_resp,_=model.chat(tokenizer, query=prompt_video_rag, history=video_rag_history, role=role)
    video_rag_file.write("问题:"+video_query+"\n"+"答案:"+video_rag_resp)
elif query_type=="en":
    db_video_en = Milvus.from_documents(
        docs_video,
        embeddings_en,
        collection_name="video_en",
        connection_args={"host": "10.23.15.111", "port": "19530"},
    )
    db_video_en.add_documents(docs_video)
    
    docs_video_sim = db_video_en.similarity_search(video_query,k=3)
    docs_video_sim_ = [doc.page_content for doc in docs_video_sim]

    docs_video_simi = "\n\n".join(docs_video_sim_)
    prompt_rag_="Based on the following context information:{docs_sim}Answer the following English question in English。Question: {query} Answer in English:"
    prompt_video_rag = prompt_rag_.format(docs_sim=docs_sim_, query=video_query)
    # generate answer
    video_rag_resp,_=model.chat(tokenizer, query=prompt_video_rag, history=video_rag_history, role=role)
    video_rag_file.write("query:"+video_query+"\n"+"answer:"+video_rag_resp)
