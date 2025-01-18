import os
import re

from collections import Counter
import streamlit as st
from PyPDF2 import PdfReader
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from streamlit import button

from chains import load_llm
from dotenv import load_dotenv
from sqlite import initialize_database
from sqlite import delete_newwords
from sqlite import exec_insert
from sqlite import exec_query
import nltk
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet
import time
from streamlit_modal import Modal
import requests
from streamlit_autorefresh import st_autorefresh

load_dotenv(".env")
ollama_base_url = os.getenv("OLLAMA_BASE_URL")
llm_name = os.getenv("LLM")


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text
        self.complete_text = ""

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.complete_text += token

    def get_text(self):
        return self.complete_text


llm = load_llm(llm_name, config={"ollama_base_url": ollama_base_url})
memory = ConversationBufferMemory(return_messages=True)
conversation = ConversationChain(llm=llm,memory=memory)
st.set_page_config(page_title="PDF词汇学习助手", layout="wide")
initialize_database()
def ReadLinesAsList(file):
    with open(file,'r') as f:
        lines = f.readlines()
        vocab = {word.strip().split()[0].lower()
                 for word in lines if word.strip()}

        return vocab

# 预加载词汇表
@st.cache_data
def load_vocab_tables():
    """预加载CET6和COCA词汇表"""
    # CET6词表
    cet6_url = "https://raw.githubusercontent.com/mahavivo/english-wordlists/master/CET_4%2B6_edited.txt"
    coca_url = "https://raw.githubusercontent.com/mahavivo/english-wordlists/master/COCA_20000.txt"

    cet6_path = "/app/CET_4_6_edited.txt"
    coca_path = "/app/COCA_20000.txt"

    cet6_vocab = ReadLinesAsList(cet6_path)
    coca_vocab = ReadLinesAsList(coca_path)

    return cet6_vocab ,coca_vocab

# 全局变量存储词汇表
CET6_VOCAB, COCA_VOCAB = load_vocab_tables()
wnl = WordNetLemmatizer()

def get_word_category(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.VERB

def is_valid_word(word):
    """检查是否为有效单词"""
    word = word.lower()
    # 过滤专有名词和非COCA词表单词
    if not word in COCA_VOCAB or not word.isalpha():
        # 必须全是字母
        return False

    return True


def extract_words(text):
    """提取并处理文本中的单词"""
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    valid_words = []
    for word in words:
        if len(word) > 1:  # 过滤单字母
            # 词形还原
            if is_valid_word(word):
                valid_words.append(wnl.lemmatize(word, get_word_category(pos_tag([word])[0][1])))
    return valid_words


def display_word_card(word, explanation, freq, index, total):
    """显示单词卡片"""
    st.markdown("""
    <style>
    .word-card {
        padding: 20px;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

    with st.container():
        st.markdown(f"<div class='word-card'>", unsafe_allow_html=True)
        st.markdown(f"### {word}")
        st.caption(f"在文档中出现 {freq} 次")
        st.markdown(explanation)
        st.caption(f"卡片 {index + 1}/{total}")
        st.markdown("</div>", unsafe_allow_html=True)


def get_word_explanations(words, llm):
    """使用LLM生成单词释义"""
    explanations = {}
    prompt_template = """请为以下英语单词提供简明的中文释义和一个简短的例句:
    单词: {word}
    要求:
    1. 给出最常用的1-2个含义
    2. 提供一个简单的例句
    3. 输出格式:
       释义: [中文释义]
       例句: [英文例句]"""

    for word in words:
        container = st.empty()
        stream_handler = StreamHandler(container)
        prompt = prompt_template.format(word=word)
        llm.predict(prompt, callbacks=[stream_handler])
        explanations[word] = stream_handler.get_text()
        container.empty()  # 清除临时显示
    return explanations


def main():


    if "explanations" not in st.session_state:
        st.session_state.explanations = {}
    if "word_freq" not in st.session_state:
        st.session_state.word_freq = None
    if "unknown_words_list" not in st.session_state:
        st.session_state.unknown_words_list = []


    st.title("📚 PDF词汇学习助手")
    st.markdown("""
    这是一个帮助你提升英语词汇量的工具:
    1. 上传英语PDF文档
    2. 选择你的词汇水平
    3. 获取超出你词汇水平的单词解释和例句
    """)

    # 侧边栏配置

    # 主界面
    pdf = st.file_uploader("上传英语PDF文档", type="pdf")

    if pdf is not None:
        with st.spinner('正在分析文档...'):
            # 读取PDF
            pdf_reader = PdfReader(pdf)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()

            # 分析词汇
            words = extract_words(text)
            st.session_state.word_freq = Counter(words)
            unknown_words = {word for word in st.session_state.word_freq.keys()
                             if word not in CET6_VOCAB}

            if unknown_words:
                # 仅在首次上传PDF时生成释义
                if not st.session_state.explanations:
                    with st.spinner('正在生成单词释义...'):
                        st.session_state.explanations = get_word_explanations(unknown_words, llm)
                    st.session_state.unknown_words_list = sorted(unknown_words)


                for word in unknown_words:

                    insert_sql = "INSERT INTO new_words (word , explanations,insert_time) VALUES (?,?,?)"
                    cur_time = int(time.time()*1000)
                    res=exec_insert(insert_sql,(word,st.session_state.explanations[word] ,cur_time))
                    if res:
                        print("INSERT ERROR:{}".format(res))


                # 导航按钮
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("⬅️ 上一个"):
                        st.session_state.current_word_index = (st.session_state.current_word_index - 1) % len(
                            st.session_state.unknown_words_list)
                with col2:
                    if st.button("➡️ 下一个"):
                        st.session_state.current_word_index = (st.session_state.current_word_index + 1) % len(
                            st.session_state.unknown_words_list)
                with col3:
                    if st.button("🔄 重置"):
                        st.session_state.current_word_index = 0

                # 从缓存中获取并显示当前单词卡片
                current_word = st.session_state.unknown_words_list[st.session_state.current_word_index]
                display_word_card(
                    current_word,
                    st.session_state.explanations[current_word],
                    st.session_state.word_freq[current_word],
                    st.session_state.current_word_index,
                    len(st.session_state.unknown_words_list)
                )

def recall_new_words():
    num_words = st.session_state.num_new_words
    st.header("需要复习的生词:{}".format(num_words))

    query_sql = "SELECT word , explanations FROM new_words ORDER BY insert_time LIMIT ?"

    res = exec_query(query_sql,(num_words,))
    new_words=[]
    explanations={}

    for row in res:
        new_words.append(row[0])
        explanations[row[0]]=row[1]

    print("需要复习的单词：")
    print(new_words)
    print(explanations)
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("⬅️ 上一个"):
            st.session_state.current_word_index = (st.session_state.current_word_index - 1) % len(
                new_words)
    with col2:
        if st.button("➡️ 下一个"):
            print("下一个:{}".format(st.session_state.current_word_index))
            st.session_state.current_word_index = (st.session_state.current_word_index + 1) % len(
               new_words)
    with col3:
        if st.button("🔄 重置"):
            st.session_state.current_word_index = 0

    # 从缓存中获取并显示当前单词卡片
    current_word = new_words[st.session_state.current_word_index]
    display_word_card(
        current_word,
        explanations[current_word],
        1,
        st.session_state.current_word_index,
        len(new_words)
    )




if __name__ == "__main__":

   if "page" not in st.session_state:
       st.session_state.page = "PDF"
   if "num_new_words" not in st.session_state:
        st.session_state.num_new_words=0
   if "chat_history" not in st.session_state:
       st.session_state["chat_history"] = []
   if "current_word_index" not in st.session_state:
       st.session_state.current_word_index = 0

   interval=1
   main_container = st.empty()
   modal = Modal(key="Demo Key",title="警告")
   page = st.selectbox("选择功能",["PDF" ,"复习单词"])

   if page== "PDF":
       st.session_state.page="PDF"
   elif page=="复习单词":
       if st.session_state.num_new_words > 0:
          
            st.session_state.page="recall_words"
   with st.sidebar:
       st.header("设置")
       level = st.selectbox(
           "选择你的词汇水平",
           ["CET6"],
           help="目前仅支持CET6水平检测"
       )


       num_words = int(st.number_input("输入要复习的生词数量"))
       st.session_state.num_new_words = num_words

       user_input = st.text_input("单词小助手")

       if user_input:
           st.session_state["chat_history"].append(f"用户输入:{user_input}")

           response = conversation.predict(input=user_input)

           st.session_state["chat_history"].append(f"AI助手:{response}")

       for message in st.session_state["chat_history"]:
            st.write(message)

   if st.session_state.page== "PDF":
       main()
   elif st.session_state.page== "recall_words":
       recall_new_words()
