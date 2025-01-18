import os
import re
from collections import Counter
import streamlit as st
from PyPDF2 import PdfReader
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from chains import load_llm
from dotenv import load_dotenv
from sqlite import initialize_database
from sqlite import exec_insert
from sqlite import exec_insert
import nltk
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet
import time
import requests

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


