import os

cet6_path = "CET_4_6_edited.txt"
coca_path = "./COCA_20000.txt"

def ReadLinesAsList(file):
    with open(file,'r') as f:
        lines = f.readlines()
        vocab = {word.strip().split()[0].lower()
                 for word in lines if word.strip()}

        return vocab


print(ReadLinesAsList(cet6_path))

'''
   try:
       # 加载CET6词表
       cet6_response = requests.get(cet6_url)
       cet6_response.raise_for_status()
       cet6_vocab = {word.strip().split()[0].lower()
                     for word in cet6_response.text.splitlines() if word.strip()}

       # 加载COCA词表
       coca_response = requests.get(coca_url)
       coca_response.raise_for_status()
       coca_vocab = {word.strip().split()[0].lower()
                     for word in coca_response.text.splitlines() if word.strip()}

       return cet6_vocab, coca_vocab
   except Exception as e:
       st.error(f"词汇表加载失败: {str(e)}")
       return set(), set()

   '''





