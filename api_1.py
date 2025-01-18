from flask import Flask, request, jsonify, send_from_directory
from bot import llm_chain, rag_chain
import os
from langchain.callbacks.base import BaseCallbackHandler

app = Flask(__name__)

class ChatCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.response = ""

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.response += token

# 服务静态文件
@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('.', path)

# 聊天API
@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    question = data.get('question')
    chat_history = data.get('chat_history', [])
    rag_enabled = request.args.get('rag') == 'enabled'
    
    try:
        # 创建回调处理器
        callback_handler = ChatCallbackHandler()
        
        # 根据RAG模式选择不同的chain
        chain = rag_chain if rag_enabled else llm_chain
        
        # 调用chain处理问题
        result = chain({
            "question": question, 
            "chat_history": chat_history
        }, callbacks=[callback_handler])
        
        # 如果result中没有answer，使用callback_handler中收集的响应
        answer = result.get("answer", callback_handler.response)
        
        return jsonify({
            "success": True,
            "answer": answer
        })
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")  # 添加错误日志
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True) 