from neo4j import GraphDatabase
import time

class WordWiseDB:
    """WordWise 应用的数据库操作类
    
    该类封装了所有与 Neo4j 数据库的交互操作，包括：
    - 用户管理：创建、查询、更新用户信息
    - 词单管理：创建词单、添加单词
    - 学习追踪：打卡记录、学习统计
    - 数据导入：支持从不同格式文件导入单词数据
    """
    
    def __init__(self, uri, user, password):
        """初始化数据库连接
        
        Args:
            uri (str): Neo4j 数据库连接地址 (例如: "neo4j://localhost:7687")
            user (str): 数据库用户名
            password (str): 数据库密码
        """
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def create_user(self, account, password, email, avatar_url=None):
        """创建新用户节点
        
        Args:
            account (str): 用户账号
            password (str): 用户密码（注意：实际使用时应该加密存储）
            email (str): 用户邮箱
            avatar_url (str, optional): 用户头像URL
            
        Returns:
            str: 新创建用户的唯一ID (uid)
            
        Note:
            创建用户时会初始化以下属性：
            - streak_days: 连续打卡天数
            - total_study_days: 总学习天数
            - theme_color: 默认主题色
            - reminder_time: 默认提醒时间
            - created_at: 账号创建时间
            - last_login: 最后登录时间
        """
        with self.driver.session() as session:
            result = session.run("""
                CREATE (u:User {
                    uid: apoc.create.uuid(),
                    account: $account,
                    password: $password,
                    email: $email,
                    avatar_url: $avatar_url,
                    streak_days: 0,
                    total_study_days: 0,
                    theme_color: '#4A90E2',
                    reminder_time: '08:00',
                    created_at: datetime(),
                    last_login: datetime()
                })
                RETURN u.uid as uid
            """, account=account, password=password, email=email, avatar_url=avatar_url)
            return result.single()["uid"]

    def create_wordlist(self, wid, name, description, owner_uid):
        """创建新的单词列表
        
        Args:
            wid (str): 词单唯一标识符
            name (str): 词单名称
            description (str): 词单描述
            owner_uid (str): 创建者的用户ID
            
        Note:
            会建立用户与词单之间的 OWNS 关系
        """
        with self.driver.session() as session:
            session.run("""
                MATCH (u:User {uid: $owner_uid})
                CREATE (wl:WordList {
                    wid: $wid,
                    name: $name,
                    description: $description
                })
                CREATE (u)-[:OWNS]->(wl)
            """, wid=wid, name=name, description=description, owner_uid=owner_uid)

    def add_word_to_list(self, word_text, wordlist_id):
        """向词单中添加单词
        
        Args:
            word_text (str): 单词文本
            wordlist_id (str): 目标词单ID
            
        Note:
            使用 MERGE 确保不会创建重复的单词节点
            建立词单与单词之间的 CONTAINS 关系
        """
        with self.driver.session() as session:
            session.run("""
                MATCH (wl:WordList {wid: $wid})
                MERGE (w:Word {text: $word_text})
                CREATE (wl)-[:CONTAINS]->(w)
            """, wid=wordlist_id, word_text=word_text)

    def mark_word_as_mastered(self, user_id, word_text):
        """将单词标记为已掌握状态
        
        Args:
            user_id (str): 用户ID
            word_text (str): 单词文本
            
        Note:
            - 创建用户到单词的 MASTERED 关系
            - 移除之前的 LEARNING 关系（如果存在）
        """
        with self.driver.session() as session:
            session.run("""
                MATCH (u:User {uid: $uid}), (w:Word {text: $word_text})
                MERGE (u)-[:MASTERED]->(w)
                REMOVE (u)-[:LEARNING]->(w)
            """, uid=user_id, word_text=word_text)

    def update_user_settings(self, uid, theme_color=None, reminder_time=None, study_goal=None):
        """更新用户设置
        
        Args:
            uid (str): 用户ID
            theme_color (str, optional): 主题颜色（例如：'#4A90E2'）
            reminder_time (str, optional): 提醒时间（格式：'HH:MM'）
            study_goal (int, optional): 每日学习目标
        """
        with self.driver.session() as session:
            update_query = "MATCH (u:User {uid: $uid}) "
            params = {"uid": uid}
            
            if theme_color:
                update_query += "SET u.theme_color = $theme_color "
                params["theme_color"] = theme_color
            if reminder_time:
                update_query += "SET u.reminder_time = $reminder_time "
                params["reminder_time"] = reminder_time
            if study_goal:
                update_query += "SET u.study_goal = $study_goal "
                params["study_goal"] = study_goal
                
            session.run(update_query, params)

    def check_in(self, uid):
        """用户打卡记录
        
        Args:
            uid (str): 用户ID
            
        Returns:
            Record: 包含 streak_days（连续打卡天数）和 total_study_days（总学习天数）
            
        Note:
            - 更新最后打卡时间
            - 增加总学习天数
            - 根据上次打卡时间更新连续打卡天数
        """
        current_time = int(time.time())
        with self.driver.session() as session:
            result = session.run("""
                MATCH (u:User {uid: $uid})
                WITH u, datetime($current_time) as current_time,
                     datetime(u.last_checkin) as last_checkin
                SET u.last_checkin = $current_time,
                    u.total_study_days = u.total_study_days + 1,
                    u.streak_days = CASE
                        WHEN date(last_checkin) = date(datetime().minus(duration('P1D')))
                        THEN u.streak_days + 1
                        ELSE 1
                    END
                RETURN u.streak_days as streak, u.total_study_days as total
            """, uid=uid, current_time=current_time)
            return result.single()

    def get_user_stats(self, uid):
        """获取用户学习统计信息
        
        Args:
            uid (str): 用户ID
            
        Returns:
            dict: 包含以下统计信息：
                - streak_days: 连续打卡天数
                - total_study_days: 总学习天数
                - mastered_words: 已掌握单词数
                - theme_color: 主题颜色
                - reminder_time: 提醒时间
                - study_goal: 学习目标
        """
        with self.driver.session() as session:
            result = session.run("""
                MATCH (u:User {uid: $uid})
                OPTIONAL MATCH (u)-[:MASTERED]->(w:Word)
                WITH u, count(w) as mastered_words
                RETURN {
                    streak_days: u.streak_days,
                    total_study_days: u.total_study_days,
                    mastered_words: mastered_words,
                    theme_color: u.theme_color,
                    reminder_time: u.reminder_time,
                    study_goal: u.study_goal
                } as stats
            """, uid=uid)
            return result.single()["stats"]

    def update_user_profile(self, uid, data):
        """更新用户信息"""
        with self.driver.session() as session:
            query = """
            MATCH (u:User {uid: $uid})
            SET u += $data
            RETURN u
            """
            return session.run(query, uid=uid, data=data)

    def get_user_by_account(self, account):
        """通过账号获取用户信息"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (u:User {account: $account})
                RETURN u
            """, account=account)
            return result.single()

    def batch_import_words(self, words_data, wordlist_id):
        """批量导入单词到指定词单
        
        Args:
            words_data (list): 单词数据列表，每个元素应包含：
                - word: 单词文本
                - translation: 翻译
                - phonetic: 音标
                - difficulty: 难度等级
            wordlist_id (str): 目标词单ID
            
        Note:
            支持从 CET_4_6_edited.txt 和 COCA_20000.txt 等文件导入
        """
        with self.driver.session() as session:
            session.run("""
                UNLOAD CSV WITH HEADERS FROM 'file:///words.csv' AS row
                MATCH (wl:WordList {wid: $wid})
                MERGE (w:Word {text: row.word})
                SET w.translation = row.translation,
                    w.phonetic = row.phonetic,
                    w.difficulty = toInteger(row.difficulty)
                CREATE (wl)-[:CONTAINS]->(w)
            """, wid=wordlist_id)

    def import_from_file(self, file_path, wordlist_id):
        """从文件导入单词数据
        
        Args:
            file_path (str): 源数据文件路径
            wordlist_id (str): 目标词单ID
            
        Supported formats:
            - CSV文件
            - JSON文件
            - SQLite数据库
            - CET_4_6_edited.txt
            - COCA_20000.txt
            
        Note:
            确保文件使用UTF-8编码
        """
        words_data = []
        
        if file_path.endswith('.csv'):
            import csv
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                words_data = list(reader)
                
        elif file_path.endswith('.json'):
            import json
            with open(file_path, 'r', encoding='utf-8') as f:
                words_data = json.load(f)
                
        elif file_path.endswith('.sqlite'):
            import sqlite3
            conn = sqlite3.connect(file_path)
            cursor = conn.cursor()
            cursor.execute("SELECT word, translation, phonetic, difficulty FROM words")
            columns = ['word', 'translation', 'phonetic', 'difficulty']
            words_data = [dict(zip(columns, row)) for row in cursor.fetchall()]
            conn.close()

        self.batch_import_words(words_data, wordlist_id)

    