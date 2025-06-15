import streamlit as st
import json
import os
import re
import hashlib
import time
import csv
import pandas as pd
import configparser
from datetime import datetime
from typing import Dict, List, Tuple, DefaultDict
from collections import defaultdict
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from tenacity import retry, wait_exponential, stop_after_attempt

# 预设值
PRESET_TYPES = ["地质工程", "地质特征", "勘测技术", "试验", "地质资料", "工具", "样本"]
PRESET_PREDICATES = ["包含", "需要", "来源", "影响", "用于", "得到", "参考", "调查", "适用", "反映","避开"]
TYPE_MAP = {
    'PC': '地质工程', 'GF': '地质特征', 'RH': '勘测技术',
    'TL': '工具', 'GI': '地质资料', 'SP': '样本', 'EP': '试验'
}


# ------------------- TextSegmentor 类 -------------------
class TextSegmentor:
    def __init__(self,provider_config):
        self.provider_config = provider_config['seg']
        self.llm = self._init_llm(self.provider_config['provider'])


    def _init_llm(self, provider):
        config = self.provider_config.get(provider, {})
        api_key = config.get('api_key', '')
        
        if provider == "kimi":
            base_url = config.get('base_url', "https://api.moonshot.cn/v1")
            return ChatOpenAI(
                api_key=api_key,
                base_url=base_url,
                model="moonshot-v1-32k",
                temperature=0.1
            )
        else:  # deepseek
            base_url = config.get('base_url', "https://api.deepseek.com/v1")
            return ChatOpenAI(
                api_key=api_key,
                base_url=base_url,
                model="deepseek-chat",
                temperature=0.5
            )

    def semantic_segment(self, text: str) -> List[str]:
        prompt = """
            请严格按"x.x.x"格式的数字小节编号进行分段，规则：
            1. 当出现"数字.数字.数字"格式的编号时（如4.3.1），将该编号之后的内容视为新段落
            2. 段落持续到下一个同级编号出现为止（如4.3.1的内容持续到4.3.2之前）
            3. 保留自然段落结构，不要合并不同小节内容
            4. 保留编号前缀，输出段落内容
            
            特殊处理要求：
            - 数字编号后的空格/换行符等格式差异不影响分段判断
            - 保留条款中的数字编号（如"1  隧道应选择..."中的1/2/3）

            示例：
            原文：4.3.1 隧道位置...原则：
            1 隧道应选择...有利。
            2 隧道应避开...
            处理后：
            隧道位置...原则：
            1 隧道应选择...有利。
            2 隧道应避开...

            需要分段的文本：{text}
            """
        chunks = []
        for i in range(0, len(text), 3000):
            chunk = text[i:i+3000]
            response = self.llm.invoke(prompt.format(text=chunk)).content
            chunks.extend([p.strip() for p in response.split('\n\n') if p.strip()])
        return chunks

# ------------------- RelationExtractor 类 -------------------
class RelationExtractor:
    def __init__(self,provider_config, relation_rules: str = None):
        self.segmentor = TextSegmentor(provider_config)
        self.provider_config = provider_config['extract']
        self.relations = self.parse_relation_rules(relation_rules) if relation_rules else {}
        self.api_config = {'max_workers': 6, 'retries': 3, 'delay': 0.8}
        self.validation_rules = {'predicate_whitelist': set(PRESET_PREDICATES)}
        
        # 初始化提示词
        self.template = """作为地质勘探邻域的专家，从以下地质勘探邻域文本联合提取三元组以构建知识图谱：{text}
                        并满足以下要求：
                        1. 从文本中识别实体，并按照给定的实体类型给他们分类：{entity_types}
                        2. 根据上下文以及结合地质领域的专业知识推断实体间的逻辑关系
                        3. 关系类型为：{predicate_list}，如果推断的关系不在给定的类型中：
                           - 若与给定关系类型所表述的意思一致，则替换为给定关系类型
                           - 若与给定关系类型所表述的意思不一致，则忽略该三元组
                        4.每个三元组必须包含完整要素
                    
                        避免提取类似以下的内容：
                        1.抽象概念如"本规范第四章"
                        2.泛指表述如"相关要求"
                        3.不属于地质勘探邻域的内容

                        实体类型及示例：
                        地质工程(PC) - 示例：改建铁路、边坡工程
                        地质特征(GF) - 示例：砂土层、断裂带
                        勘测技术(RH) - 示例：地质钻探、物探
                        工具(TL) - 示例：钻机、测量仪
                        地质资料(GI) - 示例：地质勘察报告、剖面图
                        样本(SP) - 示例：岩芯样本、土样
                        试验(EP) - 示例：承载力试验、渗透试验

                        返回格式示例：[地质钻探|GF]→用于→[断裂带|GF]，不要包含*号等特殊符号，不要提取纯数字、比例尺等无意义内容"""
        
        self.prompt = PromptTemplate(
            template=self.template,
            input_variables=["text"],
            partial_variables={
                "entity_types": "、".join(f"{v}({k})" for k, v in TYPE_MAP.items()),
                "predicate_list": "、".join(self.validation_rules['predicate_whitelist']) 
            }
        )
        
        # 初始化大模型
        self.extract_llm = self.segmentor._init_llm("deepseek")
        
        # 缓存
        self.entity_cache = defaultdict(set)
        self.relation_cache = defaultdict(set)

    def _init_llm(self, provider):
        # 从session_state获取配置
        config = self.provider_config.get(provider, {})
        api_key = config.get('api_key', '')
        base_url = config.get('base_url', '')
        model_name = config.get('model', 'deepseek-chat')  # 获取选择的模型名称
    

        if not api_key:
            st.error(f"请提供{provider}的API密钥")
            return None
            
        if provider == "kimi":
            base_url = base_url or "https://api.moonshot.cn/v1"
            return ChatOpenAI(
                api_key=api_key,
                base_url=base_url,
                model="moonshot-v1-32k",
                temperature=0.1
            )
        elif provider == "qwen":
            return ChatOpenAI(
                api_key=config['api_key'],
                base_url=config['base_url'],
                model=config.get('model', 'qwen-max'),
                temperature=0.3
            )

        elif provider == "doubao":
            return ChatOpenAI(
                api_key=config['api_key'],
                base_url=config['base_url'],
                model=config.get('model', 'chat'),
                temperature=0.5
            )
        
        else:  # deepseek
            base_url = base_url or "https://api.deepseek.com/v1"
            return ChatOpenAI(
                api_key=api_key,
                base_url=base_url,
                model=model_name,
                temperature=0.5
            )
    @staticmethod
    def parse_relation_rules(content: str) -> Dict[Tuple[str, str], str]:
        reader = csv.reader(content.strip().splitlines())
        headers = next(reader)[1:]
        relations = {}
        
        for row in reader:
            if len(row) < 1: continue
            target = row[0].split()[-1]
            for idx, rel in enumerate(row[1:]):
                if idx >= len(headers): break
                source = headers[idx].split()[-1]
                if rel.strip() not in ('/', ''):
                    relations[(source, target)] = rel.strip()
        return relations

    def _preprocess_text(self, text: str) -> str:
        text = re.sub(r'\d+[)、.]', '', text)
        return re.sub(r'；', '。', text)

    def _extract_segment_id(self, text: str) -> Tuple[str, str]:
        patterns = [
            r'^(\d+\.\d+\.\d+)\s+',
            r'^(\d+\.\d+)\s+',
            r'^第(\d+条)\s+',
            r'^(Article \d+)\s+'
        ]
        for pattern in patterns:
            if match := re.search(pattern, text[:100]):
                return (match.group(1), text[match.end():])
        return (f'seg_{int(time.time())}', text)

    def _validate_triples(self, context: str, triples: List[Dict]) -> List[Dict]:
        return [t for t in triples if self._check_relation(context, t)]
    
    def _validate_triple_structure(self, triple: Dict) -> bool:
        required_keys = {'subject', 'subject_type', 'predicate', 'object', 'object_type'}
        return all(key in triple and bool(triple[key]) for key in required_keys)

    @retry(wait=wait_exponential(multiplier=1.5), stop=stop_after_attempt(3))
    def _safe_api_call(self, func, *args, **kwargs):
        time.sleep(self.api_config['delay'])
        try:
            result = func(*args, **kwargs)
            if "rate limit" in str(result).lower():
                raise Exception("API速率限制触发")
            return result
        except Exception as e:
            if "429" in str(e):
                time.sleep(30)
            raise

    def _process_chunk(self, chunk: str) -> List[Dict[str, str]]:
        cache_key = hashlib.md5(chunk[:200].encode()).hexdigest()
        if cache_key in self.relation_cache:
            return json.loads(self.relation_cache[cache_key])

        try:
            chunk = self._preprocess_text(chunk) if chunk else ""
            formatted_prompt = self.prompt.format(text=chunk)
            response = self.extract_llm.invoke(formatted_prompt, timeout=30).content
            
            raw_response = response if hasattr(response, 'content') else str(response)
            initial_triples = self._parse_joint_results(raw_response)
            
            if not isinstance(initial_triples, list):
                initial_triples = []
                
            processed_triples = []
            for triple in initial_triples:
                if not self._validate_triple_structure(triple):
                    continue

                processed_triples.append({
                    "subject": str(triple.get("subject", "")),
                    "subject_type": str(triple.get("subject_type", "")),
                    "predicate": str(triple.get("predicate", "")),
                    "object": str(triple.get("object", "")),
                    "object_type": str(triple.get("object_type", ""))
                })
                
            valid_triples = self._validate_triples(chunk, processed_triples)
            self.relation_cache[cache_key] = json.dumps(valid_triples)
            return valid_triples

        except Exception as e:
            error_msg = str(e)
            if "401" in error_msg and "Authentication Fails" in error_msg:
                st.error("段落处理失败，检测到 API Key 认证失败，请检查配置页面的 API Key 是否正确。")
                print("段落处理失败，检测到 API Key 认证失败，请检查配置页面的 API Key 是否正确。")
            else:
                st.error(f"段落处理失败: {error_msg}")
                print(f"段落处理失败: {error_msg}")
            return []

    def _parse_joint_results(self, raw: str) -> List[Dict]:
        triples = []  
        pattern = re.compile(
            r"\d*\.?\s*\[([^|\]]+)[\s|]*([A-Z]{2})\][\s→|-]+([^→]+?)[\s→|-]+\[([^|\]]+)[\s|]*([A-Z]{2})\]"
        )

        raw = re.sub(r'(\d+\.)\n', r'\1', raw)
        for line in raw.split('\n'):
            line = line.replace("：", ":").strip()
            if match := pattern.search(line):
                subj, subj_type, pred, obj, obj_type = match.groups()
                triples.append({
                    "subject": subj.strip(),
                    "subject_type": TYPE_MAP.get(subj_type.strip(), subj_type),
                    "predicate": pred.strip(),
                    "object": obj.strip(),
                    "object_type": TYPE_MAP.get(obj_type.strip(), obj_type),
                    "validated": False
                })
        return triples

    def _check_relation(self, context: str, triple: Dict) -> bool:
        if triple['subject_type'].split()[0] == triple['object_type'].split()[0]:
            return False
        
        if len(triple['subject']) < 2 or len(triple['object']) < 2:
            return False

        return True

    def extract(self, text: str, relation_rules: str = None) -> Dict:
        if relation_rules:
            self.relations = self.parse_relation_rules(relation_rules)
        
        chunks = self.segmentor.semantic_segment(text)
        results = []
        has_error = False  # 新增错误标记
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, chunk in enumerate(chunks):
            status_text.text(f"处理段落 {idx+1}/{len(chunks)}...")
            progress_bar.progress((idx + 1) / len(chunks))
            
            segment_id, cleaned_chunk = self._extract_segment_id(chunk)
            triples = self._process_chunk(cleaned_chunk)
            if not triples:  # 若返回空列表，说明可能出现错误
                has_error = True  # 标记出现错误

            results.append({
                "segment_id": segment_id,
                "text_content": cleaned_chunk[:1000],
                "triples": triples
            })
            
            time.sleep(self.api_config['delay'])
        
        progress_bar.empty()
        status_text.empty()
        
        return {
            "metadata": {
                "format_version": "1.1",
                "generated_at": datetime.now().isoformat(),
                "source": "地质文本提取"
            },
            "total_segments": len(chunks),
            "segments": results,
            "has_error": has_error  # 返回错误标记
        }

    #新方法
    def update_prompts(self, new_prompts: dict):
        if "关系提取" in new_prompts:
            self.template = new_prompts["关系提取"]
            self.prompt = PromptTemplate(
                template=self.template,
                input_variables=["text"],
                partial_variables={
                    "entity_types": "、".join(f"{v}({k})" for k, v in TYPE_MAP.items()),
                    "predicate_list": "、".join(self.validation_rules['predicate_whitelist']) 
                }
            )


    def to_csv(self, data: Dict, output_dir: str = ".") -> Tuple[str, str]:
        all_triples = []
        for seg in data.get('segments', []):
            if isinstance(seg, dict) and 'triples' in seg:
                all_triples.extend([
                    {**triple, "segment_id": seg.get('segment_id', '')}
                    for triple in seg.get('triples', [])
                ])
        
        entities = {}
        for idx, triple in enumerate(all_triples, 1):
            for role in ['subject', 'object']:
                name = triple[f'{role}']
                ent_type = triple[f'{role}_type']
                if name not in entities:
                    entities[name] = {
                        'id': len(entities) + 1,
                        'name': name,
                        'entity_type': ent_type
                        
                    }

        entities_df = pd.DataFrame(entities.values())
        relations_df = pd.DataFrame([
            {
                'start_id': entities[triple['subject']]['id'],
                'end_id': entities[triple['object']]['id'],
                'type': triple['predicate']
            }
            for triple in all_triples
            if triple['subject'] in entities and triple['object'] in entities
        ])

        base_name = "geological_triples"
        entities_path = os.path.join(output_dir, f"{base_name}_entities.csv")
        relations_path = os.path.join(output_dir, f"{base_name}_relations.csv")

        entities_df.to_csv(entities_path, index=False,encoding='utf-8-sig')
        relations_df.to_csv(relations_path, index=False,encoding='utf-8-sig')
        
        return entities_path, relations_path
    
###########--------------------------------############
# ------------------- Streamlit 应用 -------------------
##########---------------------------------############

DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"

PRESET_PROMPTS = {
    "分段规则": TextSegmentor.semantic_segment.__doc__,
    "关系提取": RelationExtractor.__init__.__doc__
}
def parse_prompt_update(response: str) -> dict:
    """解析AI生成的提示词修改，强化结构匹配"""
    patterns = [
        r"## (分段规则|关系提取)[\s：:]*([\s\S]+?)(?=## |$)",  # 强化板块匹配
        r"【(分段规则|关系提取)】[\s：:]*([\s\S]+?)(?=【|$)"
    ]
    
    matches = []
    for pattern in patterns:
        matches.extend(re.findall(pattern, response, flags=re.DOTALL))
    
    return {
        key: value.strip()  # 去除多余修饰词
        for key, value in matches
        if key in ["分段规则", "关系提取"]  # 严格匹配预设键
    }


def main():
    global PRESET_PROMPTS
    
    # 设置页面配置，包含绿色主题
    st.set_page_config(
        page_title="地质工程三元组提取与校正系统", 
        layout="wide",
        initial_sidebar_state="expanded",  # 侧边栏默认展开
        page_icon="🌱"  # 绿色主题图标
    )
    
    # 自定义CSS样式，设置绿色主题
    st.markdown("""
        <style>
        /* 主标题样式 */
        .main-title {
            text-align: center;
            color: #2E7D32;
            font-size: 2.5rem;
            font-weight: bold;
            margin-bottom: 2rem;
            padding: 1rem;
            background: linear-gradient(90deg, #E8F5E8, #C8E6C9, #E8F5E8);
            border-radius: 10px;
            border-left: 5px solid #4CAF50;
        }
        
        /* 侧边栏按钮样式 */
        .stButton > button {
            width: 100%;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.75rem;
            margin: 0.25rem 0;
            font-weight: bold;
            transition: all 0.3s;
        }
        
        .stButton > button:hover {
            background-color: #45A049;
        }
        </style>
    """, unsafe_allow_html=True)

    # 居中显示主标题
    st.markdown(
        '<h1 class="main-title">🌱 地质工程三元组提取与校正系统</h1>', 
        unsafe_allow_html=True
    )
    
    # 初始化 session_state（保持你原有的初始化代码）
    if 'extracted_data' not in st.session_state:
        st.session_state.extracted_data = None
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 1
    if 'prompt_history' not in st.session_state:
        st.session_state.prompt_history = []
    if 'provider_config' not in st.session_state:
        st.session_state.provider_config = {
            'seg':{
                'provider': 'kimi',
                'kimi': {'api_key': '', 'base_url': 'https://api.moonshot.cn/v1', 'model': 'moonshot-v1-32k'},
                'deepseek': {'api_key': '', 'base_url': 'https://api.deepseek.com/v1','model':'deepseek-chat'},
                'qwen': {'api_key': '', 'base_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1','model':'qwen-max'},
                'doubao': {'api_key': '', 'base_url': 'https://doubao.com/api/v1/','model':'chat'}
            },
            'extract':{
                'provider': 'deepseek',
                'kimi':{'api_key': '', 'base_url': 'https://api.moonshot.cn/v1', 'model': 'moonshot-v1-32k'},
                'deepseek': {'api_key': '', 'base_url': 'https://api.deepseek.com/v1','model':'deepseek-chat'},
                'qwen': {'api_key': '', 'base_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1','model':'qwen-max'},
                'doubao': {'api_key': '', 'base_url': 'https://doubao.com/api/v1/','model':'chat'}
            }
        }
    if 'correction_records' not in st.session_state:
        st.session_state.correction_records = []
    if 'original_triple_count' not in st.session_state:
        st.session_state.original_triple_count = 0

    # 侧边栏导航
    with st.sidebar:
        st.markdown("### 📋 功能导航")
        
        # 初始化当前页面状态
        if 'current_tab' not in st.session_state:
            st.session_state.current_tab = 'config'
        
        # 导航按钮
        if st.button('⚙️ 配置', key='nav_config', use_container_width=True):
            st.session_state.current_tab = 'config'
            st.rerun()
            
        if st.button('📝 文本提取', key='nav_extract', use_container_width=True):
            st.session_state.current_tab = 'extract'
            st.rerun()
            
        if st.button('✏️ 三元组校正', key='nav_correct', use_container_width=True):
            st.session_state.current_tab = 'correct'
            st.rerun()
            
        if st.button('💾 导出结果', key='nav_export', use_container_width=True):
            st.session_state.current_tab = 'export'
            st.rerun()
            
        if st.button('🤖 AI助手', key='nav_assistant', use_container_width=True):
            st.session_state.current_tab = 'assistant'
            st.rerun()
        
        st.markdown("---")
        
        # 侧边栏状态信息
        if st.session_state.extracted_data:
            st.success("✅ 数据已加载")
            total_triples = sum(len(seg["triples"]) for seg in st.session_state.extracted_data["segments"])
            st.metric("三元组总数", total_triples)
        else:
            st.info("ℹ️ 暂无数据")

    # 主内容区域 - 直接用条件判断，不需要额外函数
    if st.session_state.current_tab == 'config':
        # ========== 配置页面内容（原tab_config内容） ==========
        st.header("API配置")
        st.info("请在此配置用于分句和关系提取的API服务")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("分句服务")
            st.session_state.provider_config['seg']['provider'] = st.selectbox(
                "分段服务商",
                options=["kimi", "qwen", "doubao","deepseek"],
                index=["kimi", "qwen", "doubao","deepseek"].index(
                    st.session_state.provider_config['seg']['provider']),
                    key = "seg_provider_select"
            )
            provider = st.session_state.provider_config['seg']['provider']
            model_options = {
                "kimi": ["moonshot-v1-128k", "moonshot-v1-32k"],
                "qwen": ["qwen-max", "qwen-plus", "qwen-turbo"],
                "doubao": ["chat", "pro", "max"],
                "deepseek": ["deepseek-chat", "deepseek-reasoner"]
            }

            current_models = model_options.get(provider, ["default-model"])

            st.session_state.provider_config['seg'][provider]['model'] = st.selectbox(
                f"{provider.upper()} 模型",
                options=current_models,
                index=current_models.index(
                    st.session_state.provider_config['seg'][provider].get('model', current_models[0])
                ),
                key=f"seg_{provider}_model"
            )

            st.session_state.provider_config['seg'][provider]['api_key'] = st.text_input(
                f"{provider.upper()} API密钥",
                value=st.session_state.provider_config['seg'][provider]['api_key'],
                type="password",
                key=f"seg_{provider}_api_key"
            )
            st.session_state.provider_config['seg'][provider]['base_url'] = st.text_input(
                f"{provider.upper()} API地址",
                value=st.session_state.provider_config['seg'][provider]['base_url'],
                key=f"seg_{provider}_base_url"
            )
            
        with col2:
            st.subheader("关系提取服务 ")
            st.session_state.provider_config['extract']['provider'] = st.selectbox(
                "提取服务商",
                options=["deepseek", "qwen", "doubao",'kimi'],
                index=["deepseek", "qwen", "doubao","kimi"].index(
                    st.session_state.provider_config['extract']['provider']),
                    key="extract_provider_select"
            )
            
            provider = st.session_state.provider_config['extract']['provider']
            model_options = {
                "deepseek": ["deepseek-chat", "deepseek-reasoner"],
                "qwen": ["qwen-max", "qwen-plus", "qwen-turbo"],
                "doubao": ["chat", "pro", "max"],
                "kimi": ["moonshot-v1-128k", "moonshot-v1-32k"]
            }
            current_models = model_options.get(provider, ["default-model"])
            current_model = st.session_state.provider_config['extract'][provider].get('model', current_models[0])
    
            if current_model not in current_models:
                current_model = current_models[0]
            st.session_state.provider_config['extract'][provider]['model'] = st.selectbox(
                f"{provider.upper()} 模型",
                options=current_models,
                index=current_models.index(current_model),
                key=f"extract_{provider}_model"
            )
            st.session_state.provider_config['extract'][provider]['api_key'] = st.text_input(
                f"{provider.upper()} API密钥",
                value=st.session_state.provider_config['extract'][provider]['api_key'],
                type="password",
                key=f"extract_{provider}_api_key"
            )
            st.session_state.provider_config['extract'][provider]['base_url'] = st.text_input(
                f"{provider.upper()} API地址",
                value=st.session_state.provider_config['extract'][provider]['base_url'],
                key=f"extract_{provider}_base_url"
            )
        
        st.divider()
        st.subheader("智能助手服务")
        seg_provider = st.session_state.provider_config['seg']['provider']
        extract_provider = st.session_state.provider_config['extract']['provider']
        available_providers = list({seg_provider, extract_provider})
        
        assistant_provider = st.selectbox(
            "选择服务商（从已配置服务中选择）",
            options=available_providers,
            index=0
        )
        st.session_state.assistant_provider = assistant_provider

        st.subheader("使用说明")
        st.markdown("""
        1. **Kimi**: 用于将长文本分割成有意义的段落
        2. **DeepSeek**: 用于从段落中提取地质工程三元组
        3. **豆包(Doubao)**: 支持分句和关系提取服务
        4. **QWEN**: 支持分句和关系提取服务
        """)
        
    elif st.session_state.current_tab == 'extract':
        # ========== 文本提取页面内容（原tab1内容） ==========
        st.header("地质文本三元组提取")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("上传文本")
            text_file = st.file_uploader("上传地质工程文本文件", type=["txt"])
            text_content = st.text_area("或直接输入文本内容", height=300)
            
            st.subheader("关系规则 (可选)")
            rules_file = st.file_uploader("上传关系规则CSV", type=["csv"])
        
        with col2:
            st.subheader("提取设置")
            st.caption("API配置请在配置页面设置")

            seg_provider = st.session_state.provider_config['seg']['provider']
            extract_provider = st.session_state.provider_config['extract']['provider']
        
            seg_configured = bool(st.session_state.provider_config['seg'][seg_provider]['api_key'])
            extract_configured = bool(st.session_state.provider_config['extract'][extract_provider]['api_key'])
            api_ready = seg_configured and extract_configured

            input_ready = text_content.strip() or text_file is not None
        
            button_disabled = not api_ready or not input_ready
            if not api_ready:
                st.warning("请先在⚙️配置页面完成API密钥设置")
            elif not input_ready:
                st.warning("请上传文本文件或输入文本内容")
            
            if st.button("开始提取三元组", 
                    use_container_width=True,
                    disabled=button_disabled):
                
                text = text_content if text_content else text_file.read().decode("utf-8")
                rules = rules_file.read().decode("utf-8") if rules_file else None
                
                extractor = RelationExtractor(
                    provider_config=st.session_state.provider_config,
                    relation_rules=rules
                )
                with st.spinner("正在提取三元组，请稍候..."):
                    st.session_state.extracted_data = extractor.extract(text, rules)
                    st.session_state.current_page = 1
                    st.session_state.original_triple_count = sum(len(seg["triples"]) for seg in st.session_state.extracted_data["segments"])

                    if st.session_state.extracted_data.get("segments")and not st.session_state.extracted_data.get("has_error"):
                        st.success("三元组提取完成！")
                    else:
                        st.error("三元组提取失败，请检查API配置和输入文本")
        
        # 显示提取结果
        if st.session_state.extracted_data:
            st.subheader("提取结果概览")
            st.json(st.session_state.extracted_data["metadata"], expanded=False)
            
            total_segments = st.session_state.extracted_data["total_segments"]
            total_triples = sum(len(seg["triples"]) for seg in st.session_state.extracted_data["segments"])
            
            st.metric("总段落数", total_segments)
            st.metric("总三元组数", total_triples)
            
            if total_segments > 0:
                seg = st.session_state.extracted_data["segments"][0]
                st.subheader(f"段落示例: {seg['segment_id']}")
                st.caption(f"文本内容: {seg['text_content'][:200]}...")
                
                if seg["triples"]:
                    st.write("提取的三元组:")
                    for triple in seg["triples"]:
                        st.code(f"{triple['subject']} ({triple['subject_type']}) → {triple['predicate']} → {triple['object']} ({triple['object_type']})")
                else:
                    st.info("该段落未提取到三元组")

    elif st.session_state.current_tab == 'correct':
        # ========== 三元组校正页面内容（原tab2内容） ==========
        if not st.session_state.extracted_data:
            st.info("请先在'文本提取'页面提取三元组")
        else:
            st.header("三元组校正")
            # 你的原始tab2内容...
            st.info("三元组校正功能（请将原tab2的完整内容复制到这里）")
            if not st.session_state.extracted_data:
                with st.spinner('🌀 正在加载数据...'):
                    time.sleep(0.5)  # 模拟加载过程
                st.info("请先在'文本提取'标签页提取三元组")
                return
            
  
            data = st.session_state.extracted_data
            text_keys = [seg["segment_id"] for seg in data["segments"]]
            
            col_sidebar, col_main = st.columns([1, 3])
            
            with col_sidebar:
                st.subheader("段落选择")
                selected_segment_id = st.selectbox("选择段落", text_keys)
                selected_segment = next(seg for seg in data["segments"] if seg["segment_id"] == selected_segment_id)
                
                st.subheader("文本内容")
                st.caption(selected_segment["text_content"][:500] + ("..." if len(selected_segment["text_content"]) > 500 else ""))
                
                # 新增三元组
                if st.button("➕ 新增三元组", use_container_width=True):
                    if "triples" not in selected_segment:
                        selected_segment["triples"] = []
                    
                    new_triple = {
                        "subject": "[新主体]",
                        "subject_type": PRESET_TYPES[0],
                        "predicate": PRESET_PREDICATES[0],
                        "object": "[新客体]",
                        "object_type": PRESET_TYPES[0],
                        "is_custom" : True
                    }

                    selected_segment["triples"].append(new_triple)
                    st.session_state.extracted_data = data
                    # 记录新增操作
                    st.session_state.correction_records.append({
                        "action": "add",
                        "added": new_triple,
                        "segment_id": selected_segment_id
                    })
                    st.rerun()
                    
                # 下载当前状态
                st.download_button(
                    label="💾 下载当前数据",
                    data=json.dumps(st.session_state.extracted_data, ensure_ascii=False, indent=2),
                    file_name="geological_triples.json",
                    mime="application/json"
                )
            
            with col_main:
                spo_list = selected_segment.get("triples", [])
                num_spo = len(spo_list)
                
                if num_spo == 0:
                    st.info("该段落没有三元组")
                    return
                
                current_model = st.session_state.provider_config['extract'][
                    st.session_state.provider_config['extract']['provider']
                ]['model']
                st.caption(f"当前使用模型：{current_model} | 配置于⚙️页面")

                # 分页控制
                page = st.session_state.current_page - 1
                if page >= num_spo:
                    page = num_spo - 1
                    st.session_state.current_page = num_spo
                
                # 分页导航
                col_nav1, col_nav2, col_nav3 = st.columns([1, 2, 1])
                with col_nav1:
                    if st.button("⬅️ 上一个") and page > 0:
                        st.session_state.current_page -= 1
                        st.rerun()
                with col_nav2:
                    st.markdown(f"**三元组 {page+1}/{num_spo}**", help="使用左右箭头导航")
                with col_nav3:
                    if st.button("➡️ 下一个") and page < num_spo - 1:
                        st.session_state.current_page += 1
                        st.rerun()
                
                # 删除当前三元组
                if st.button("🗑️ 删除当前三元组", type="primary"):
                    original_spo = spo_list[page].copy()
                    del spo_list[page]

                    is_original = not original_spo.get("is_custom", False)
                    st.session_state.extracted_data = data
                    # 记录删除操作
                    st.session_state.correction_records.append({
                        "action": "delete",
                        "original": original_spo,
                        "segment_id": selected_segment_id,
                        "is_original": is_original
                    })
                    st.success("删除成功！")
                    if page >= len(spo_list) and len(spo_list) > 0:
                        st.session_state.current_page = len(spo_list)
                    st.rerun()
                
                # 编辑表单
                spo = spo_list[page]
                with st.form(key="spo_form"):
                    cols = st.columns(2)
                    with cols[0]:
                        new_subject = st.text_input(
                            "主体(Subject)", 
                            value=spo.get("subject", ""),
                            help="地质实体名称，如'砂岩层'"
                        )
                        new_predicate = st.selectbox(
                            "谓词(Predicate)", 
                            PRESET_PREDICATES,
                            index=PRESET_PREDICATES.index(spo.get("predicate", PRESET_PREDICATES[0])),
                            help="选择或输入地质关系类型"
                        )
                    with cols[1]:
                        new_object = st.text_input("客体(Object)", value=spo.get("object", ""))
                        new_subject_type = st.selectbox(
                            "主体类型", 
                            PRESET_TYPES,
                            index=PRESET_TYPES.index(spo.get("subject_type", PRESET_TYPES[0]))
                        )
                        new_object_type = st.selectbox(
                            "客体类型", 
                            PRESET_TYPES,
                            index=PRESET_TYPES.index(spo.get("object_type", PRESET_TYPES[0]))
                        )
                    
                    if st.form_submit_button("💾 保存修改"):
                        original_spo = spo.copy()
                        spo_list[page] = {
                            "subject": new_subject,
                            "subject_type": new_subject_type,
                            "predicate": new_predicate,
                            "object": new_object,
                            "object_type": new_object_type,
                            "is_custom" : spo.get("is_custom", False)
                        }
                        is_original = not original_spo.get("is_custom", False)
            
                        st.session_state.extracted_data = data
                        st.session_state.correction_records.append({
                            "action": "edit",
                            "original": original_spo,
                            "modified": spo_list[page],
                            "segment_id": selected_segment_id,
                            "is_original": is_original
                        })
                        st.success("修改已保存！")
                
                # 当前三元组预览
                st.subheader("当前三元组")
                st.json(spo)

            ##当前的三元组总数
            edited_count = sum(1 for r in st.session_state.correction_records if r["action"] == "edit")
            deleted_count = sum(1 for r in st.session_state.correction_records if r["action"] == "delete")
            added_count = sum(1 for r in st.session_state.correction_records if r["action"] == "add")

        

            if st.session_state.correction_records:
                st.write("详细校正记录:")
                for record in st.session_state.correction_records:
                    if record["action"] == "delete":
                        st.write(f"删除三元组: {record['original']} (段落ID: {record['segment_id']})")
                    elif record["action"] == "edit":
                        st.write(f"修改三元组: 原始 {record['original']} → 修改后 {record['modified']} (段落ID: {record['segment_id']})")
                    elif record["action"] == "add":
                        st.write(f"新增三元组: {record['added']} (段落ID: {record['segment_id']})")

                    # 显示校正记录和正确率
            st.divider()
            st.subheader("校正记录统计")
            cols = st.columns(3)
            with cols[0]:
                st.metric("修改个数", edited_count)
            with cols[1]:
                st.metric("删除个数", deleted_count)
            with cols[2]:
                st.metric("新增个数", added_count)
            
    elif st.session_state.current_tab == 'export':
        # ========== 导出结果页面内容（原tab3内容） ==========
        if not st.session_state.extracted_data:
            st.info("请先在'文本提取'页面提取三元组")
        else:
            st.header("导出结果")
            # 你的原始tab3内容...
            st.info("导出结果功能（请将原tab3的完整内容复制到这里）")
            if not st.session_state.extracted_data:
                st.info("请先在'文本提取'标签页提取三元组")
                return
            
            st.header("导出结果")


            
            data = st.session_state.extracted_data
            
            st.subheader("JSON导出")
            st.download_button(
                label="下载JSON数据",
                data=json.dumps(data, ensure_ascii=False, indent=2),
                file_name="geological_triples.json",
                mime="application/json"
            )
            
            st.subheader("CSV导出")
            if st.button("生成CSV文件", use_container_width=True):
                extractor = RelationExtractor(provider_config=st.session_state.provider_config)
                with st.spinner("正在生成CSV文件..."):
                    entities_path, relations_path = extractor.to_csv(data)
                    st.success("CSV文件生成完成！")
                    
                    col_csv1, col_csv2 = st.columns(2)
                    with col_csv1:
                        # 修复编码问题：使用二进制模式读取并指定UTF-8编码
                        with open(entities_path, "rb") as f:  # 改为二进制模式
                            st.download_button(
                                label="下载实体表",
                                data=f,
                                file_name="entities.csv",
                                mime="text/csv",
                                key="entities_download"
                            )
                    
                    with col_csv2:
                        # 修复编码问题：使用二进制模式读取并指定UTF-8编码
                        with open(relations_path, "rb") as f:  # 改为二进制模式
                            st.download_button(
                                label="下载关系表",
                                data=f,
                                file_name="relations.csv",
                                mime="text/csv",
                                key="relations_download"
                            )
            
            st.subheader("Neo4j导入")
            st.code("""
            // 💥 删除所有旧数据（可选）

            MATCH (n) DETACH DELETE n;
            // 📌 导入节点数据

            LOAD CSV WITH HEADERS FROM 'file:///Nodes.csv' AS row
            CREATE (n:Entity {
              id: toInteger(row.id),
              name: row.name,
              entity_type: row.entity_type
            });

            

            // 正确创建动态类型的关系
            LOAD CSV WITH HEADERS FROM 'file:///relationship.csv' AS row
            MATCH (a:Entity {id: toInteger(row.start_id)})
            MATCH (b:Entity {id: toInteger(row.end_id)})
            CALL apoc.create.relationship(a, row.type, {}, b) YIELD rel
            RETURN count(rel);

            MATCH (n) RETURN n LIMIT 25

            MATCH (n)-[r]->(m)
            RETURN n, r, m
            LIMIT 100;


            MATCH (n:Entity)
            REMOVE n:`试验 EP`, n:`勘测技术 RH`, n:`地质工程 PC`, n:`地质资料 GI`, n:`地质特征 GF`;


            MATCH (n:Entity)
            WITH n.entity_type AS et, collect(n) AS batch
            CALL apoc.create.addLabels(batch, [et]) YIELD node
            RETURN count(node) AS 标记节点总数;

            :style
            node.Entity {
              size: 40px;
              caption: '{name}';
              color: grey;
            }
            node.`试验 EP` {
              color: #e67e22; // 橙色
            }
            node.`勘测技术 RH` {
              color: #9b59b6; // 紫色
            }
            node.`地质工程 PC` {
              color: #3498db; // 蓝色
            }
            node.`地质资料 GI` {
              color: #f1c40f; // 黄色
            }
            node.`地质特征 GF` {
              color: #2ecc71; // 绿色
            }
            relationship {
              color: #e74c3c;
              caption: '{type}';
            }
                    """)
    elif st.session_state.current_tab == 'assistant':
        # ========== AI助手页面内容（原tab4内容） ==========
        st.header("智能提示词助手")
        # 你的原始tab4内容...
        st.info("AI助手功能（请将原tab4的完整内容复制到这里）")
        if st.session_state.extracted_data:
            with st.expander("📌 当前提取内容参考", expanded=True):
                st.caption("以下为最新提取内容，可用于提示词优化参考")
                total_segments = st.session_state.extracted_data["total_segments"]
                total_triples = sum(len(seg["triples"]) for seg in st.session_state.extracted_data["segments"])
                
                cols = st.columns([1,2])
                with cols[0]:
                    st.metric("总段落数", total_segments)
                    st.metric("总三元组数", total_triples)
                    
                with cols[1]:
                    sample_segment = st.session_state.extracted_data["segments"][0]
                    st.caption("示例段落内容（前200字）:")
                    st.code(sample_segment["text_content"][:200] + "...")
                    
                    # 移除嵌套的expander，改为普通显示
                    st.caption("完整段落内容（前500字）:")
                    st.text(sample_segment["text_content"][:500] + ("..." if len(sample_segment["text_content"]) > 500 else ""))
                    
                    if sample_segment["triples"]:
                        st.caption("示例三元组:")
                        st.json(sample_segment["triples"][0])

        if not hasattr(st.session_state, 'assistant_provider') or not st.session_state.assistant_provider:
            st.error("请先在配置页面完成分句和关系提取服务的配置")
            return
        provider = st.session_state.assistant_provider
        try:
            # 判断服务商属于分句还是关系提取服务
            if provider in st.session_state.provider_config['seg']:
                config = st.session_state.provider_config['seg'][provider]
            else:
                config = st.session_state.provider_config['extract'][provider]
                
            if not config.get('api_key'):
                st.error(f"{provider} API密钥未配置，请前往⚙️配置页面设置")
                return
        except KeyError as e:
            st.error(f"服务商配置错误: {str(e)}，请检查配置页面")
            return

        st.header("智能提示词助手")
        
        
        if "prompt_history" not in st.session_state:
            st.session_state.prompt_history = []
        # 初始化对话历史
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if st.session_state.prompt_history:
            with st.expander("🕒 提示词修改历史", expanded=True):
                for idx, record in enumerate(st.session_state.prompt_history):
                    cols = st.columns([1,3,2])
                    with cols[0]:
                        st.markdown(f"**版本 {idx+1}**")
                    with cols[1]:
                        st.caption(f"修改时间: {datetime.fromisoformat(record['modified'].get('generated_at', datetime.now().isoformat()))}")
                    with cols[2]:
                        if st.button("⏮️ 回滚到此版本", key=f"revert_{idx}"):
                            # 恢复历史版本
                            PRESET_PROMPTS.clear()
                            PRESET_PROMPTS.update(record["original"])
                            # 移除之后的修改记录
                            st.session_state.prompt_history = st.session_state.prompt_history[:idx]
                            st.session_state.extracted_data = None
                            st.success("已回滚到此版本，请重新执行提取操作")
                            st.rerun()
        
        
        # 显示历史对话
        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])


        # 对话输入
        if prompt := st.chat_input("请输入您对提示词的改进要求"):
            # 初始化智能助手
            assistant = ChatOpenAI(
            api_key=config['api_key'],
            base_url=config['base_url'],
            model=config['model'],
            temperature=0.5
        )
            # 构建系统提示
            system_prompt = f"""你是一个专门用于地质工程领域的三元组提取提示词优化助手。请直接返回修改后的完整提示词模板，严格遵循以下格式：

                ## 关系提取
                【在此编写完整的改进后提示词模板】
                必须包含所有变量占位符如{{text}}、{{entity_types}}、{{predicate_list}}
                保持原有核心要素，仅优化地质领域相关部分

                ## 分段规则
                【在此编写完整的改进后分句规则】
                保持原有分段逻辑，强化地质特征识别

                当前实体类型：{PRESET_TYPES}
                当前关系类型：{PRESET_PREDICATES}
                现有提示模板：
                {json.dumps(PRESET_PROMPTS, indent=2, ensure_ascii=False)}"""
            
            # 执行对话
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("assistant"):
                response = assistant.invoke([
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=prompt)
                ])
                st.markdown(response.content)
                st.session_state.messages.append({"role": "assistant", "content": response.content})
                col_accept, col_reject = st.columns(2)
            
            
                with col_accept:
                    if st.button("✅ 接受建议", key="accept_btn"):
                        updated_prompts = parse_prompt_update(response.content)
                        if updated_prompts:
                            PRESET_PROMPTS.update(updated_prompts)
                            st.session_state.prompt_history.append({
                                "original": PRESET_PROMPTS.copy(),
                                "modified": updated_prompts
                            })
                            st.session_state.extracted_data = None
                            st.rerun()
                    else:
                        st.error("未检测到有效的提示词修改，请确认格式符合要求")

                with col_reject:
                    if st.button("❌ 拒绝建议", key="reject_btn"):
                        st.session_state.messages.pop()  # 移除最后一条AI消息
                        st.rerun()


if __name__ == "__main__":
    main()