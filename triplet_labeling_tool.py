import streamlit as st
import json

PRESET_TYPES = ["地质工程 PC", "地质特征 GF", "勘测技术 RH", "试验 EP", "地质资料 GI"]
PRESET_PREDICATES = ["包含", "需要", "来源", "影响", "用于", "得到", "参考"]

def load_data(uploaded_file):
    return json.load(uploaded_file)

def main():
    st.title("地质工程三元组编辑器")
    
    # 初始化session状态
    if 'data' not in st.session_state:
        st.session_state.data = None
    
    uploaded_file = st.file_uploader("上传JSON文件", type=["json"])
    
    if uploaded_file:
        if st.session_state.data is None:
            st.session_state.data = load_data(uploaded_file)
        
        data = st.session_state.data
        
        text_keys = [k for k in data.keys() if k.startswith("text")]
        
        selected_key = st.sidebar.selectbox("选择文本段落", text_keys)
        
        st.subheader("原始文本内容")
        st.write(data[selected_key])
        
        spo_key = selected_key.replace("text", "spo_list")
        if spo_key not in data:
            st.warning("该段落没有对应的三元组")
            return
            
        spo_list = data[spo_key]
        num_spo = len(spo_list)
        
        # 页码处理
        current_page = int(st.query_params.get("page", 1))
        page = max(0, min(current_page-1, num_spo-1)) if num_spo > 0 else 0

        # 删除功能
        if st.button("🗑️ 删除当前三元组"):
            if num_spo > 0:
                del spo_list[page]
                st.session_state.data[spo_key] = spo_list
                st.success("删除成功！")
                # 自动调整页码
                new_page = max(0, page-1) if page > 0 else 0
                st.query_params.update({"page": new_page+1})
                st.rerun()

        # 新增功能
        if st.button("➕ 新增三元组"):
            new_spo = {
                "subject": "[新主体]",
                "subject_type": PRESET_TYPES[0],
                "predicate": PRESET_PREDICATES[0],
                "object": "[新客体]",
                "object_type": PRESET_TYPES[0]
            }
            spo_list.append(new_spo)
            st.session_state.data[spo_key] = spo_list
            st.success("新增成功！")
            st.query_params.update({"page": len(spo_list)})
            st.rerun()

        # 显示编辑界面
        if num_spo > 0:
            st.subheader(f"编辑三元组 ({page+1}/{num_spo})")
            spo = spo_list[page]

            with st.form(key="spo_form"):
                cols = st.columns(2)
                with cols[0]:
                    new_subject = st.text_input("主体(Subject)", value=spo.get("subject", ""))
                    new_predicate = st.selectbox(
                        "谓词(Predicate)", 
                        PRESET_PREDICATES,
                        index=PRESET_PREDICATES.index(spo.get("predicate", PRESET_PREDICATES[0]))
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
                
                if st.form_submit_button("保存修改"):
                    spo_list[page] = {
                        "subject": new_subject,
                        "subject_type": new_subject_type,
                        "predicate": new_predicate,
                        "object": new_object,
                        "object_type": new_object_type
                    }
                    st.session_state.data[spo_key] = spo_list
                    st.success("修改已保存！")

            st.subheader("当前三元组数据预览")
            st.json(spo)

            # 新增导航按钮（放在编辑界面内部）
            col1, col2, col3 = st.columns([1,1,3])
            with col1:
                if st.button("⬅️ 上一个") and page > 0:
                    st.query_params.update({"page": page})
                    st.rerun()
            with col2:
                if st.button("➡️ 下一个") and page < num_spo-1:
                    st.query_params.update({"page": page+2})
                    st.rerun()

        # 下载功能
        st.sidebar.download_button(
            label="💾 下载修改后的JSON",
            data=json.dumps(st.session_state.data, ensure_ascii=False, indent=2),
            file_name="modified_data.json",
            mime="application/json"
        )

if __name__ == "__main__":
    main()

