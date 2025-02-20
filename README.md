# PDFVideosummary
## 使用模型
LLM：  
chatglm3-6b  
加载embedding模型：  
bge-base-en（英文）  
bge-base-zh（中文）  
向量数据库使用Milvus  
当搜索问题时，通过相似查询得出相近内容，通过组成prompt输入模型，形成RAG

读取pdf内容：
使用非结构化文件加载器加载pdf文件，对其page_content叠加为test得出pdf所有文本内容，再使用相应的文本分割将内容分割成最长600，且内容相交最多100的文本  
 ![image](https://github.com/user-attachments/assets/1aef1c4e-aa2b-4f90-964a-facb471ad4ab)  

Prompt应用：
为了使模型能完成特定任务，通过prompt模板相应赋值，将上述的pdf内容作为text输入，针对问题QA采用结合向量数据库RAG  
![image](https://github.com/user-attachments/assets/7d90d843-092f-44db-bcb4-142d161e09da)

最终结果：  
![image](https://github.com/user-attachments/assets/671c151e-63a0-407e-8490-61ed98397397)

![image](https://github.com/user-attachments/assets/70992816-dd9f-4339-986d-29f842c5576a)

