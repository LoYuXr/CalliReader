import json
import os
import re
import time
import traceback
import threading
import queue
from json import JSONDecodeError
from tqdm import tqdm
import hashlib
from openai import OpenAI

def extract_json_substring(text):
    """增强型JSON提取，支持嵌套结构"""
    stack = []
    start_index = -1
    for i, c in enumerate(text):
        if c == '{':
            if not stack:
                start_index = i
            stack.append(c)
        elif c == '}':
            if stack:
                stack.pop()
                if not stack and start_index != -1:
                    return text[start_index:i + 1]
    return text  # 回退返回原文本

def parse_model_response(raw_response):
    """多层级JSON解析策略"""
    cleaned = re.sub(r'^```(?:json)?\s*|\s*```$', '', raw_response, flags=re.IGNORECASE | re.MULTILINE)

    attempts = [
        lambda: json.loads(cleaned),
        lambda: json.loads(extract_json_substring(cleaned)),
        lambda: json.loads(raw_response.strip())
    ]

    for attempt in attempts:
        try:
            return attempt()
        except (JSONDecodeError, AttributeError):
            continue
    return None

def safe_write(output_path, data, max_retries=3):
    """增强型原子写入"""
    temp_path = f"{output_path}.tmp"

    for attempt in range(max_retries):
        try:
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            if os.path.exists(output_path):
                os.remove(output_path)

            os.replace(temp_path, output_path)
            return
        except PermissionError:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise
        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass

def process_single_file(filename, input_dir, output_dir, client, system_prompt, progress_queue):
    """处理单个文件的线程函数"""
    error_log = []
    success = False
    try:
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        # 读取文件内容
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (JSONDecodeError, UnicodeDecodeError) as e:
        error_msg = f"文件解析失败 {filename}: {str(e)}"
        error_log.append(error_msg)
        progress_queue.put((filename, False, error_log))
        return
    except Exception as e:
        error_msg = f"打开文件 {filename} 失败: {str(e)}"
        error_log.append(error_msg)
        progress_queue.put((filename, False, error_log))
        return

    try:
        item_progress = tqdm(data, desc=f'Processing items in {filename}', leave=False)
        # cnt = 20
        for item in item_progress:
            # cnt -= 1
            # if cnt <= 0:
            #     break
            content = item.get('calligraphy_content', '')
            answer = item.get('answer', '')
            if not answer or not content:
                continue

            model_input = f"'calligraphy_content': {content}, 'model_answer': {answer}"

            max_attempts = 3
            retry_delay = 1
            raw_reply = None

            for attempt in range(1, max_attempts + 1):
                try:
                    response = client.chat.completions.create(
                        model="deepseek-chat",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": model_input},
                        ],
                        stream=False,
                        temperature=0.9
                    )
                    raw_reply = response.choices[0].message.content
                    break
                except Exception as e:
                    error_msg = f"{model_input} | 第 {attempt} 次尝试失败 ({type(e).__name__}): {str(e)}"
                    error_log.append(error_msg)
                    if attempt == max_attempts:
                        error_msg = f"{model_input} | 连续 {max_attempts} 次请求失败，终止重试"
                        error_log.append(error_msg)
                        break
                    sleep_time = retry_delay * (2 ** (attempt - 1))
                    error_log.append(f"{model_input} | {sleep_time}秒后重试...")
                    time.sleep(sleep_time)

            if raw_reply:
                parsed = parse_model_response(raw_reply)
                if isinstance(parsed, dict):
                    item.update(parsed)
                else:
                    item['judge_msg'] = raw_reply

            # 实时保存进度
            safe_write(output_path, data)
            item_progress.update(1)

        item_progress.close()
        success = True
    except Exception as e:
        error_msg = f"处理文件 {filename} 时发生未知错误: {str(e)}"
        error_log.append(error_msg)
    finally:
        progress_queue.put((filename, success, error_log))

def process_files(input_dir, output_dir, client, system_prompt):
    """多线程文件处理器"""
    os.makedirs(output_dir, exist_ok=True)
    processed_files = set(os.listdir(output_dir))

    files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    files_to_process = [f for f in files if f not in processed_files]

    progress_queue = queue.Queue()
    threads = []

    # 创建并启动线程
    for filename in files_to_process:
        thread = threading.Thread(
            target=process_single_file,
            args=(filename, input_dir, output_dir, client, system_prompt, progress_queue)
        )
        thread.start()
        threads.append(thread)

    # 等待所有线程完成
    for thread in threads:
        thread.join()

    # 处理进度更新和错误日志
    success_count = 0
    error_log = []
    progress_bar = tqdm(total=len(files_to_process), desc='Processing files', unit='file')
    while not progress_queue.empty():
        filename, success, errors = progress_queue.get()
        if success:
            success_count += 1
        error_log.extend(errors)
        progress_bar.update(1)
        progress_bar.set_postfix_str(f"Processed: {filename}")
    progress_bar.close()

    # 生成总结报告
    print(f"\n处理完成: {success_count}/{len(files_to_process)} 文件成功")
    if error_log:
        print("\n错误日志：")
        print("\n".join(error_log[:10]))
        with open(os.path.join(output_dir, "error_log.txt"), 'w') as f:
            f.write("\n".join(error_log))
if True:
    client = OpenAI(
        api_key="",
        base_url="https://api.deepseek.com"
    )

    SYSTEM_PROMPT = '''
    As a **Calligraphy Intent Analysis Evaluator**, assess responses based on core intent alignment. Factual minutiae are secondary.

    ### **User Input Format**  
    ```json
    {'calligraphy content': "...", 'model_answer': " ..."}
    ```

    ---

    ### **Step 1: Key Intent Extraction**  
    Extract from `calligraphy content`:  
    1. **Primary Creative Motivation**  
    2. **Intended Usage Scenario**  
    3. **Target Audience**  

    ---

    ### **Step 2: Core Assessment**  

    #### **1. Task Completion** (Pass/Fail)  
       - ✔️ **Readable**: Answer is in coherent English
       - ✔️ **Complete**: All important intent elements are mentioned

    #### **2. Intent Accuracy** (80% weight)  
       - ✅ **Main Intent Match**: 10pts if primary motivation correct (half-credit for partial matches)  
       - 🌐 **Context Plausibility**: 8-10pts for reasonable scenario/audience interpretation

    #### 3. Basic Support (20% weight)
       - **Relevant Linking**: 8-10 points will be awarded for effectively connecting the content to the intent of the question. Examples are optional but can enhance clarity.
       - **Specific Analysis Required**: Answers must focus on analyzing the specific content of the given calligraphy work. Overly broad or generic statements will result in point deductions.
       - **Leniency for minor errors**: Small misunderstandings or inaccuracies in highly detailed aspects will be overlooked, provided they do not significantly impact the overall analysis.
    ---

    ### **Step 3: Scoring**  
    - **Fail** only if:  
      - Task Completion missing 
      - Primary motivation completely wrong (0pts)  
    - **Score Calculation**:  
      `(Intent Accuracy × 0.8) + (Basic Support × 0.2)`  

    ---

    ### **Step 4: Output Format**  
    ```json
    {
      "basis": {
        "Task Completion": "[Is the answer readable and complete?]",
        "Intent Accuracy": "[Does the answer capture the intent accurately?]",
        "Basic Support": "[Is the answer specific, avoiding overly broad statements? Are minor errors overlooked if insignificant?]"
      },
      "score": [0-10 score]
    }
    ``` 
    '''

    process_files(
        input_dir="./051701",
        output_dir="scores_051701_ds1",
        client=client,
        system_prompt=SYSTEM_PROMPT
    )