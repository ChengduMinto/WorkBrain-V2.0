import gradio as gr
import os
import sys
import base64
from PIL import Image
import io
import asyncio
from typing import Generator, List, Tuple
from config.config import tools
from prompt_gate_network.prompt_gate import gated_network

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# CSS样式
custom_css = """
:root {
    --accent-color: #1890ff;
    --border-color: #d9d9d9;
}

.chat-container {
    background: #fafafa;
    border-radius: 8px;
    padding: 16px;
    height: calc(100vh - 120px);
    display: flex;
    flex-direction: column;
}

#fixed-bottom {
    position: sticky;
    bottom: 0;
    background: white;
    padding: 16px 0 0 0;
    margin-top: auto;
    display: flex;
    align-items: center;
    justify-content: space-between;
    width: 100%;
}

.input-row {
    position: relative;
    width: 100%;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.input-row textarea {
    box-shadow: none !important;
}

.text-input {
    padding: 12px 16px !important;
    border-radius: 24px !important;
    width: calc(100% - 100px) !important;
    min-height: 48px !important;
    border: 1px solid var(--border-color) !important;
    transition: all 0.2s ease;
    flex-grow: 1;
}

.text-input:focus {
    border-color: var(--accent-color) !important;
    box-shadow: 0 0 0 2px rgba(24,144,255,0.2) !important;
}

.button-group {
    display: flex;
    gap: 8px;
    z-index: 1;
    background: linear-gradient(to right, transparent 0%, white 20%);
    flex: 0 1 0%;
    min-width: min(120px, 100%);
}

.upload-button, .send-button {
    width: 50px !important;
    height: 50px !important;
    min-width: 50px !important;
    min-height: 50px !important;
    padding: 0 !important;
    border-radius: 50% !important;
    display: flex !important;
    align-items: center;
    justify-content: center;
    transition: all 0.2s ease !important;
    background: transparent !important;
    border: 1px solid var(--border-color) !important;
}

.upload-button:hover, .send-button:hover {
    background: rgba(0,0,0,0.05) !important;
    transform: scale(1.1);
}

.upload-button::before {
    content: "上传文件（支持图片、音频、视频及常见文档格式）";
    position: absolute;
    bottom: 120%;
    left: 50%;
    transform: translateX(-50%);
    background: rgba(0,0,0,0.9);
    color: white;
    padding: 6px 12px;
    border-radius: 4px;
    font-size: 12px;
    white-space: nowrap;
    opacity: 0;
    transition: opacity 0.2s;
    pointer-events: none;
}

.upload-button:hover::before {
    opacity: 1;
}

.send-button {
    color: var(--accent-color) !important;
    border-color: var(--accent-color) !important;
}

.conversation-panel {
    flex: 1;
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
}

.media-content {
    max-width: 400px;
    border-radius: 8px;
    margin: 8px 0;
}

.dark .text-input {
    background: #333 !important;
    border-color: #555 !important;
    color: white !important;
}

@media (max-width: 768px) {
    .text-input {
        padding-right: 100px !important;
    }
    .button-group {
        right: 8px;
        gap: 4px;
    }
    .upload-button, .send-button {
        width: 36px !important;
        height: 36px !important;
    }
}

/* 消息样式 */
.message {
    margin: 0;
    padding: 12px;
    border-radius: 8px;
    box-shadow: none !important; /* 移除阴影 */
}

.user-message {
    margin-left: auto;
    background: var(--accent-color);
    color: white;
}

.bot-message {
    background: #f5f5f5;
}

.badge {
    font-size: 0.8em;
    color: #666;
    margin-bottom: 4px;
}

.content {
    white-space: pre-wrap;
    word-break: break-word;
}

.media {
    max-width: 100%;
    border-radius: 8px;
    margin-top: 8px;
}

.thumbnail {
    max-width: 50px;
    max-height: 50px;
    border-radius: 8px;
    margin-bottom: 8px;
}
.message-row{
    margin: 0 !important;
}
.flex-wrap{
    border: none !important;
    background: none !important;
    background-color: none !important;
    box-shadow: none !important;
}
.loading-cursor {
    color: #666;
    font-size: 1em;
}

.loading-cursor::after {
    content: '...';
    animation: blink 1.5s infinite steps(4, end);
}

@keyframes blink {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}
"""

async def create_upload_file(img: Image.Image) -> str:
    """将PIL图像转换为base64字符串"""
    try:
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        return base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"图像转换失败: {e}")
        raise

async def stream_text_generation(prompt: str) -> Generator[str, None, None]:
    """流式文本生成"""
    try:
        text_generator = tools["文本生成能力"](prompt)
        if isinstance(text_generator, Generator):
            for chunk in text_generator:
                yield chunk
                await asyncio.sleep(0.02)
        elif asyncio.iscoroutinefunction(text_generator):
            async for chunk in text_generator:
                yield chunk
        else:
            for chunk in text_generator:
                yield chunk
                await asyncio.sleep(0.02)
    except Exception as e:
        yield f"文本生成错误: {str(e)}"

async def execute_high_probability_tools(tools, probabilities_and_prompts):
    """修复所有工具的执行逻辑"""
    try:
        if not probabilities_and_prompts:
            return {}

        results = {}
        for tool_name, (probability, prompt) in probabilities_and_prompts.items():
            if probability > 0.3:
                tool_func = tools.get(tool_name)
                if not tool_func:
                    continue

                try:
                    if tool_name == "文本生成能力":
                        results[tool_name] = (stream_text_generation(prompt.get('text', '')), probability)
                    
                    elif tool_name == "以文生图能力":
                        if asyncio.iscoroutinefunction(tool_func):
                            result = await tool_func(prompt.get('text'))
                        else:
                            result = await asyncio.to_thread(tool_func, prompt.get('text'))
                        results[tool_name] = (result, probability)
                    
                    elif tool_name == "图片理解能力":
                        if prompt.get('image'):
                            img_content = await create_upload_file(prompt['image'])
                            if asyncio.iscoroutinefunction(tool_func):
                                result = await tool_func(img_content, prompt.get('text', '图片描述'))
                            else:
                                result = await asyncio.to_thread(tool_func, img_content, prompt.get('text', '图片描述'))
                            results[tool_name] = (result, probability)
                    
                    elif tool_name == "视频理解能力":
                        if prompt.get('video') and os.path.exists(prompt['video']):
                            if asyncio.iscoroutinefunction(tool_func):
                                result = await tool_func(prompt['video'], prompt.get('text', '视频分析'))
                            else:
                                result = await asyncio.to_thread(tool_func, prompt['video'], prompt.get('text', '视频分析'))
                            results[tool_name] = (result, probability)
                    
                    elif tool_name == "语音识别能力":
                        if prompt.get('audio') and os.path.exists(prompt['audio']):
                            if asyncio.iscoroutinefunction(tool_func):
                                result = await tool_func(prompt['audio'], prompt.get('text', '语音识别'))
                            else:
                                result = await asyncio.to_thread(tool_func, prompt['audio'], prompt.get('text', '语音识别'))
                            results[tool_name] = (result, probability)
                    
                    elif tool_name == "语音合成能力":
                        if text_to_speak := prompt.get('text'):
                            if asyncio.iscoroutinefunction(tool_func):
                                audio_bytes = await tool_func(text_to_speak)
                            else:
                                audio_bytes = await asyncio.to_thread(tool_func, text_to_speak)
                            results[tool_name] = ({"audio_bytes": audio_bytes}, probability)
                    
                    elif tool_name == "文档问答能力":
                        if prompt.get('document') and os.path.exists(prompt['document']):
                            if asyncio.iscoroutinefunction(tool_func):
                                result = await tool_func(prompt['document'], prompt.get('text', '文档分析'))
                            else:
                                result = await asyncio.to_thread(tool_func, prompt['document'], prompt.get('text', '文档分析'))
                            results[tool_name] = (result, probability)
                    
                    elif tool_name == "外部接口能力":
                        if asyncio.iscoroutinefunction(tool_func):
                            result = await tool_func(prompt.get('api_params', {}))
                        else:
                            result = await asyncio.to_thread(tool_func, prompt.get('api_params', {}))
                        results[tool_name] = (result, probability)

                except Exception as e:
                    print(f"工具 {tool_name} 执行异常: {e}")
        return results
    except Exception as e:
        print(f"工具执行失败: {e}")
        raise

async def process_data_stream(input_text: str, input_file=None, chat_history: List[Tuple[str, str]] = None) -> Generator:
    """流式处理数据，支持多轮对话"""
    try:
        # 构建完整的对话上下文
        context = "\n".join([f"User: {user_msg}\nAssistant: {bot_msg}" for user_msg, bot_msg in chat_history]) if chat_history else ""
        full_prompt = f"{context}\nUser: {input_text}" if context else input_text
        
        input_data = {'text': full_prompt}

        if input_file:
            file_ext = os.path.splitext(input_file.name)[1].lower()
            if file_ext in ['.png', '.jpg', '.jpeg']:
                with Image.open(input_file.name) as img:
                    input_data['image'] = img.convert('RGB')
            elif file_ext in ['.wav', '.mp3']:
                input_data['audio'] = input_file.name
            elif file_ext in ['.mp4', '.avi']:
                input_data['video'] = input_file.name
            elif file_ext in ['.docx', '.txt', '.pptx', '.pdf', '.xlsx']:
                input_data['document'] = input_file.name

        gated_output = gated_network(input_data['text'], tools) or {}
        probabilities = {
            tool_name: (float(prob), {
                'text': instruction,
                'image': input_data.get('image'),
                'audio': input_data.get('audio'),
                'video': input_data.get('video'),
                'document': input_data.get('document')
            }) for tool_name, [prob, instruction] in gated_output.items()
        }

        results = await execute_high_probability_tools(tools, probabilities)

        # 新增默认处理逻辑
        if not results:
            default_prompt = input_data.get('text', '')
            text_gen = stream_text_generation(default_prompt)
            results['文本生成能力'] = (text_gen, 0.0)
        
        media_contents = []
        text_generator = None
        
        # 优先处理非文本生成内容
        for tool_name, (result, _) in results.items():
            if tool_name == "以文生图能力":
                if isinstance(result, dict) and (image_path := result.get('image_path')):
                    if os.path.exists(image_path):
                        with Image.open(image_path) as img:
                            media_contents.append(('image', img))
                            yield ("media_update", img)
            elif tool_name != "文本生成能力":
                if isinstance(result, dict):
                    if response := result.get('response'):
                        yield ("text_update", response)
                    if audio_bytes := result.get('audio_bytes'):
                        media_contents.append(('audio', audio_bytes))
                elif isinstance(result, str):
                    yield ("text_update", result)

        # 处理文本生成流
        if "文本生成能力" in results:
            text_generator = results["文本生成能力"][0]
            full_text = ""
            async for chunk in text_generator:
                full_text += chunk
                yield ("text_update", full_text)

        # 最终处理
        if text_generator:
            yield ("text_final", full_text)

        if media_contents:
            yield ("media_final", media_contents)

    except Exception as e:
        print(f"处理异常: {e}")
        yield ("error", "服务暂时不可用，请稍后再试")


def format_media(content) -> str:
    """格式化媒体内容"""
    if isinstance(content, Image.Image):
        buffered = io.BytesIO()
        content.save(buffered, format="PNG")
        return f'<img class="media" src="data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode()}">'
    elif isinstance(content, bytes):
        return f'<audio class="media" controls><source src="data:audio/wav;base64,{base64.b64encode(content).decode()}" type="audio/wav"></audio>'
    return f'<div class="text-message">{content}</div>'

def format_message(content: str, is_user: bool = True) -> str:
    """格式化消息内容"""
    message_class = "user-message" if is_user else "bot-message"
    return f'<div class="message {message_class}"><div class="content">{content}</div></div>'

async def chat_round(input_text: str, input_file, chat_history: List[Tuple[str, str]]):
    """流式对话处理，支持多轮对话"""
    # 用户消息 - 立即显示用户输入
    user_content = format_message(input_text, is_user=True) if input_text else ""
    if input_file:
        try:
            if input_file.name.lower().endswith(('.png', '.jpg', '.jpeg')):
                with Image.open(input_file.name) as img:
                    user_content += format_media(img.convert('RGB'))
            elif input_file.name.lower().endswith(('.wav', '.mp3')):
                with open(input_file.name, 'rb') as f:
                    user_content += format_media(f.read())
        except Exception as e:
            print(f"文件处理失败: {e}")

    # 初始化机器人消息 - 添加"正在生成中..."光标动画
    loading_html = """
    <div class="message bot-message">
        <div class="content">
            <span class="loading-cursor">正在生成中</span>
            <style>
                .loading-cursor::after {
                    content: '...';
                    animation: blink 1.5s infinite steps(4, end);
                }
                @keyframes blink {
                    0%, 100% { opacity: 1; }
                    50% { opacity: 0.5; }
                }
            </style>
        </div>
    </div>
    """
    
    # 更新聊天历史记录
    updated_history = chat_history + [(user_content, loading_html)]
    yield updated_history, "", None

    full_response = ""
    media_elements = []
    
    async for content_type, content in process_data_stream(input_text, input_file, chat_history):
        if content_type == "text_update":
            full_response = content
            updated_history[-1] = (
                user_content,
                format_message(full_response, is_user=False)
            )
        elif content_type == "media_update":
            media_html = format_media(content)
            if media_html not in media_elements:
                media_elements.append(media_html)
            updated_history[-1] = (
                user_content,
                format_message(full_response, is_user=False) + "".join(media_elements))
        elif content_type == "text_final":
            full_response = content
            updated_history[-1] = (
                user_content,
                format_message(full_response, is_user=False) + "".join(media_elements))
        elif content_type == "media_final":
            media_elements = [format_media(c) for _, c in content]
            updated_history[-1] = (
                user_content,
                format_message(full_response, is_user=False) + "".join(media_elements))
        elif content_type == "error":
            updated_history[-1] = (
                user_content,
                format_message(content, is_user=False))
        
        yield updated_history, "", None

with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    <div style="text-align:center;padding:16px 0;">
        <h1 style="margin:0;">WorkBrain 智能体模型 V2.0</h1>
        <div style="color:#666;font-size:0.9em;">多模态交互端到端模型</div>
    </div>
    """)
    
    with gr.Column(elem_classes="chat-container"):
        chatbot = gr.Chatbot(
            label="对话记录",
            show_label=False,
            bubble_full_width=False,
            avatar_images=(None, None),
            elem_classes="conversation-panel",
            sanitize_html=False,
            render_markdown=False
        )

        with gr.Row(elem_id="fixed-bottom"):
            with gr.Column(elem_id="input-wrapper"):
                with gr.Row(elem_classes="input-row"):
                    text_input = gr.Textbox(
                        elem_classes="text-input",
                        placeholder="请输入指令...",
                        lines=1,
                        max_lines=3,
                        container=False,
                        show_label=False
                    )
                    
                    with gr.Row(elem_classes="button-group"):
                        file_input = gr.UploadButton(
                            "📎",
                            file_types=["image", "audio", "video", ".pdf", ".docx"],
                            elem_classes="upload-button",
                            visible=True,
                            scale=0
                        )
                        submit_btn = gr.Button(
                            "⬆️",
                            elem_classes="send-button",
                            variant="primary",
                            scale=0
                        )

    submit_btn.click(
        fn=chat_round,
        inputs=[text_input, file_input, chatbot],
        outputs=[chatbot, text_input, file_input],
        concurrency_limit=20
    )
    text_input.submit(
        fn=chat_round,
        inputs=[text_input, file_input, chatbot],
        outputs=[chatbot, text_input, file_input],
        concurrency_limit=20
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=8005,
        share=True,
        show_error=True,
        max_threads=100
    )