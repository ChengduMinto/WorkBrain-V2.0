import asyncio


from typing import Optional
from contextlib import AsyncExitStack
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
import threading
from queue import Queue
import platform
import nest_asyncio
import re
import os
import base64
import shutil
import uuid
import mimetypes
import traceback
from io import BytesIO
from PIL import Image  # 用于图片处理
import time
import json
import sys
import httpx

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import OpenAI  # 使用OpenAI兼容客户端代替ZhipuAI
# import zhipuai
from dotenv import load_dotenv
import json
import sys
import httpx

# 加载环境变量
load_dotenv()

# 应用nest_asyncio来解决事件循环问题
nest_asyncio.apply()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['OUTPUT_FOLDER'] = 'static/outputs'
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200MB最大上传限制
socketio = SocketIO(app, 
                    cors_allowed_origins="*", 
                    async_mode='threading', 
                    logger=False, 
                    engineio_logger=False,
                    ping_timeout=60,  # 增加ping超时时间到60秒
                    ping_interval=25,  # 增加ping间隔到25秒
                    max_http_buffer_size=100 * 1024 * 1024)  # 增加HTTP缓冲区到100MB

# 创建必要的目录
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# 创建一个事件循环
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

# 添加图片处理函数
def process_image_file(file_path, max_size=1024, quality=85):
    """处理图片文件，压缩大小
    
    Args:
        file_path: 图片文件路径
        max_size: 最大尺寸（长边）
        quality: JPEG压缩质量
        
    Returns:
        (str, int): 返回base64编码的图片数据和大小
    """
    try:
        print(f"处理图片: {file_path}")
        # 打开图片
        img = Image.open(file_path)
        
        # 获取原始尺寸
        original_width, original_height = img.size
        print(f"原始尺寸: {original_width}x{original_height}")
        
        # 如果图片太大，进行缩放
        if max(original_width, original_height) > max_size:
            # 计算缩放比例
            scale = max_size / max(original_width, original_height)
            new_width = int(original_width * scale)
            new_height = int(original_height * scale)
            
            # 调整大小
            img = img.resize((new_width, new_height), Image.LANCZOS)
            print(f"调整尺寸至: {new_width}x{new_height}")
        
        # 确定图片格式
        if img.format not in ['JPEG', 'PNG']:
            img = img.convert('RGB')
            img_format = 'JPEG'
            mime_type = 'image/jpeg'
        else:
            img_format = img.format
            mime_type = f'image/{img_format.lower()}'
        
        # 保存到内存缓冲区
        buffer = BytesIO()
        img.save(buffer, format=img_format, quality=quality)
        buffer.seek(0)
        
        # 获取处理后的图片大小
        processed_size = buffer.getbuffer().nbytes
        print(f"处理后大小: {processed_size} 字节")
        
        # 转换为base64
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # 创建data URI
        data_uri = f"data:{mime_type};base64,{img_base64}"
        
        return data_uri, processed_size
    except Exception as e:
        print(f"图片处理错误: {str(e)}")
        traceback.print_exc()
        # 如果处理失败，返回原始文件
        with open(file_path, 'rb') as f:
            file_content = base64.b64encode(f.read()).decode('utf-8')
        mime_type = mimetypes.guess_type(file_path)[0] or 'image/jpeg'
        return f"data:{mime_type};base64,{file_content}", 0

def process_media_file(file_path, media_type='image', max_size=1024, quality=85):
    """处理媒体文件
    
    Args:
        file_path: 文件路径
        media_type: 媒体类型 ('image', 'video', 'audio')
        max_size: 最大尺寸（图片长边或视频分辨率）
        quality: 压缩质量
        
    Returns:
        (str, int): 返回base64编码的媒体数据和大小
    """
    try:
        print(f"处理{media_type}文件: {file_path}")
        
        if media_type == 'image':
            # 图片处理逻辑保持不变
            img = Image.open(file_path)
            original_width, original_height = img.size
            print(f"原始尺寸: {original_width}x{original_height}")
            
            if max(original_width, original_height) > max_size:
                scale = max_size / max(original_width, original_height)
                new_width = int(original_width * scale)
                new_height = int(original_height * scale)
                img = img.resize((new_width, new_height), Image.LANCZOS)
                print(f"调整尺寸至: {new_width}x{new_height}")
            
            if img.format not in ['JPEG', 'PNG']:
                img = img.convert('RGB')
                img_format = 'JPEG'
                mime_type = 'image/jpeg'
            else:
                img_format = img.format
                mime_type = f'image/{img_format.lower()}'
            
            buffer = BytesIO()
            img.save(buffer, format=img_format, quality=quality)
            buffer.seek(0)
            processed_size = buffer.getbuffer().nbytes
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            data_uri = f"data:{mime_type};base64,{img_base64}"
            
        elif media_type == 'video':
            # 视频处理逻辑
            # 这里应该添加视频压缩和格式转换的代码
            # 为了示例，这里只返回原始视频的data URI
            with open(file_path, 'rb') as f:
                video_content = f.read()
                processed_size = len(video_content)
                video_base64 = base64.b64encode(video_content).decode('utf-8')
                mime_type = mimetypes.guess_type(file_path)[0] or 'video/mp4'
                data_uri = f"data:{mime_type};base64,{video_base64}"
                
            print(f"视频大小: {processed_size} 字节")
            
        elif media_type == 'audio':
            # 音频处理逻辑
            with open(file_path, 'rb') as f:
                audio_content = f.read()
                processed_size = len(audio_content)
                audio_base64 = base64.b64encode(audio_content).decode('utf-8')
                
                # 获取文件扩展名和MIME类型
                file_ext = os.path.splitext(file_path)[1][1:].lower()
                
                # 支持的音频格式及其MIME类型映射
                supported_audio_formats = {
                    'mp3': 'audio/mpeg',
                    'wav': 'audio/wav',
                    'ogg': 'audio/ogg',
                    'flac': 'audio/flac',
                    'm4a': 'audio/mp4',
                    'aac': 'audio/aac'
                }
                
                # 确定MIME类型
                if file_ext in supported_audio_formats:
                    mime_type = supported_audio_formats[file_ext]
                    audio_format = file_ext
                else:
                    # 默认为mp3
                    mime_type = 'audio/mpeg'
                    audio_format = 'mp3'
                    print(f"警告: 未识别的音频格式 '{file_ext}'，将默认使用mp3格式")
                
                data_uri = f"data:{mime_type};base64,{audio_base64}"
                
            print(f"音频处理完成: 大小={processed_size} 字节，格式={audio_format}，支持的格式: mp3, wav, ogg, flac, m4a, aac")
            
        return data_uri, processed_size
        
    except Exception as e:
        print(f"媒体处理错误: {str(e)}")
        traceback.print_exc()
        # 如果处理失败，返回原始文件
        with open(file_path, 'rb') as f:
            file_content = base64.b64encode(f.read()).decode('utf-8')
        mime_type = mimetypes.guess_type(file_path)[0] or 'application/octet-stream'
        return f"data:{mime_type};base64,{file_content}", 0

class MCPWebClient:
    """
    多模态理解处理流程：
    1. 用户输入可能包含：文本+图片、文本+音频、文本+视频、或者无文本仅多模态内容
    2. 流程：
       a. 用户上传的多模态内容在process_query中处理，转换为base64格式
       b. 发送用户消息到前端，确保前端能够正确显示用户上传的附件
       c. 将用户多模态内容保存到会话历史中
       d. 使用qwen-max模型处理请求，模型可能会通过function call调用analyze_multimodal_content工具
       e. analyze_multimodal_content工具使用qwen-omni模型分析多模态内容
       f. 将分析结果返回给qwen-max模型，生成最终回应
       g. 将完整的对话保存到历史中，包括多模态内容
    """
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        # 使用DashScope API Key
        self.api_key = os.getenv('DASHSCOPE_API_KEY')
        if not self.api_key:
            raise ValueError("请设置DASHSCOPE_API_KEY环境变量")
        
        # 更改模型为qwen-max
        self.llm_model = "qwen-max"
        
        # 初始化OpenAI兼容模式客户端
        self.client = OpenAI(
            api_key=self.api_key, 
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        
        # 存储可用工具列表
        self.available_tools = []
        
        # 修改系统消息内容
        system_message = ("你是由成都明途科技有限公司开发的workbrain智能体大模型，你的名字是workbrain。"
                          "You are a Workbrain intelligent agent large model developed by Chengdu Mingtu Technology Co., Ltd. Your name is Workbrain."
                          "You are a helpful assistant with multimodal capabilities. "
                         "When analyzing images, videos or audio, provide descriptive analysis based on what you see. "
                         "DO NOT generate or return markdown image links. "
                         "ONLY analyze the content provided by the user. "
                         "For images, describe what you see in detail.")
        
        # 更新系统消息
        self.messages = [
            {
                'role': 'system',
                'content': [{'type': 'text', 'text': system_message}]
            }
        ]
        
        # 支持的输入媒体类型 (Qwen-Omni-Turbo支持的输入格式)
        self.supported_input_types = {
            'text': {'type': 'text', 'format': 'text'},  # 文本输入
            'image': {'type': 'image_url', 'format': 'image_url'},  # 图片输入
            'video': {'type': 'video_url', 'format': 'video_url'},  # 视频输入
        }
        
        # MCP多媒体生成工具 - 这些工具专门用于生成各种媒体内容
        self.media_generation_tools = {
            'image': 'generate_image',    # 生成图片
            'video': 'generate_video',    # 生成视频
            'audio': 'generate_speech_stream'  # 生成音频(流式)
        }
        
        # 添加对话历史保存和加载功能
        self.history_dir = 'history'
        os.makedirs(self.history_dir, exist_ok=True)
        self.session_id = str(uuid.uuid4().hex)  # 为每个会话生成唯一ID
        self.history_file = f'{self.history_dir}/session_{self.session_id}.json'
        print(f"创建新会话，ID: {self.session_id}")
        
        # 尝试加载历史，如果不存在则创建新文件
        self.try_load_history()
        
    def save_history(self):
        """保存当前对话历史到文件，同一会话的所有对话保存在同一个文件中"""
        try:
            if not os.path.exists(self.history_dir):
                os.makedirs(self.history_dir)
            
            # 直接保存到当前会话文件，不再生成新文件
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(self.messages, f, ensure_ascii=False, indent=2)
            
            print(f"对话历史已保存到: {self.history_file}")
            return True
        except Exception as e:
            print(f"保存对话历史出错: {str(e)}")
            traceback.print_exc()
            return False
    
    def try_load_history(self):
        """尝试从文件加载当前会话的对话历史"""
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    loaded_messages = json.load(f)
                
                # 验证加载的消息格式
                if isinstance(loaded_messages, list) and len(loaded_messages) > 0:
                    self.messages = loaded_messages
                    print(f"成功加载会话历史，消息数: {len(self.messages)}")
            else:
                # 如果历史文件不存在，保存初始系统消息
                self.save_history()
                print(f"创建新的对话历史文件: {self.history_file}")
        except Exception as e:
            print(f"加载对话历史出错: {str(e)}")
            traceback.print_exc()
            # 出错时保存当前消息，确保不丢失
            self.save_history()
        
    def clear_history(self):
        """清空对话历史，但保留系统提示，并创建新的会话文件"""
        system_message = ("You are a helpful assistant with multimodal capabilities. "
                         "When analyzing images, videos or audio, provide descriptive analysis based on what you see. "
                         "DO NOT generate or return markdown image links. "
                         "ONLY analyze the content provided by the user. "
                         "For images, describe what you see in detail.")
        
        # 重置消息历史为仅包含系统消息
        self.messages = [
            {
                'role': 'system',
                'content': [{'type': 'text', 'text': system_message}]
            }
        ]
        
        # 生成新的会话ID和文件
        self.session_id = str(uuid.uuid4().hex)
        self.history_file = f'{self.history_dir}/session_{self.session_id}.json'
        
        # 保存新的历史记录
        self.save_history()
        
        print(f"对话历史已清除，创建新会话，ID: {self.session_id}")
        return True

    async def connect_to_server(self, server_script_path: str):
        """连接到MCP服务器"""
        try:
            socketio.emit('status', {'message': f'正在连接到服务器: {server_script_path}'})
            
            is_python = server_script_path.endswith('.py')
            is_js = server_script_path.endswith('.js')
            if not (is_python or is_js):
                raise ValueError("服务器脚本必须是.py或.js文件")
                
            command = "python" if is_python else "node"
            server_params = StdioServerParameters(
                command=command,
                args=[server_script_path],
                env=None
            )
            
            stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
            self.stdio, self.write = stdio_transport
            self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
            
            await self.session.initialize()
            
            # 获取可用工具列表
            response = await self.session.list_tools()
            self.available_tools = [{
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema
                }
            } for tool in response.tools]
            
            tool_names = [tool['function']['name'] for tool in self.available_tools]
            print(f'连接成功，可用工具: {tool_names}')
            socketio.emit('status', {'message': f'连接成功，可用工具: {tool_names}'})
            return True
            
        except Exception as e:
            socketio.emit('error', {'message': f'连接服务器失败: {str(e)}'})
            return False

    def process_media_result(self, result_content, tool_name):
        """处理媒体结果，返回处理后的内容和类型"""
        try:
            # 针对multimodal_content分析工具的特殊处理
            if tool_name == "analyze_multimodal_content":
                # 多模态内容分析工具的结果是纯文本，直接返回
                return result_content, 'text'
                
            # 首先尝试解析JSON结果
            try:
                result_json = json.loads(result_content)
                # 检查是否为我们设定的结构化JSON格式
                if "type" in result_json and "url" in result_json:
                    media_type = result_json["type"]
                    url = result_json["url"]
                    # 对错误类型特殊处理
                    if media_type == "error":
                        return result_json["message"], "text"
                    
                    # 确保URL是绝对路径
                    if url.startswith("/static/"):
                        # 添加调试日志
                        print(f"返回媒体URL: {url}, 类型: {media_type}")
                    
                    # 返回处理后的结果
                    return url, media_type
            except json.JSONDecodeError:
                # 不是JSON格式，继续按原有逻辑处理
                pass
            
            # 原有逻辑 - 处理data URI或文件路径
            # 如果结果已经是data URI格式，直接返回
            if result_content.startswith('data:'):
                media_type = 'audio' if 'audio' in result_content else 'image' if 'image' in result_content else 'video' if 'video' in result_content else 'text'
                return result_content, media_type
            
            # 检查是否为直接路径
            if os.path.exists(result_content) and os.path.isfile(result_content):
                return self._process_file_path(result_content)
            
            # 尝试从文本中提取路径
            if os.path.sep in result_content:
                # 使用正则匹配所有可能的文件路径
                matches = re.findall(r'[/\\][\w\-. /\\]+\.(png|jpg|jpeg|gif|webp|mp3|wav|ogg|mp4|webm|mov)', result_content)
                for match in matches:
                    full_match = re.search(r'[/\\][\w\-. /\\]+\.'+match, result_content)
                    if full_match:
                        path = full_match.group(0).strip().replace('\\', '/')
                        if os.path.exists(path) and os.path.isfile(path):
                            return self._process_file_path(path)
            
            # 没找到媒体文件，返回原始文本
            return result_content, 'text'
        except Exception as e:
            print(f"解析媒体结果出错: {str(e)}")
            traceback.print_exc()
            return f"解析媒体结果出错: {str(e)}", 'text'
    
    def _process_file_path(self, file_path):
        """处理文件路径，复制到静态目录并返回web路径"""
        try:
            # 如果文件路径已经是web路径（以/static/开头），直接返回
            if file_path.startswith('/static/'):
                # 确定媒体类型
                file_ext = os.path.splitext(file_path)[1].lower()
                if file_ext in ['.png', '.jpg', '.jpeg', '.gif', '.webp']:
                    media_type = 'image'
                elif file_ext in ['.mp4', '.webm', '.mov']:
                    media_type = 'video'
                elif file_ext in ['.mp3', '.wav', '.ogg']:
                    media_type = 'audio'
                else:
                    return f"不支持的文件类型: {file_ext}", 'text'
                return file_path, media_type
                
            # 确定文件类型
            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext in ['.png', '.jpg', '.jpeg', '.gif', '.webp']:
                media_type = 'image'
            elif file_ext in ['.mp4', '.webm', '.mov']:
                media_type = 'video'
            elif file_ext in ['.mp3', '.wav', '.ogg']:
                media_type = 'audio'
            else:
                return f"不支持的文件类型: {file_ext}", 'text'
            
            # 复制文件到静态目录
            filename = os.path.basename(file_path)
            os.makedirs('static/outputs', exist_ok=True)
            destination_path = f'static/outputs/{filename}'
            
            shutil.copy2(file_path, destination_path)
            print(f"已复制媒体文件: {file_path} -> {destination_path}")
            web_path = f'/static/outputs/{filename}'
            return web_path, media_type
        except Exception as e:
            print(f"处理文件路径出错: {str(e)}")
            traceback.print_exc()
            return f"处理文件失败: {str(e)}", 'text'

    async def process_media_tool(self, tool_name, tool_args, tool_call_id):
        """处理媒体生成工具的调用"""
        try:
            # 调用工具
            result = await self.session.call_tool(tool_name, tool_args)
            result_content = str(result) if not isinstance(result, str) else result
            
            # 使用共同的工具结果处理逻辑
            await self._process_tool_result(result, tool_name, tool_call_id)
            
            # 生成回应
            await self._generate_followup_response()
            
            return True
            
        except Exception as e:
            print(f"媒体工具调用错误: {str(e)}")
            traceback.print_exc()
            socketio.emit('error', {'message': f'媒体工具调用错误: {str(e)}'})
            return False

    async def _process_tool_result(self, result, tool_name, tool_call_id):
        """处理工具调用结果并发送到前端"""
        try:
            # 提取结果内容
            if hasattr(result, 'content'):
                # 从CallToolResult对象中提取内容
                content_list = result.content
                result_content = ""
                for item in content_list:
                    if hasattr(item, 'text'):
                        result_content += item.text
            else:
                # 如果是字符串
                result_content = str(result)
            
            print(f"工具 {tool_name} 的结果: {result_content[:100]}...")
            
            # 处理可能的媒体结果
            content, media_type = self.process_media_result(result_content, tool_name)
            
            # 多模态分析工具的结果需要特殊处理
            if tool_name == "analyze_multimodal_content":
                print(f"多模态分析结果已保存，将由后续回应处理")
                
                # 检查是否包含音频转录内容，如果是，直接发送到前端
                if "[音频转录]" in content:
                    print("检测到音频转录内容，发送到前端")
                    
                    # 构建音频转录消息
                    audio_transcription_message = {
                        'type': 'assistant',
                        'content': content,
                        'is_tool_result': True,
                        'tool_name': tool_name,
                        'is_audio_transcription': True
                    }
                    
                    # 发送音频转录结果到前端
                    socketio.emit('message', audio_transcription_message)
                    
                return True
            
            # 构建工具响应消息
            message_data = {
                'type': 'assistant',  # 修改为'assistant'类型，让前端以统一方式处理
                'content': content if media_type == 'text' else '',
                'media_type': media_type if media_type != 'text' else None,
                'media_content': content if media_type != 'text' else None,
                'is_tool_result': True,  # 添加标记，表明这是工具结果
                'tool_name': tool_name
            }
            
            print(f"发送到前端的消息类型: {message_data['type']}, 内容类型: {media_type}, 工具名称: {tool_name}")
            
            # 发送结果到前端
            socketio.emit('message', message_data)
            
            return True
        except Exception as e:
            error_message = f"处理工具结果出错: {str(e)}"
            print(error_message)
            traceback.print_exc()
            socketio.emit('error', {'message': error_message})
            return False

    async def process_query(self, query: str, media_files=None):
        """处理用户查询，包括多媒体输入和工具调用"""
        try:
            print(f"处理查询: 文本={query}, 媒体文件={media_files is not None}")
            
            # 初始化用户内容为列表，以支持多模态内容
            user_content = []
            
            # 处理媒体文件
            image_urls = []
            video_urls = []
            audio_url = None
            audio_format = None
            
            if media_files and media_files.get('has_media'):
                print(f"检测到媒体文件: {len(media_files.get('files', []))}个")
                
                for file_info in media_files.get('files', []):
                    file_path = file_info.get('path')
                    file_type = file_info.get('type')
                    
                    print(f"处理{file_type}文件: {file_path}")
                    
                    if not file_path or not os.path.exists(file_path):
                        print(f"错误: 文件不存在或路径为空: {file_path}")
                        continue
                    
                    try:
                        # 根据文件类型处理
                        if file_type == 'image':
                            print(f"处理图片文件: {file_path}")
                            # 使用图片处理函数压缩图片
                            data_uri, _ = process_image_file(file_path, max_size=1024, quality=85)
                            
                            # 添加到用户内容
                            user_content.append({
                                'type': 'image_url',
                                'image_url': {'url': data_uri}
                            })
                            
                            # 保存图片URL到列表
                            image_urls.append(data_uri)
                            print(f"已添加图片到images列表，当前图片数量: {len(image_urls)}")
                            
                        elif file_type == 'video':
                            print(f"处理视频文件: {file_path}")
                            # 处理视频
                            data_uri, _ = process_media_file(file_path, media_type='video')
                            
                            # 添加到用户内容
                            user_content.append({
                                'type': 'video_url',
                                'video_url': {'url': data_uri}
                            })
                            
                            # 保存视频URL到列表
                            video_urls.append(data_uri)
                            print(f"已添加视频到videos列表，当前视频数量: {len(video_urls)}")
                            
                        elif file_type == 'audio':
                            print(f"处理音频文件: {file_path}")
                            # 处理音频
                            data_uri, _ = process_media_file(file_path, media_type='audio')
                            
                            # 获取音频格式
                            audio_format = os.path.splitext(file_path)[1][1:].lower() or 'mp3'
                            print(f"检测到音频格式: {audio_format}")
                            
                            # 添加到用户内容 - 使用正确的input_audio格式
                            user_content.append({
                                'type': 'input_audio',
                                'input_audio': {
                                    'data': data_uri,
                                    'format': audio_format
                                }
                            })
                            
                            # 保存音频URL和格式
                            audio_url = data_uri
                            print(f"已设置音频URL，格式: {audio_format}")
                        else:
                            print(f"警告: 不支持的文件类型: {file_type}, 将忽略该文件")
                    
                    except Exception as e:
                        print(f"处理媒体文件出错: {str(e)}")
                        traceback.print_exc()
                        continue
            
            # 添加文本内容
            if query:
                user_content.append({
                    'type': 'text',
                    'text': query
                })
            
            # 如果没有有效内容，返回错误
            if not user_content:
                print("错误: 用户内容为空")
                socketio.emit('error', {'message': '没有有效的用户输入'})
                socketio.emit('response_complete')
                return
            
            # 发送用户消息到前端
            self._send_user_message_to_frontend(query, media_files)
            
            # 添加用户消息到历史记录
            user_message = {
                'role': 'user',
                'content': user_content
            }
            
            # 为纯文本查询转换为简单格式
            if not media_files or not media_files.get('has_media'):
                # 提取文本内容
                text_content = query or ""
                user_message = {
                    'role': 'user',
                    'content': text_content
                }
                
            self.messages.append(user_message)
            
            # 保存对话历史
            self.save_history()
            
            # 检查是否有多模态内容
            has_multimodal = bool(image_urls or video_urls or audio_url)
            
            if has_multimodal:
                print(f"检测到多模态内容: 图片数={len(image_urls)}, 视频数={len(video_urls)}, 音频={'有' if audio_url else '无'}")
                
                # 有多模态内容时，直接调用analyze_multimodal_content工具
                print("直接调用analyze_multimodal_content工具处理多模态内容...")
                socketio.emit('status', {'message': '正在分析多模态内容...'})
                
                # 准备工具参数
                tool_args = {
                    'content': query or "",
                    'images': image_urls,
                    'videos': video_urls,
                }
                
                # 只有当实际上传了音频文件时，才添加音频相关参数
                if audio_url:
                    tool_args['audio'] = audio_url
                    tool_args['audio_format'] = audio_format
                    print(f"添加音频参数，格式：{audio_format}")
                else:
                    print("没有检测到音频内容，不添加音频参数")
                
                try:
                    # 生成唯一的工具调用ID
                    tool_call_id = f"call_{uuid.uuid4().hex}"
                    
                    # 向前端发送工具调用状态
                    socketio.emit('tool_status', {
                        'status': 'start',
                        'tool_name': 'analyze_multimodal_content',
                        'message': '正在分析多模态内容...'
                    })
                    
                    # 添加助手消息（包含工具调用）到历史
                    assistant_message = {
                        "role": "assistant", 
                        "content": None,
                        "tool_calls": [
                            {
                                "id": tool_call_id,
                                "type": "function",
                                "function": {
                                    "name": "analyze_multimodal_content",
                                    "arguments": json.dumps(tool_args)
                                }
                            }
                        ]
                    }
                    self.messages.append(assistant_message)
                    
                    # 调用工具
                    print(f"执行工具: analyze_multimodal_content，参数: {tool_args}")
                    result = await self.session.call_tool("analyze_multimodal_content", tool_args)
                    
                    # 提取结果内容
                    result_content = await self._extract_tool_result_content(result)
                    
                    # 添加工具结果到历史
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": result_content
                    })
                    
                    # 多模态分析不直接向前端发送结果，而是通过followup_response处理
                    # 只通知前端工具调用状态
                    socketio.emit('tool_status', {
                        'status': 'end',
                        'tool_name': 'analyze_multimodal_content',
                        'message': '多模态内容分析完成'
                    })
                    
                    # 生成后续回应
                    await self._generate_followup_response()
                    
                except Exception as e:
                    error_message = f"多模态内容分析失败: {str(e)}"
                    print(error_message)
                    traceback.print_exc()
                    socketio.emit('error', {'message': error_message})
                    
                    # 添加错误信息到历史
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id if 'tool_call_id' in locals() else f"error_{uuid.uuid4().hex}",
                        "content": error_message
                    })
            else:
                # 纯文本内容，使用qwen-max模型处理请求
                print("使用Qwen-Max处理纯文本请求...")
                socketio.emit('status', {'message': '正在分析内容...'})
                
                # 调用模型生成响应
                await self._generate_ai_response(image_urls, video_urls, audio_url, query)
            
        except Exception as e:
            print(f"处理查询失败: {str(e)}")
            traceback.print_exc()
            socketio.emit('error', {'message': f'处理查询失败: {str(e)}'})
        finally:
            socketio.emit('response_complete')
    
    def _send_user_message_to_frontend(self, query, media_files):
        """向前端发送用户消息，包括媒体内容"""
        try:
            # 默认消息
            message_data = {
                'type': 'user',
                'content': query or "",
                'has_combined': True
            }
            
            # 如果有媒体文件，添加媒体文件信息
            if media_files and media_files.get('has_media'):
                print(f"处理媒体文件: {media_files}")
                
                for file_info in media_files.get('files', []):
                    file_path = file_info.get('path')
                    file_type = file_info.get('type')
                    
                    if not file_path:
                        print("错误: 文件路径为空")
                        continue
                    
                    if not os.path.exists(file_path):
                        print(f"错误: 文件不存在: {file_path}")
                        continue
                    
                    try:
                        # 根据文件类型处理
                        if file_type == 'image':
                            # 处理图片
                            data_uri, _ = process_image_file(file_path, max_size=1024, quality=85)
                            message_data['media_type'] = 'image'
                            message_data['media_content'] = data_uri
                            print("成功处理图片文件")
                            break  # 前端目前只支持每条消息显示一个媒体文件
                            
                        elif file_type == 'video':
                            # 处理视频
                            data_uri, _ = process_media_file(file_path, media_type='video')
                            message_data['media_type'] = 'video'
                            message_data['media_content'] = data_uri
                            print("成功处理视频文件")
                            break  # 前端目前只支持每条消息显示一个媒体文件
                            
                        elif file_type == 'audio':
                            # 处理音频
                            data_uri, _ = process_media_file(file_path, media_type='audio')
                            message_data['media_type'] = 'audio'
                            message_data['media_content'] = data_uri
                            print("成功处理音频文件")
                            break  # 前端目前只支持每条消息显示一个媒体文件
                            
                    except Exception as e:
                        print(f"处理媒体文件失败: {str(e)}")
                        traceback.print_exc()
                        continue
            
            # 确保消息有实际内容后发送
            if message_data['content'].strip() or 'media_type' in message_data:
                print("发送用户消息到前端")
                socketio.emit('message', message_data)
            else:
                print("警告: 没有有效的消息内容可发送")
            
        except Exception as e:
            print(f"发送用户消息到前端失败: {str(e)}")
            traceback.print_exc()
    
    async def _generate_ai_response(self, image_urls, video_urls, audio_url, query):
        """使用qwen-max的function calling能力生成响应并处理工具调用"""
        try:
            # 检查是否有多模态内容
            has_multimodal = bool(image_urls or video_urls or audio_url)
            
            # 使用完整的消息历史，但确保格式一致
            conversation_history = []
            for msg in self.messages:
                sanitized_msg = msg.copy()
                # 确保content始终有正确的格式
                if 'content' in msg:
                    if msg.get('content') is None:
                        sanitized_msg['content'] = ""  # 将None替换为空字符串
                    elif isinstance(msg.get('content'), list):
                        # 列表格式的content保持不变
                        pass
                    elif isinstance(msg.get('content'), str):
                        # 确保所有的字符串内容在需要时转换为列表格式
                        if msg.get('role') == 'system':
                            sanitized_msg['content'] = [{'type': 'text', 'text': msg.get('content')}]
                conversation_history.append(sanitized_msg)
            
            print("调用qwen-max模型...")
            socketio.emit('status', {'message': '正在分析内容...'})
            
            # 将conversation_history转换为纯文本格式
            # 对于纯文本对话，使用简化的消息格式
            simple_history = []
            for msg in conversation_history:
                simple_msg = {"role": msg["role"]}
                
                # 处理content字段
                if isinstance(msg.get("content"), list):
                    # 从列表格式中提取文本
                    text_content = ""
                    for item in msg["content"]:
                        if item.get("type") == "text":
                            text_content += item.get("text", "")
                    simple_msg["content"] = text_content
                else:
                    # 直接使用字符串格式
                    simple_msg["content"] = msg.get("content", "")
                
                # 处理tool_calls字段
                if "tool_calls" in msg:
                    simple_msg["tool_calls"] = msg["tool_calls"]
                
                simple_history.append(simple_msg)
            
            print(f"简化对话历史：{json.dumps(simple_history, ensure_ascii=False)[:200]}...")
            
            # 使用流式API调用模型
            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=simple_history,
                tools=self.available_tools,
                tool_choice="auto",
                temperature=0.7,
                stream=True
            )
            
            # 处理流式响应
            buffer = ""
            tool_calls_info = []
            current_tool_call = None
            
            for chunk in response:
                # 处理文本内容
                if (hasattr(chunk, 'choices') and chunk.choices and 
                    hasattr(chunk.choices[0], 'delta') and 
                    hasattr(chunk.choices[0].delta, 'content') and 
                    chunk.choices[0].delta.content is not None):
                    
                    buffer += chunk.choices[0].delta.content
                    socketio.emit('message', {'content': chunk.choices[0].delta.content, 'type': 'assistant'})
                
                # 处理工具调用
                if (hasattr(chunk, 'choices') and chunk.choices and 
                    hasattr(chunk.choices[0], 'delta') and 
                    hasattr(chunk.choices[0].delta, 'tool_calls') and
                    chunk.choices[0].delta.tool_calls is not None):
                    
                    delta_tool_calls = chunk.choices[0].delta.tool_calls
                    for delta in delta_tool_calls:
                        # 确保工具调用ID不为None
                        if not hasattr(delta, 'id') or delta.id is None:
                            continue
                            
                        tool_call_id = delta.id
                        
                        # 查找或创建工具调用信息
                        found = False
                        for tool_info in tool_calls_info:
                            if tool_info['id'] == tool_call_id:
                                found = True
                                current_tool_call = tool_info
                                break
                        
                        if not found and tool_call_id:
                            current_tool_call = {
                                'id': tool_call_id,
                                'function': {
                                    'name': '',
                                    'arguments': ''
                                },
                                'type': 'function'
                            }
                            tool_calls_info.append(current_tool_call)
                        
                        # 更新工具名称
                        if hasattr(delta.function, 'name') and delta.function.name:
                            current_tool_call['function']['name'] = delta.function.name
                        
                        # 更新参数
                        if hasattr(delta.function, 'arguments') and delta.function.arguments:
                            current_tool_call['function']['arguments'] += delta.function.arguments
            
            # 添加助手消息到历史
            assistant_message = {
                "role": "assistant",
                "content": buffer if not tool_calls_info else None,
            }
            if tool_calls_info:
                assistant_message["tool_calls"] = tool_calls_info
            
            self.messages.append(assistant_message)
            self.save_history()
            
            # 执行工具调用
            if tool_calls_info:
                print(f"执行工具调用: {tool_calls_info}")
                
                for tool_info in tool_calls_info:
                    tool_name = tool_info['function']['name']
                    tool_call_id = tool_info['id']
                    
                    try:
                        socketio.emit('tool_status', {
                            'status': 'start',
                            'tool_name': tool_name,
                            'message': f'正在执行工具: {tool_name}...'
                        })
                        
                        # 解析参数
                        args_str = tool_info['function']['arguments']
                        
                        try:
                            tool_args = json.loads(args_str)
                        except json.JSONDecodeError:
                            # 尝试提取有效的JSON
                            start_idx = args_str.find('{')
                            end_idx = args_str.rfind('}') + 1
                            if start_idx >= 0 and end_idx > 0:
                                valid_json = args_str[start_idx:end_idx]
                                tool_args = json.loads(valid_json)
                            else:
                                raise ValueError(f"无法解析参数: {args_str}")
                        
                        # 特殊处理多模态内容分析工具
                        if tool_name == "analyze_multimodal_content":
                            # 确保工具参数包含最新的多模态内容，但只在有音频时添加音频参数
                            tool_args = {
                                'content': tool_args.get('content', query or ""),
                                'images': image_urls,
                                'videos': video_urls,
                            }
                            
                            # 只在有音频内容时添加音频参数
                            if audio_url:
                                tool_args['audio'] = audio_url
                                tool_args['audio_format'] = audio_format
                                
                            print(f"使用多模态内容: 文本='{tool_args['content']}', 图片数={len(tool_args['images'])}, 视频数={len(tool_args['videos'])}, 音频={'有' if audio_url else '无'}")
                        
                        # 调用工具
                        print(f"执行工具: {tool_name}，参数: {tool_args}")
                        result = await self.session.call_tool(tool_name, tool_args)
                        
                        # 处理结果
                        result_content = await self._extract_tool_result_content(result)
                        
                        # 添加工具结果到历史
                        self.messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": result_content
                        })
                        
                        # 处理并显示工具结果（非多模态分析工具）
                        await self._process_tool_result(result, tool_name, tool_call_id)
                        
                        socketio.emit('tool_status', {
                            'status': 'end',
                            'tool_name': tool_name,
                            'message': f'工具 {tool_name} 执行完成'
                        })
                        
                    except Exception as e:
                        error_message = f"工具执行错误: {str(e)}"
                        print(error_message)
                        traceback.print_exc()
                        socketio.emit('error', {'message': error_message})
                        
                        # 添加错误信息到历史
                        self.messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": error_message
                        })
                
                # 生成最终回应
                await self._generate_followup_response()
            
        except Exception as e:
            print(f"生成AI响应失败: {str(e)}")
            traceback.print_exc()
            socketio.emit('error', {'message': f'生成响应失败: {str(e)}'})

    async def _extract_tool_result_content(self, result):
        """从工具结果中提取内容"""
        try:
            if result is None:
                print("警告: 工具结果为None")
                return "工具返回空结果"
                
            if hasattr(result, 'content'):
                # 处理CallToolResult对象
                content_list = result.content
                if content_list is None:
                    print("警告: 工具结果content为None")
                    return "工具返回空内容"
                    
                result_content = ""
                for item in content_list:
                    if hasattr(item, 'text'):
                        result_content += item.text
                return result_content
            else:
                # 如果是字符串
                return str(result)
        except Exception as e:
            error_msg = f"提取工具结果内容时出错: {str(e)}"
            print(error_msg)
            traceback.print_exc()
            return error_msg
    
    async def _generate_followup_response(self):
        """生成模型的后续回应"""
        try:
            # 构建对话历史
            sanitized_messages = []
            for msg in self.messages:
                sanitized_msg = msg.copy()
                # 对于工具响应，确保content是字符串
                if msg.get('role') == 'tool':
                    if not isinstance(msg.get('content'), str):
                        if msg.get('content') is None:
                            sanitized_msg['content'] = ""
                        else:
                            sanitized_msg['content'] = json.dumps(msg.get('content'))
                    
                    # 检查是否包含音频转录内容
                    content = sanitized_msg.get('content', '')
                    if isinstance(content, str) and "[音频转录]" in content:
                        # 为Max模型添加额外的上下文，帮助它理解这是音频转录内容
                        sanitized_msg['content'] = f"以下是从音频文件中转录的内容:\n{content}"
                        print("为Max模型添加音频转录上下文")
                        
                # 确保系统消息格式一致
                elif msg.get('role') == 'system' and isinstance(msg.get('content'), str):
                    sanitized_msg['content'] = [{'type': 'text', 'text': msg.get('content')}]
                # 对于其他消息，确保content不为None
                elif msg.get('content') is None:
                    sanitized_msg['content'] = ""
                
                sanitized_messages.append(sanitized_msg)
            
            # 将sanitized_messages转换为纯文本格式
            simple_messages = []
            for msg in sanitized_messages:
                simple_msg = {"role": msg["role"]}
                
                # 处理content字段
                if isinstance(msg.get("content"), list):
                    # 从列表格式中提取文本
                    text_content = ""
                    for item in msg["content"]:
                        if item.get("type") == "text":
                            text_content += item.get("text", "")
                    simple_msg["content"] = text_content
                else:
                    # 直接使用字符串格式
                    simple_msg["content"] = msg.get("content", "")
                
                # 处理tool_calls字段
                if "tool_calls" in msg:
                    simple_msg["tool_calls"] = msg["tool_calls"]
                
                # 如果是tool消息，保留tool_call_id
                if msg.get("role") == "tool" and "tool_call_id" in msg:
                    simple_msg["tool_call_id"] = msg["tool_call_id"]
                
                simple_messages.append(simple_msg)
            
            # 请求模型生成后续回应
            print("请求后续回应...")
            print(f"后续对话历史：{json.dumps(simple_messages, ensure_ascii=False)[:200]}...")
            
            # 如果最后一条消息是工具消息，检查是否包含音频转录
            last_tool_msg = next((msg for msg in reversed(simple_messages) if msg.get('role') == 'tool'), None)
            if last_tool_msg and "[音频转录]" in last_tool_msg.get('content', ''):
                print("检测到最后一条工具消息包含音频转录内容，添加系统提示以指导Max模型")
                # 添加临时系统消息，指导Max模型如何处理音频转录内容
                simple_messages.insert(1, {
                    "role": "system",
                    "content": "用户上传了一段音频文件，已经转录为文本。请基于这段转录内容回应用户的查询或提供适当的反馈。"
                })
            
            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=simple_messages,
                temperature=0.7,
                stream=True
            )
            
            # 处理流式响应
            buffer = ""
            for chunk in response:
                if (hasattr(chunk, 'choices') and chunk.choices and 
                    hasattr(chunk.choices[0], 'delta') and 
                    hasattr(chunk.choices[0].delta, 'content') and 
                    chunk.choices[0].delta.content is not None):
                    
                    buffer += chunk.choices[0].delta.content
                    socketio.emit('message', {'content': chunk.choices[0].delta.content, 'type': 'assistant'})
            
            # 将后续回应添加到历史
            if buffer:
                self.messages.append({
                    "role": "assistant",
                    "content": buffer
                })
                self.save_history()
                
        except Exception as e:
            print(f"生成后续回应失败: {str(e)}")
            traceback.print_exc()
            socketio.emit('error', {'message': f'生成后续回应失败: {str(e)}'})

    async def cleanup(self):
        """清理资源"""
        try:
            await self.exit_stack.aclose()
        except Exception as e:
            print(f"清理资源时出错: {str(e)}")
            
# 创建全局客户端实例
client = MCPWebClient()

@app.route('/')
def index():
    """渲染主页"""
    return render_template('index.html')

@app.route('/static/outputs/<filename>')
def serve_output(filename):
    """提供静态文件访问"""
    return send_from_directory('static/outputs', filename)

@app.route('/static/uploads/<filename>')
def serve_upload(filename):
    """提供上传文件访问"""
    return send_from_directory('static/uploads', filename)

@app.route('/upload', methods=['POST'])
def upload_file():
    """处理文件上传"""
    try:
        print("收到文件上传请求")
        if 'file' not in request.files:
            print("错误: 请求中没有文件")
            return jsonify({'error': '没有找到文件'}), 400
            
        file = request.files['file']
        
        if file.filename == '':
            print("错误: 空文件名")
            return jsonify({'error': '没有选择文件'}), 400
            
        # 获取文件信息
        content_type = file.content_type
        print(f"文件内容类型: {content_type}")
        
        # 更严格地判断文件类型
        if content_type.startswith('image/'):
            file_type = 'image'
            # 额外验证图片格式
            valid_image_types = ['image/jpeg', 'image/png', 'image/gif', 'image/webp']
            if content_type not in valid_image_types:
                print(f"警告: 不常见的图片格式: {content_type}，但仍将作为图片处理")
        elif content_type.startswith('audio/'):
            file_type = 'audio'
            # 额外验证音频格式
            valid_audio_types = ['audio/mpeg', 'audio/mp3', 'audio/wav', 'audio/ogg']
            if content_type not in valid_audio_types:
                print(f"警告: 不常见的音频格式: {content_type}，但仍将作为音频处理")
        elif content_type.startswith('video/'):
            file_type = 'video'
            # 额外验证视频格式
            valid_video_types = ['video/mp4', 'video/webm', 'video/quicktime']
            if content_type not in valid_video_types:
                print(f"警告: 不常见的视频格式: {content_type}，但仍将作为视频处理")
        else:
            file_type = 'other'
            print(f"警告: 不支持的文件类型: {content_type}")
            
        print(f"处理文件: {file.filename}, 类型: {content_type}, 分类: {file_type}")
            
        # 生成唯一文件名
        filename = str(uuid.uuid4()) + str(int(time.time() * 1000)) + '_' + file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # 保存文件
        try:
            file.save(file_path)
            print(f"文件保存成功: {file_path}")
        except Exception as save_error:
            print(f"保存文件失败: {str(save_error)}")
            return jsonify({'error': f'保存文件失败: {str(save_error)}'}), 500
            
        # 验证文件是否已正确保存
        if not os.path.exists(file_path):
            print(f"错误: 文件保存后不存在: {file_path}")
            return jsonify({'error': '文件保存失败'}), 500
            
        file_size = os.path.getsize(file_path)
        print(f"文件大小: {file_size} 字节")
        
        # 返回文件信息
        response_data = {
            'success': True,
            'file_path': file_path,
            'file_url': f'/static/uploads/{filename}',
            'file_type': file_type,
            'file_size': file_size
        }
        print(f"上传成功: {response_data}")
        return jsonify(response_data)
        
    except Exception as e:
        print(f"文件上传处理错误: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': f'上传文件出错: {str(e)}'}), 500

@socketio.on('connect')
def handle_connect():
    """处理WebSocket连接"""
    emit('status', {'message': '已连接到服务器'})

@socketio.on('disconnect')
def handle_disconnect():
    """处理WebSocket断开连接"""
    print('客户端断开连接')

@socketio.on('message')
def handle_message(data):
    """处理接收到的消息"""
    query = data.get('message', '')
    media_files = data.get('media_files', None)
    
    try:
        # 使用全局事件循环
        future = asyncio.run_coroutine_threadsafe(
            client.process_query(query, media_files),
            loop
        )
        future.result()  # 等待结果
    except Exception as e:
        print(f"消息处理错误: {str(e)}")
        traceback.print_exc()
        socketio.emit('error', {'message': f'消息处理错误: {str(e)}'})

@app.route('/clear-history', methods=['POST'])
def clear_chat_history():
    """清除对话历史"""
    try:
        print("清除对话历史")
        client.clear_history()
        return jsonify({'success': True, 'message': '对话历史已清除'}), 200
    except Exception as e:
        print(f"清除对话历史失败: {str(e)}")
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'清除对话历史失败: {str(e)}'}), 500

def run_flask():
    """运行Flask应用"""
    try:
        socketio.run(app, host='0.0.0.0', port=30008, debug=False, use_reloader=False)
    except Exception as e:
        print(f"Flask应用运行错误: {str(e)}")

def run_event_loop():
    """运行事件循环"""
    try:
        loop.run_forever()
    except Exception as e:
        print(f"事件循环错误: {str(e)}")
    finally:
        loop.close()

if __name__ == "__main__":
    try:
        if len(sys.argv) < 2:
            print("用法: python bigmodel_client3_web.py <服务器脚本路径>")
            sys.exit(1)

        # 启动事件循环线程
        loop_thread = threading.Thread(target=run_event_loop, daemon=True)
        loop_thread.start()

        # 连接到服务器
        future = asyncio.run_coroutine_threadsafe(
            client.connect_to_server(sys.argv[1]),
            loop
        )
        if future.result():  # 等待连接完成
            # 启动Flask应用
            run_flask()
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"程序错误: {str(e)}")
    finally:
        # 清理资源
        future = asyncio.run_coroutine_threadsafe(
            client.cleanup(),
            loop
        )
        future.result()
        loop.call_soon_threadsafe(loop.stop)
