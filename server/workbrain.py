import os
import base64
import json
import asyncio
import tempfile
import requests
from urllib.parse import urlparse, unquote
from pathlib import PurePosixPath
import logging
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

# DashScope相关导入
import dashscope
from dashscope.audio.tts_v2 import SpeechSynthesizer, ResultCallback, AudioFormat
from dashscope import ImageSynthesis, VideoSynthesis

# 导入OpenAI兼容客户端用于调用Qwen-Omni
from openai import OpenAI

# 加载环境变量
load_dotenv()

# 获取API Key
api_key = os.getenv('DASHSCOPE_API_KEY')
if not api_key:
    raise ValueError("请设置DASHSCOPE_API_KEY环境变量")

# 设置DashScope API Key
dashscope.api_key = api_key

# 初始化OpenAI兼容客户端
client = OpenAI(
    api_key=api_key, 
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 创建FastMCP服务
mcp = FastMCP("dashscope-multimodal")

# 确保输出目录存在
output_dir = "static/outputs"
os.makedirs(output_dir, exist_ok=True)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("dashscope-service")

@mcp.tool("analyze_multimodal_content")
async def analyze_multimodal_content(content: str = "", images: list = [], videos: list = [], audio: str = "", audio_format: str = "mp3") -> str:
    """使用Qwen-Omni模型分析多模态内容（图片、视频、音频和文本）
    
    Args:
        content: 文本内容
        images: 图片URL或Base64列表
        videos: 视频URL或Base64列表
        audio: 音频URL或Base64
        audio_format: 音频格式（如mp3, wav等）
        
    Returns:
        str: 分析结果文本
    """
    logger.info(f"开始分析多模态内容: 文本长度={len(content)}, 图片数量={len(images)}, 视频数量={len(videos)}, 音频={'有' if audio else '无'}, 音频格式={audio_format if audio else 'N/A'}")
    
    try:
        # 构建消息内容
        message_content = []
        
        # 添加文本内容
        if content:
            message_content.append({
                "type": "text",
                "text": content
            })
        
        # 添加图片内容
        for img_url in images:
            # 检查图片URL是否有效
            if img_url and len(img_url) > 100:  # 确保URL有一定长度
                message_content.append({
                    "type": "image_url",
                    "image_url": {"url": img_url}
                })
                logger.info(f"添加图片: URL长度={len(img_url)}")
        
        # 添加视频内容
        for video_url in videos:
            # 检查视频URL是否有效
            if video_url and len(video_url) > 100:  # 确保URL有一定长度
                message_content.append({
                    "type": "video_url",
                    "video_url": {"url": video_url}
                })
                logger.info(f"添加视频: URL长度={len(video_url)}")
        
        # 添加音频内容（如果有）
        has_audio = False
        if audio and len(audio) > 100:  # 确保音频URL有一定长度
            # 使用正确的input_audio格式
            message_content.append({
                "type": "input_audio",
                "input_audio": {
                    "data": audio,
                    "format": audio_format
                }
            })
            has_audio = True
            logger.info(f"添加音频: URL长度={len(audio)}, 格式={audio_format}")
            
            # 对于音频，添加明确的指示给模型
            message_content.append({
                "type": "text",
                "text": "请帮我转录并理解这段音频内容，并在回复开头注明'[音频转录]:'。"
            })
        
        # 确保有内容可分析
        if not message_content:
            return "未提供任何可分析的内容"
        
        # 调用Qwen-Omni模型
        logger.info("调用Qwen-Omni模型进行分析...")
        try:
            # 设置适当的系统提示语
            system_message = "你是一个擅长理解多种模态内容的助手。"
            if has_audio:
                system_message += "当分析音频时，请先进行准确的转录，然后提供对内容的理解和分析。请在回复开头标注[音频转录]。"
            else:
                system_message += "当分析图片、视频或音频时，请提供基于你观察到的内容的详细描述。"
            
            # 尝试使用流式响应
            response = client.chat.completions.create(
                model="qwen2.5-omni-7b",
                messages=[
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": system_message}]
                    },
                    {
                        "role": "user",
                        "content": message_content
                    }
                ],
                temperature=0.7,
                max_tokens=1500,
                stream=True  # 设置为True以启用流式响应
            )
            
            # 处理流式响应
            full_response = ""
            for chunk in response:
                if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response += content
            
            if not full_response:
                raise ValueError("流式响应未返回内容，尝试非流式请求")
            
            # 打印完整响应到控制台，帮助观察
            logger.info(f"Omni模型分析结果: {full_response}")
            
            # 对于音频，如果模型没有自动添加标记，我们手动添加
            if has_audio and not full_response.startswith("[音频转录]"):
                full_response = f"[音频转录]: {full_response}"
                
            logger.info(f"分析完成，流式结果长度: {len(full_response)}")
            return full_response
            
        except Exception as stream_error:
            # 如果流式响应失败，尝试非流式请求
            logger.warning(f"流式响应失败: {str(stream_error)}，尝试非流式请求")
            
            # 非流式请求
            non_stream_response = client.chat.completions.create(
                model="qwen2.5-omni-7b",
                messages=[
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": system_message}]
                    },
                    {
                        "role": "user",
                        "content": message_content
                    }
                ],
                temperature=0.7,
                max_tokens=1500,
                stream=False
            )
            
            result = non_stream_response.choices[0].message.content
            
            # 打印完整响应到控制台，帮助观察
            logger.info(f"Omni模型分析结果(非流式): {result}")
            
            # 对于音频，如果模型没有自动添加标记，我们手动添加
            if has_audio and not result.startswith("[音频转录]"):
                result = f"[音频转录]: {result}"
                
            logger.info(f"分析完成，非流式结果长度: {len(result)}")
            return result
        
    except Exception as e:
        error_msg = f"多模态内容分析失败: {str(e)}"
        logger.error(error_msg)
        logger.exception(e)
        return error_msg

@mcp.tool("generate_speech_stream")
async def generate_speech_stream(text: str, voice: str = "longxiaochun") -> str:
    """生成语音流（返回音频文件路径）
    
    Args:
        text: 要转换为语音的文本
        voice: 语音角色名称，默认为龙小春
        
    Returns:
        str: 包含音频信息的JSON字符串
    """
    logger.info(f"开始生成语音: text='{text}', voice='{voice}'")
    
    if not text:
        return json.dumps({
            "type": "error",
            "message": "文本内容为空"
        })
    
    try:
        # 使用TTS生成语音
        synthesizer = SpeechSynthesizer(
            model="cosyvoice-v1",
            voice=voice
        )
        
        # 生成音频数据
        logger.info("调用DashScope语音合成API...")
        audio_data = synthesizer.call(text)
        
        # 保存到文件
        filename = f"speech_{hash(text) & 0xffffffff}.mp3"
        output_path = os.path.join(output_dir, filename)
        
        with open(output_path, 'wb') as f:
            f.write(audio_data)
            
        # 生成web路径
        web_path = f"/static/outputs/{filename}"
        
        logger.info(f"语音生成成功: {output_path}")
        
        # 返回结构化JSON
        result = {
            "type": "audio",
            "url": web_path,
            "text": text,
            "voice": voice,
            "description": f"使用{voice}生成的语音"
        }
        
        return json.dumps(result)
        
    except Exception as e:
        logger.error(f"语音生成失败: {str(e)}")
        return json.dumps({
            "type": "error",
            "message": f"语音生成失败: {str(e)}"
        })

@mcp.tool("generate_image")
async def generate_image(prompt: str, size: str = "1024*1024", n: int = 1) -> str:
    """生成图片"""
    if not prompt:
        return json.dumps({"type": "error", "message": "提示词为空"})
        
    try:
        # 调用API生成图片
        response = ImageSynthesis.call(
            model="wanx2.1-t2i-turbo",
            prompt=prompt,
            n=n,
            size=size
        )
        
        if response.status_code == 200:
            # 下载第一张图片
            image_url = response.output.results[0].url
            file_name = PurePosixPath(unquote(urlparse(image_url).path)).parts[-1]
            output_path = os.path.join(output_dir, file_name)
            
            with open(output_path, 'wb+') as f:
                f.write(requests.get(image_url).content)
            
            # 返回结构化结果
            return json.dumps({
                "type": "image",
                "url": f"/static/outputs/{file_name}",
                "description": f"基于提示词'{prompt}'生成的图片"
            })
        else:
            return json.dumps({
                "type": "error", 
                "message": f"图片生成失败: {response.message}"
            })
            
    except Exception as e:
        return json.dumps({
            "type": "error",
            "message": f"图片生成出错: {str(e)}"
        })

@mcp.tool("generate_video")
async def generate_video(prompt: str, size: str = "1280*720") -> str:
    """生成视频（返回视频文件路径）
    
    Args:
        prompt: 视频生成提示词
        size: 视频尺寸，格式为"宽*高"
        
    Returns:
        str: 包含视频信息的JSON字符串
    """
    logger.info(f"开始生成视频: prompt='{prompt}', size='{size}'")
    
    if not prompt:
        return json.dumps({
            "type": "error",
            "message": "提示词为空"
        })
        
    try:
        # 调用DashScope的视频生成API
        logger.info("调用DashScope视频生成API...")
        response = VideoSynthesis.call(
            model='wanx2.1-t2v-turbo',
            prompt=prompt,
            size=size
        )
        
        # 检查响应
        if response.status_code == 200:
            # 获取视频URL
            video_url = response.output.video_url
            
            # 生成视频文件名
            file_name = f"video_{hash(prompt) & 0xffffffff}.mp4"
            output_path = os.path.join(output_dir, file_name)
            
            # 下载视频
            logger.info(f"下载视频: {video_url}")
            with open(output_path, 'wb+') as f:
                f.write(requests.get(video_url).content)
            
            # 生成web路径
            web_path = f"/static/outputs/{file_name}"
            
            logger.info(f"视频生成成功: {output_path}")
            
            # 返回结构化JSON
            result = {
                "type": "video",
                "url": web_path,
                "prompt": prompt,
                "size": size,
                "description": f"基于提示词'{prompt}'生成的视频"
            }
            
            return json.dumps(result)
        else:
            error_msg = f"视频生成失败: {response.code}, {response.message}"
            logger.error(error_msg)
            return json.dumps({
                "type": "error",
                "message": error_msg
            })
            
    except Exception as e:
        logger.error(f"视频生成出错: {str(e)}")
        return json.dumps({
            "type": "error",
            "message": f"视频生成出错: {str(e)}"
        })

# 运行服务器
if __name__ == "__main__":
    logger.info("启动DashScope多模态服务...")
    mcp.run()
