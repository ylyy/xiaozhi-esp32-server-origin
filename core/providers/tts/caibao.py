import requests
import time
import os
import uuid
import json
import logging
from datetime import datetime
from core.providers.tts.base import TTSProviderBase

logger = logging.getLogger(__name__)

class TTSProvider(TTSProviderBase):
    def __init__(self, config, delete_audio_file):
        """
        初始化CaibaoTTS
        :param config: 配置字典，包含api_key、base_url等配置信息
        """
        super().__init__(config, delete_audio_file)
        self.api_key = config.get('api_key')
        self.base_url = config.get('base_url', "https://caibaotts.tingwu.co/api")
        self.voice_id = config.get('voice_id', "S_5tnqkzou13bb")
        logger.info(f"初始化CaibaoTTS: api_key={self.api_key}, voice_id={self.voice_id}")
    
    def generate_filename(self, extension=".wav"):
        """生成输出文件名"""
        return os.path.join(self.output_file, f"tts-{datetime.now().date()}@{uuid.uuid4().hex}{extension}")
    
    async def text_to_speak(self, text, output_file):
        """
        将文本转换为语音
        :param text: 要转换的文本
        :param output_file: 输出文件路径
        """
        try:
            logger.info(f"开始处理文本: {text}")
            # 创建任务
            create_result = self.create_task(text)
            if create_result["error"] != 0:
                error_msg = f"创建任务失败: {create_result.get('msg', '未知错误')}, 完整响应: {json.dumps(create_result, ensure_ascii=False)}"
                logger.error(error_msg)
                raise Exception(error_msg)
            
            task_id = create_result["task"]["task_id"]
            
            # 等待任务完成
            result = self.wait_for_completion(task_id)
            if not result:
                raise Exception("等待任务完成时发生错误")
            
            # 获取音频URL并下载
            audio_url = result["task"].get("audio_url")
            if not audio_url:
                error_msg = f"任务完成但未返回音频URL，完整响应: {json.dumps(result, ensure_ascii=False)}"
                logger.error(error_msg)
                raise Exception(error_msg)
                
            audio_response = requests.get(audio_url)
            if audio_response.status_code != 200:
                raise Exception(f"下载音频失败，状态码: {audio_response.status_code}")
            
            # 保存音频文件
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, "wb") as f:
                f.write(audio_response.content)
            
            
        except Exception as e:
            error_msg = f"TTS转换失败: {str(e)}"
            logger.error(error_msg)
            # 不在这里抛出异常，让基类的重试机制处理
            if os.path.exists(output_file):
                os.remove(output_file)
    
    def create_task(self, text: str, language: str = "中文"):
        """创建TTS任务"""
        try:
            url = f"{self.base_url}/zbtask.php"
            params = {
                "action": "create_caibao_task",
                "apikey": self.api_key
            }
            data = {
                "language": language,
                "text": text,
                "voice_id": self.voice_id,
                "is_cache": 1
            }
            
            # logger.info(f"发送TTS请求: URL={url}, params={json.dumps(params, ensure_ascii=False)}, data={json.dumps(data, ensure_ascii=False)}")
            response = requests.post(url, params=params, data=data)
            response.raise_for_status()  # 检查HTTP错误
            
            result = response.json()
            # logger.info(f"TTS API响应: {json.dumps(result, ensure_ascii=False)}")
            return result
            
        except requests.exceptions.RequestException as e:
            error_msg = f"API请求失败: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)
        except ValueError as e:
            error_msg = f"解析API响应失败: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)
    
    def get_task_detail(self, task_id: str):
        """获取任务详情"""
        try:
            url = f"{self.base_url}/zbtask.php"
            params = {
                "action": "get_task_detail",
                "task_id": task_id,
                "apikey": self.api_key
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            result = response.json()
            # logger.info(f"获取任务详情响应: {json.dumps(result, ensure_ascii=False)}")
            return result
            
        except requests.exceptions.RequestException as e:
            error_msg = f"获取任务详情失败: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)
    
    def wait_for_completion(self, task_id: str, timeout: int = 300, interval: int = 0.3):
        """等待任务完成"""
        start_time = time.time()
        while True:
            if time.time() - start_time > timeout:
                error_msg = "任务等待超时"
                logger.error(error_msg)
                raise TimeoutError(error_msg)
            
            try:
                result = self.get_task_detail(task_id)
                if result["error"] != 0:
                    error_msg = f"获取任务详情失败: {result.get('msg', '未知错误')}, 完整响应: {json.dumps(result, ensure_ascii=False)}"
                    logger.error(error_msg)
                    raise Exception(error_msg)
                
                task = result["task"]
                task_status = task.get("task_status")
                
                if task_status == "1" or task_status == 1:  # 任务完成
                    if not task.get("audio_url"):
                        error_msg = f"任务完成但未返回音频URL，完整响应: {json.dumps(result, ensure_ascii=False)}"
                        logger.error(error_msg)
                        raise Exception(error_msg)
                    # logger.info(f"任务完成: {task_id}, 完整响应: {json.dumps(result, ensure_ascii=False)}")
                    return result
                elif task_status == "2" or task_status == 2:  # 任务失败
                    error_msg = f"任务执行失败，错误信息: {task.get('error_msg', '未知错误')}, 完整响应: {json.dumps(result, ensure_ascii=False)}"
                    logger.error(error_msg)
                    raise Exception(error_msg)
                elif task_status == "3" or task_status == 3:  # 任务处理中
                    # logger.info(f"任务进行中: {task_id}, 状态: {task_status}")
                    pass
                else:
                    # logger.info(f"未知任务状态: {task_id}, 状态: {task_status}, 完整响应: {json.dumps(result, ensure_ascii=False)}")
                    pass
                
            except Exception as e:
                error_msg = f"检查任务状态时出错: {str(e)}"
                logger.error(error_msg)
                raise Exception(error_msg)
            
            time.sleep(interval) 