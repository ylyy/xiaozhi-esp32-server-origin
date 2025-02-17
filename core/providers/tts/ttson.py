import os
import uuid
import json
import logging
import requests
from datetime import datetime
from core.providers.tts.base import TTSProviderBase

logger = logging.getLogger(__name__)

class TTSProvider(TTSProviderBase):
    def __init__(self, config, delete_audio_file):
        super().__init__(config, delete_audio_file)
        self.token = config.get("token", "ht-44e0784309820c3d11649a14")
        self.voice_id = config.get("voice_id", 430)
        self.format = config.get("format", "wav")
        self.speed_factor = config.get("speed_factor", 1.0)
        self.pitch_factor = config.get("pitch_factor", 0)
        self.volume_change_db = config.get("volume_change_db", 0)
        
        # 可选择更快的服务器
        self.api_urls = [
            f"https://ht.ttson.cn:37284/flashsummary/tts?token={self.token}",
            f"https://u95167-bd74-2aef8085.westx.seetacloud.com:8443/flashsummary/tts?token={self.token}"
        ]

    def generate_filename(self, extension=".wav"):
        return os.path.join(self.output_file, f"tts-{datetime.now().date()}@{uuid.uuid4().hex}{extension}")

    async def text_to_speak(self, text, output_file):
        if not text:
            raise ValueError("Text cannot be empty")

        payload = {
            "voice_id": self.voice_id,
            "text": text,
            "to_lang": "ZH",
            "format": "wav",  # 强制使用wav格式
            "zip_level": 4,   # 设置采样率为16000
            "speed_factor": self.speed_factor,
            "pitch_factor": self.pitch_factor,
            "volume_change_dB": self.volume_change_db
        }

        last_error = None
        # 尝试两个API URL
        for api_url in self.api_urls:
            try:
                logger.info(f"Trying TTS API endpoint: {api_url}")
                response = requests.post(api_url, json=payload)
                
                if response.status_code != 200:
                    logger.error(f"API request failed with status code {response.status_code}: {response.text}")
                    continue

                response_json = response.json()
                if "url" not in response_json or "port" not in response_json or "voice_path" not in response_json:
                    logger.error(f"Invalid API response format: {response_json}")
                    continue

                # 构建音频获取URL
                audio_url = f"{response_json['url']}:{response_json['port']}/flashsummary/retrieveFileData?stream=True&token={self.token}&voice_audio_path={response_json['voice_path']}"
                
                logger.info(f"Downloading audio from: {audio_url}")
                audio_response = requests.get(audio_url)
                
                if audio_response.status_code != 200:
                    logger.error(f"Audio download failed with status code {audio_response.status_code}: {audio_response.text}")
                    continue

                # 验证内容类型
                content_type = audio_response.headers.get('content-type', '')
                if not content_type.startswith(('audio/', 'application/octet-stream')):
                    logger.error(f"Unexpected content type: {content_type}")
                    continue

                # 保存音频文件
                with open(output_file, "wb") as f:
                    f.write(audio_response.content)

                # 验证文件是否成功创建
                if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                    logger.info(f"Successfully generated audio file: {output_file}")
                    return
                else:
                    logger.error(f"Failed to create audio file or file is empty: {output_file}")
                    continue

            except Exception as e:
                last_error = str(e)
                logger.error(f"Error during TTS process: {str(e)}", exc_info=True)
                continue

        error_msg = f"Failed to generate audio with all available API endpoints. Last error: {last_error}"
        logger.error(error_msg)
        raise Exception(error_msg) 