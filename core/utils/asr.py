import time
import wave
import os
from abc import ABC, abstractmethod
import logging
from typing import Optional, Tuple, List
import uuid
import asyncio
import io
import webrtcvad
import struct
import aiohttp
from pydub import AudioSegment

import opuslib
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

logger = logging.getLogger(__name__)


class ASR(ABC):
    @abstractmethod
    def save_audio_to_file(self, opus_data: List[bytes], session_id: str) -> str:
        """解码Opus数据并保存为WAV文件"""
        pass

    @abstractmethod
    async def speech_to_text(self, opus_data: List[bytes], session_id: str) -> Tuple[Optional[str], Optional[str]]:
        """将语音数据转换为文本"""
        pass


class FunASR(ASR):
    def __init__(self, config: dict, delete_audio_file: bool):
        self.model_dir = config.get("model_dir")
        self.output_dir = config.get("output_dir")  # 修正配置键名
        self.delete_audio_file = delete_audio_file

        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)

        self.model = AutoModel(
            model=self.model_dir,
            vad_kwargs={"max_single_segment_time": 30000},
            disable_update=True,
            hub="hf"
            # device="cuda:0",  # 启用GPU加速
        )

    def save_audio_to_file(self, opus_data: List[bytes], session_id: str) -> str:
        """将Opus音频数据解码并保存为WAV文件"""
        file_name = f"asr_{session_id}_{uuid.uuid4()}.wav"
        file_path = os.path.join(self.output_dir, file_name)

        decoder = opuslib.Decoder(16000, 1)  # 16kHz, 单声道
        pcm_data = []

        for opus_packet in opus_data:
            try:
                pcm_frame = decoder.decode(opus_packet, 960)  # 960 samples = 60ms
                pcm_data.append(pcm_frame)
            except opuslib.OpusError as e:
                logger.error(f"Opus解码错误: {e}", exc_info=True)

        with wave.open(file_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 2 bytes = 16-bit
            wf.setframerate(16000)
            wf.writeframes(b"".join(pcm_data))

        return file_path

    async def speech_to_text(self, opus_data: List[bytes], session_id: str) -> Tuple[Optional[str], Optional[str]]:
        """语音转文本主处理逻辑"""
        file_path = None
        try:
            # 保存音频文件
            start_time = time.time()
            file_path = self.save_audio_to_file(opus_data, session_id)
            logger.debug(f"音频文件保存耗时: {time.time() - start_time:.3f}s | 路径: {file_path}")

            # 语音识别
            start_time = time.time()
            result = self.model.generate(
                input=file_path,
                cache={},
                language="auto",
                use_itn=True,
                batch_size_s=60,
            )
            text = rich_transcription_postprocess(result[0]["text"])
            logger.debug(f"语音识别耗时: {time.time() - start_time:.3f}s | 结果: {text}")

            return text, file_path

        except Exception as e:
            logger.error(f"语音识别失败: {e}", exc_info=True)
            return None, None

        finally:
            # 文件清理逻辑
            if self.delete_audio_file and file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    logger.debug(f"已删除临时音频文件: {file_path}")
                except Exception as e:
                    logger.error(f"文件删除失败: {file_path} | 错误: {e}")


class TTSonASR(ASR):
    def __init__(self, config: dict, delete_audio_file: bool):
        self.output_dir = config.get("output_dir")
        self.delete_audio_file = delete_audio_file
        self.silent_threshold = config.get("silent_threshold", 300)
        self.audio_head = config.get("audio_head", 1)
        self.max_duration = config.get("max_duration", 10)
        self.base_url = config.get("base_url", "http://srt.ttson.cn:34001")  # 默认使用HTTP
        self.max_retries = config.get("max_retries", 3)
        self.retry_delay = config.get("retry_delay", 1)
        self.verify_ssl = config.get("verify_ssl", False)  # 默认不验证SSL
        self.use_http = config.get("use_http", True)  # 默认使用HTTP
        
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 如果配置使用HTTP，确保URL使用HTTP
        if self.use_http and self.base_url.startswith("https://"):
            self.base_url = self.base_url.replace("https://", "http://")

    def save_audio_to_file(self, opus_data: List[bytes], session_id: str) -> str:
        """将Opus音频数据解码并保存为WAV文件"""
        file_name = f"asr_{session_id}_{uuid.uuid4()}.wav"
        file_path = os.path.join(self.output_dir, file_name)

        decoder = opuslib.Decoder(16000, 1)  # 16kHz, 单声道
        pcm_data = []

        for opus_packet in opus_data:
            try:
                pcm_frame = decoder.decode(opus_packet, 960)  # 960 samples = 60ms
                pcm_data.append(pcm_frame)
            except opuslib.OpusError as e:
                logger.error(f"Opus解码错误: {e}", exc_info=True)

        with wave.open(file_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 2 bytes = 16-bit
            wf.setframerate(16000)
            wf.writeframes(b"".join(pcm_data))

        return file_path

    async def speech_to_text(self, opus_data: List[bytes], session_id: str) -> Tuple[Optional[str], Optional[str]]:
        """语音转文本主处理逻辑"""
        file_path = None
        try:
            # 保存音频文件
            start_time = time.time()
            file_path = self.save_audio_to_file(opus_data, session_id)
            logger.debug(f"音频文件保存耗时: {time.time() - start_time:.3f}s | 路径: {file_path}")

            # 创建ASRService实例
            audio = AudioSegment.from_wav(file_path)
            generator = self._audio_chunk_generator(audio, chunk_duration_ms=600)

            asr_service = ASRService(
                silent_threshold=self.silent_threshold,
                audio_head=self.audio_head,
                max_duration=self.max_duration,
                base_url=self.base_url,
                max_retries=self.max_retries,
                retry_delay=self.retry_delay,
                verify_ssl=self.verify_ssl,
                use_http=self.use_http
            )
            asr_service.set_audio_generator(generator)

            # 运行语音识别
            text = await asr_service.get_transcription()
            logger.debug(f"语音识别耗时: {time.time() - start_time:.3f}s | 结果: {text}")

            return text, file_path

        except Exception as e:
            logger.error(f"语音识别失败: {e}", exc_info=True)
            return None, None

        finally:
            # 文件清理逻辑
            if self.delete_audio_file and file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    logger.debug(f"已删除临时音频文件: {file_path}")
                except Exception as e:
                    logger.error(f"文件删除失败: {file_path} | 错误: {e}")

    def _audio_chunk_generator(self, audio, chunk_duration_ms=600):
        for i in range(0, len(audio), chunk_duration_ms):
            chunk = audio[i:i + chunk_duration_ms]
            yield chunk
            time.sleep(chunk_duration_ms / 1000)


class ASRService:
    def __init__(self, silent_threshold=300, audio_head=1, max_duration=10, base_url=None, max_retries=3, retry_delay=1, verify_ssl=False, use_http=True):
        self.silent_threshold = silent_threshold
        self.max_duration = max_duration
        self.audio_head = audio_head  # 前多少ms不检测
        self.accumulated_audio = AudioSegment.empty()
        self.text = ""
        self.session = None
        self.audio_generator = None
        self.vad = webrtcvad.Vad(2)  # 创建VAD对象，设置激进度为2
        self.frame_duration = 30
        self.is_in_silence = False
        self._base_url = base_url or "http://srt.ttson.cn:34001"  # 默认使用HTTP
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.verify_ssl = verify_ssl
        self.use_http = use_http
        
        # 如果配置使用HTTP，确保URL使用HTTP
        if self.use_http and self._base_url.startswith("https://"):
            self._base_url = self._base_url.replace("https://", "http://")

    @property
    def base_url(self):
        return f"{self._base_url}/predict"

    @property
    def headers(self):
        return {'Accept': 'application/json'}

    def set_audio_generator(self, generator):
        self.audio_generator = generator

    async def create_session(self):
        if self.session is None or self.session.closed:
            # 创建SSL上下文
            ssl_context = None if self.use_http else False if not self.verify_ssl else True
            connector = aiohttp.TCPConnector(ssl=ssl_context)
            self.session = aiohttp.ClientSession(headers=self.headers, connector=connector)

    async def close_session(self):
        if self.session and not self.session.closed:
            await self.session.close()

    async def get_transcription(self):
        await self.create_session()
        try:
            return await asyncio.wait_for(self.process_audio(), timeout=self.max_duration)
        except asyncio.TimeoutError:
            logger.warning(f"识别超时（{self.max_duration}秒），返回当前结果")
            return self.text
        finally:
            await self.close_session()

    async def upload(self, audio):
        if not audio or len(audio) == 0:
            logger.error("音频数据为空")
            return None

        buffer = io.BytesIO()
        try:
            audio.export(buffer, format="wav")
        except Exception as e:
            logger.error(f"音频导出失败: {e}")
            return None

        total_size = buffer.tell()
        logger.debug(f"音频数据大小: {total_size} 字节")

        for retry in range(self.max_retries):
            try:
                # 每次重试时重新创建buffer和FormData
                buffer.seek(0)
                form = aiohttp.FormData()
                form.add_field('content', buffer, filename='audio.wav', content_type='audio/wav')

                logger.debug(f"开始第 {retry + 1} 次请求，目标URL: {self.base_url}")
                async with self.session.post(self.base_url, data=form, timeout=10) as response:
                    if response.status != 200:
                        response_text = await response.text()
                        logger.error(f'请求失败，状态码：{response.status}，响应内容：{response_text}')
                        if retry < self.max_retries - 1:
                            await asyncio.sleep(self.retry_delay)
                            continue
                        return None

                    try:
                        data = await response.json()
                        transcription = data.get('transcription')
                        if transcription:
                            logger.debug(f"识别成功: {transcription}")
                            return transcription
                        else:
                            logger.warning("响应中没有识别结果")
                            return None
                    except Exception as e:
                        logger.error(f"解析响应JSON失败: {e}")
                        return None

            except asyncio.TimeoutError:
                logger.error(f"请求超时 (重试 {retry + 1}/{self.max_retries})")
            except aiohttp.ClientSSLError as e:
                if self.use_http:
                    logger.warning(f"SSL错误，尝试使用HTTP连接: {e}")
                    self._base_url = self._base_url.replace("https://", "http://")
                    continue
                else:
                    logger.error(f"SSL错误: {e} (重试 {retry + 1}/{self.max_retries})")
            except aiohttp.ClientConnectorError as e:
                if "SSL" in str(e) and self.use_http:
                    logger.warning(f"SSL连接错误，尝试使用HTTP连接: {e}")
                    self._base_url = self._base_url.replace("https://", "http://")
                    continue
                logger.error(f"连接错误: {e} (重试 {retry + 1}/{self.max_retries})")
            except aiohttp.ClientError as e:
                logger.error(f"客户端错误: {e} (重试 {retry + 1}/{self.max_retries})")
            except Exception as e:
                logger.error(f"未预期的错误: {e} (重试 {retry + 1}/{self.max_retries})")
            
            if retry < self.max_retries - 1:
                await asyncio.sleep(self.retry_delay)
            
        logger.error("所有重试都失败了")
        return None

    async def process_audio(self):
        if not self.audio_generator:
            logger.error("音频生成器未设置")
            return None

        start_time = time.time()
        silent_duration = 0
        chunks_processed = 0
        
        while time.time() - start_time <= self.max_duration:
            try:
                chunk = next(self.audio_generator, None)
                if chunk is None:
                    logger.debug("音频数据处理完成")
                    break

                chunks_processed += 1
                self.accumulated_audio += chunk
                
                if time.time() - start_time <= self.audio_head:
                    continue
                    
                frames = self.prepare_frame(chunk)
                if not frames:
                    logger.warning("没有有效的音频帧")
                    continue

                # 检查每一帧是否为静音
                for frame in frames:
                    is_current_frame_silent = self.is_silent(frame)
                    current_time = time.time()
                    if is_current_frame_silent:
                        if not self.is_in_silence:
                            self.is_in_silence = True
                            logger.debug(f"检测到静音开始 ({current_time - start_time:.2f}s)")
                            self.text = await self.upload(self.accumulated_audio)
                        silent_duration += self.frame_duration
                        if silent_duration >= self.silent_threshold:
                            logger.debug(f"静音持续时间达到阈值，识别结束 ({current_time - start_time:.2f}s)")
                            return self.text
                    else:
                        if self.is_in_silence:
                            self.is_in_silence = False
                            logger.debug("检测到语音恢复")
                        silent_duration = 0

            except Exception as e:
                logger.error(f"处理音频chunk时发生错误: {e}")
                return self.text

        logger.debug(f"处理了 {chunks_processed} 个音频块，总时长: {time.time() - start_time:.2f}s")
        return self.text

    def prepare_frame(self, audio_chunk):
        # 确保音频是单声道、16-bit PCM 编码
        if audio_chunk.channels > 1:
            audio_chunk = audio_chunk.set_channels(1)
        if audio_chunk.sample_width != 2:
            audio_chunk = audio_chunk.set_sample_width(2)

        if audio_chunk.frame_rate != 16000:
            audio_chunk = audio_chunk.set_frame_rate(16000)

        # 计算每帧的样本数
        samples_per_frame = int(audio_chunk.frame_rate * (self.frame_duration / 1000.0))

        # 将音频数据转换为 PCM 数据
        pcm_data = audio_chunk.get_array_of_samples()

        # 分割为固定大小的帧
        frames = []
        for i in range(0, len(pcm_data), samples_per_frame):
            frame = pcm_data[i:i + samples_per_frame]
            if len(frame) == samples_per_frame:  # 只使用完整的帧
                frames.append(struct.pack('%dh' % len(frame), *frame))

        return frames

    def is_silent(self, frame):
        return not self.vad.is_speech(frame, 16000)


def create_instance(class_name: str, *args, **kwargs) -> ASR:
    """工厂方法创建ASR实例"""
    cls_map = {
        "FunASR": FunASR,
        "TTSonASR": TTSonASR,
        # 可扩展其他ASR实现
    }

    if cls := cls_map.get(class_name):
        return cls(*args, **kwargs)
    raise ValueError(f"不支持的ASR类型: {class_name}")
