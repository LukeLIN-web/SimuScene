import os
import asyncio
from openai import AsyncOpenAI
import json
import time
from collections import deque
from typing import Deque, Optional, List, Dict, Any, Set, Tuple
import logging
from tqdm.asyncio import tqdm as tqdm_asyncio
import glob
import pandas as pd
import hashlib
from datasets import Dataset
import numpy as np
from PIL import Image
from io import BytesIO
import base64
from qwen_vl_utils import process_vision_info

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def detect_file_format(file_path: str) -> str:
    """Detect file format based on extension."""
    extension = os.path.splitext(file_path)[1].lower()
    if extension == '.parquet':
        return 'parquet'
    elif extension == '.jsonl':
        return 'jsonl'
    else:
        raise ValueError(f"Unsupported file format: {extension}. Supported formats: .jsonl, .parquet")

def read_data_file(file_path: str) -> pd.DataFrame:
    """Read data file in either JSONL or parquet format."""
    file_format = detect_file_format(file_path)
    
    if file_format == 'parquet':
        logger.debug(f"Reading parquet file with HuggingFace datasets: {file_path}")
        
        # Use HuggingFace datasets to read parquet with proper nested column support
        dataset = Dataset.from_parquet(file_path)
        # Convert back to pandas DataFrame
        df = dataset.to_pandas()
        return df
        
    elif file_format == 'jsonl':
        logger.debug(f"Reading JSONL file: {file_path}")
        return pd.read_json(file_path, lines=True)

def write_data_file(df: pd.DataFrame, file_path: str) -> None:
    """Write DataFrame to file in either JSONL or parquet format based on extension."""
    file_format = detect_file_format(file_path)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    if file_format == 'parquet':
        logger.debug(f"Writing parquet file with HuggingFace datasets: {file_path}")
        
        # Convert pandas DataFrame to HuggingFace Dataset
        # HuggingFace datasets handle nested columns natively without flattening
        dataset = Dataset.from_pandas(df)
        dataset.to_parquet(file_path)
        
    elif file_format == 'jsonl':
        logger.debug(f"Writing JSONL file: {file_path}")
        df.to_json(file_path, orient='records', lines=True)

def get_server_ips(model: str, backend: str = None) -> Tuple[List[str], int]:
    """Get list of server IPs and port from the most recent complete server instances file for a specific model and backend
    
    Args:
        model: Model name to search for
        backend: Backend type ('vllm' or 'sglang'). If None, will try to find any backend
        
    Returns:
        Tuple of (list of IPs, port number)
    """
    # Get the directory where this utils.py file is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    server_status_dir = os.path.join(current_dir, "server_status")
    
    # Clean model name for filename matching (replace / and - with _)
    model_clean = model.replace('/', '_').replace('-', '_')
    
    # Build search pattern based on backend
    if backend:
        # Search for specific backend
        search_pattern = os.path.join(server_status_dir, f"server_instances_complete_{backend}_{model_clean}_*.json")
    else:
        # Search for any backend with this model
        search_pattern = os.path.join(server_status_dir, f"server_instances_complete_*_{model_clean}_*.json")
    
    json_files = glob.glob(search_pattern)
    
    if not json_files:
        # Fallback: try to find any complete server instances file and filter by model (and backend if specified)
        fallback_pattern = os.path.join(server_status_dir, "server_instances_complete_*.json")
        all_files = glob.glob(fallback_pattern)
        
        for file in all_files:
            try:
                with open(file, 'r') as f:
                    server_info = json.load(f)
                
                # Check if any server in this file matches our model and backend
                matching_servers = [
                    info for info in server_info 
                    if info.get('model') == model and (not backend or info.get('backend') == backend)
                ]
                if matching_servers:
                    json_files = [file]
                    detected_backend = matching_servers[0].get('backend', 'unknown')
                    logger.info(f"Found servers for model '{model}' with backend '{detected_backend}' in fallback file: {file}")
                    break
            except Exception as e:
                logger.warning(f"Error reading file {file}: {e}")
                continue
        
        if not json_files:
            backend_msg = f" and backend '{backend}'" if backend else ""
            raise RuntimeError(f"No server instances file found for model '{model}'{backend_msg} in {search_pattern}")
    
    # Get the most recent file
    latest_file = max(json_files, key=os.path.getctime)
    
    with open(latest_file, 'r') as f:
        server_info = json.load(f)
    
    # Filter servers by model and backend, extract IPs and port
    ips = []
    port = None
    detected_backend = None
    
    for info in server_info:
        # Check if this server entry matches our model and backend
        if info.get('model') == model and (not backend or info.get('backend') == backend):
            if 'ip' in info:
                ips.append(info['ip'])
                # Get port from server info (prefer this over defaults)
                if port is None:
                    port = int(info.get('port', 8000))  # Default to 8000 if not specified
                # Track detected backend
                if detected_backend is None:
                    detected_backend = info.get('backend', 'unknown')
    
    if not ips:
        backend_msg = f" and backend '{backend}'" if backend else ""
        raise RuntimeError(f"No IPs found for model '{model}'{backend_msg} in server instances file {latest_file}")
    
    # If no port was found in server info, use backend-specific defaults
    if port is None:
        if detected_backend == 'sglang':
            port = 30000
        else:  # vllm or unknown
            port = 8000
        logger.info(f"No port found in server info, using default port {port} for backend '{detected_backend}'")
    
    logger.info(f"Found {len(ips)} server instances for model '{model}' with backend '{detected_backend}' on port {port}: {ips}")
    return ips, port

class RateLimiter:
    def __init__(self, max_window_size: int):
        self.max_window_size = max_window_size
        self.semaphore = asyncio.Semaphore(max_window_size)
        logger.debug(f"Created rate limiter with max_window_size={max_window_size}")
    
    async def acquire(self):
        await self.semaphore.acquire()
        logger.debug(f"Successfully acquired slot. Available slots: {self.semaphore._value}/{self.max_window_size}")
    
    async def release(self):
        self.semaphore.release()
        logger.debug(f"Successfully released slot. Available slots: {self.semaphore._value}/{self.max_window_size}")

class RoundRobinClient:
    def __init__(self, ips: List[str], port: int, api_key: str, timeout: int, rate_limiters: List[RateLimiter]):
        self.ips = ips
        self.current_index = 0
        self.port = port
        self.api_key = api_key
        self.clients = [
            AsyncOpenAI(
                base_url=f"http://{ip}:{port}/v1",
                api_key=api_key,
                timeout=timeout,
            ) for ip in ips
        ]
        self.rate_limiters = rate_limiters
        logger.info(f"Initialized RoundRobinClient with {len(ips)} instances on port {port}")
        for i, ip in enumerate(ips):
            logger.debug(f"Instance {i}: {ip}:{port} with rate limit {rate_limiters[i].max_window_size}")
    
    def _hash_prompt_to_instance(self, prompt_content: str) -> int:
        """Hash prompt content to consistently route to the same server instance."""
        # Create a hash of the prompt content
        prompt_hash = hashlib.md5(str(prompt_content).encode('utf-8')).hexdigest()
        # Convert to integer and mod by number of instances
        instance_idx = int(prompt_hash, 16) % len(self.clients)
        logger.debug(f"Prompt hash routed to instance {instance_idx}")
        return instance_idx
    
    async def get_consistent_client_for_prompt(self, prompt_content: str) -> tuple[AsyncOpenAI, RateLimiter, int]:
        """Get a client consistently routed for the same prompt to enable prefix reuse."""
        target_instance = self._hash_prompt_to_instance(prompt_content)
        target_limiter = self.rate_limiters[target_instance]
        target_client = self.clients[target_instance]
        
        logger.debug(f"Consistent routing: prompt -> instance {target_instance} ({self.ips[target_instance]}:{self.port})")
        
        # Acquire slot on the target instance (wait if necessary)
        await target_limiter.acquire()
        logger.debug(f"Acquired slot on instance {target_instance} for prefix reuse")
        
        return target_client, target_limiter, target_instance

    async def get_next_available_client(self) -> tuple[AsyncOpenAI, RateLimiter]:
        # Find the instance with the most available slots
        max_available = -1
        best_client = None
        best_limiter = None
        best_index = -1
        
        # Log current state of all instances
        logger.debug("Current instance states:")
        for i in range(len(self.clients)):
            available = self.rate_limiters[i].semaphore._value
            used = self.rate_limiters[i].max_window_size - available
            logger.debug(f"Instance {i}: {self.ips[i]}:{self.port} - Used: {used}, Available: {available}/{self.rate_limiters[i].max_window_size}")
        
        # Check all instances to find the one with most available slots
        for i in range(len(self.clients)):
            available = self.rate_limiters[i].semaphore._value
            if available > max_available and available > 0:
                max_available = available
                best_client = self.clients[i]
                best_limiter = self.rate_limiters[i]
                best_index = i
                logger.debug(f"Selected instance {best_index} ({self.ips[best_index]}:{self.port}) with {max_available} available slots")
        
        if best_client is not None:
            # Found an instance with available capacity
            await best_limiter.acquire()
            return best_client, best_limiter
        else:
            # All instances are at capacity, distribute waiting requests across all instances
            logger.debug("All instances at capacity, distributing wait across all instances")
            
            # Create a list of all instances that we can wait on
            wait_tasks = []
            for i in range(len(self.clients)):
                task = asyncio.create_task(self.rate_limiters[i].acquire())
                wait_tasks.append((i, task))
            
            # Wait for any instance to become available
            try:
                # Wait for the first instance to become available
                done, pending = await asyncio.wait(
                    [task for _, task in wait_tasks],
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                # Cancel all other waiting tasks
                for _, task in wait_tasks:
                    if task not in done:
                        task.cancel()
                
                # Find which instance became available
                for i, task in wait_tasks:
                    if task in done:
                        logger.debug(f"Instance {i} ({self.ips[i]}:{self.port}) became available")
                        return self.clients[i], self.rate_limiters[i]
                
                # This should never happen, but just in case
                raise RuntimeError("No instance became available despite wait completion")
            except Exception as e:
                logger.error(f"Error while waiting for instances: {e}")
                raise

    def check_availability(self) -> Dict[str, Any]:
        """Check current availability status across all instances"""
        total_capacity = 0
        total_available = 0
        total_used = 0
        instance_details = []
        
        for i in range(len(self.clients)):
            available = self.rate_limiters[i].semaphore._value
            capacity = self.rate_limiters[i].max_window_size
            used = capacity - available
            
            total_capacity += capacity
            total_available += available
            total_used += used
            
            instance_details.append({
                'instance_id': i,
                'ip': self.ips[i],
                'port': self.port,
                'available_slots': available,
                'used_slots': used,
                'total_slots': capacity,
                'utilization_percent': (used / capacity) * 100 if capacity > 0 else 0
            })
        
        return {
            'total_instances': len(self.clients),
            'total_capacity': total_capacity,
            'total_available': total_available,
            'total_used': total_used,
            'overall_utilization_percent': (total_used / total_capacity) * 100 if total_capacity > 0 else 0,
            'has_available_slots': total_available > 0,
            'instances': instance_details
        }
    
    async def wait_for_availability(self, min_slots: int = 1, timeout_seconds: Optional[int] = None) -> bool:
        """Wait until at least min_slots are available across all instances
        
        Args:
            min_slots: Minimum number of slots to wait for
            timeout_seconds: Maximum time to wait (None for no timeout)
            
        Returns:
            True if slots became available, False if timeout
        """
        start_time = time.time()
        
        while True:
            availability = self.check_availability()
            if availability['total_available'] >= min_slots:
                logger.info(f"Availability achieved: {availability['total_available']} slots available (needed {min_slots})")
                return True
            
            # Check timeout
            if timeout_seconds is not None:
                elapsed = time.time() - start_time
                if elapsed >= timeout_seconds:
                    logger.warning(f"Timeout waiting for availability after {elapsed:.2f}s")
                    return False
            
            # Wait a short time before checking again
            await asyncio.sleep(0.1)

def prepare_message_for_vllm(content_messages):
    """
    The frame extraction logic for videos in `vLLM` differs from that of `qwen_vl_utils`.
    Here, we utilize `qwen_vl_utils` to extract video frames, with the `media_typ`e of the video explicitly set to `video/jpeg`.
    By doing so, vLLM will no longer attempt to extract frames from the input base64-encoded images.
    """
    vllm_messages, fps_list = [], []
    for message in content_messages:
        message_content_list = message["content"]
        if not isinstance(message_content_list, list):
            vllm_messages.append(message)
            continue

        new_content_list = []
        for part_message in message_content_list:
            if 'video' in part_message:
                video_message = [{'content': [part_message]}]
                image_inputs, video_inputs, video_kwargs = process_vision_info(video_message, return_video_kwargs=True)
                assert video_inputs is not None, "video_inputs should not be None"
                video_input = (video_inputs.pop()).permute(0, 2, 3, 1).numpy().astype(np.uint8)
                fps_list.extend(video_kwargs.get('fps', []))

                # encode image with base64
                base64_frames = []
                for frame in video_input:
                    img = Image.fromarray(frame)
                    output_buffer = BytesIO()
                    img.save(output_buffer, format="jpeg")
                    byte_data = output_buffer.getvalue()
                    base64_str = base64.b64encode(byte_data).decode("utf-8")
                    base64_frames.append(base64_str)

                part_message = {
                    "type": "video_url",
                    "video_url": {"url": f"data:video/jpeg;base64,{','.join(base64_frames)}"}
                }
            new_content_list.append(part_message)
        message["content"] = new_content_list
        vllm_messages.append(message)
    return vllm_messages, {'fps': fps_list}

class LLMClient:
    def __init__(self, 
                 model: str,
                 backend: str = None,
                 timeout_seconds: int = 300,
                 max_window_size_per_instance: int = 10,
                 port: int = None,
                 api_key: str = "token-abc123",
                 temperature: float = 1.0):
        """Initialize LLMClient with configurable parameters
        
        Args:
            model: Model name to find the correct server instances
            backend: Backend type ('vllm' or 'sglang'). If None, will auto-detect
            timeout_seconds: Request timeout in seconds
            max_window_size_per_instance: Maximum concurrent requests per server instance
            port: Port number for server instances (if None, will auto-detect from server info)
            api_key: API key for authentication
        """
        self.timeout_seconds = timeout_seconds
        self.model = model
        
        # Get server IPs and port (auto-detect if not specified)
        server_ips, detected_port = get_server_ips(model, backend)
        
        # Use detected port if not explicitly specified
        if port is None:
            port = detected_port
        
        # Create a rate limiter for each instance
        rate_limiters = [RateLimiter(max_window_size_per_instance) for _ in server_ips]
        self.client_manager = RoundRobinClient(server_ips, port, api_key, timeout_seconds, rate_limiters)
        
        backend_info = f" with backend '{backend}'" if backend else ""
        logger.info(f"Initialized LLMClient for model '{model}'{backend_info} with {len(server_ips)} instances on port {port}, {max_window_size_per_instance} slots per instance")

    async def single_call(self, inputs, model, **kwargs):
        """Single call with automatic routing optimization."""
        try:
            start_time = time.time()
            logger.debug("Starting request...")
            
            # Use consistent routing for better vLLM prefix reuse
            client, rate_limiter, instance_id = await self.client_manager.get_consistent_client_for_prompt(str(inputs))
            logger.debug(f"Routed to instance {instance_id} ({client.base_url})")
            
            try:
                logger.debug(f"Making request to {client.base_url}")
                if self.model == "Qwen/Qwen2.5-VL-72B-Instruct":
                    # print("**"*100)
                    video_messages, video_kwargs = prepare_message_for_vllm(inputs)
                    response = await client.chat.completions.create(
                        model=model, 
                        messages=video_messages,
                        timeout=self.timeout_seconds,
                        extra_body={
                            "mm_processor_kwargs": video_kwargs
                        },
                        **kwargs
                    )
                else:
                    response = await client.chat.completions.create(
                        model=model, 
                        messages=inputs,
                        timeout=self.timeout_seconds,
                        # temperature=self.temperature,
                        **kwargs
                    )
                duration = time.time() - start_time
                
                if duration > self.timeout_seconds * 0.9:
                    logger.warning(f"Request took {duration:.2f}s (close to timeout {self.timeout_seconds}s)")
                
                logger.debug(f"Request completed in {duration:.2f}s")
                return response.choices[0].message.content
            finally:
                await rate_limiter.release()
                logger.debug(f"Released slot on instance {instance_id}")
        except asyncio.TimeoutError:
            logger.error(f"Request timed out after {self.timeout_seconds}s")
            return None
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            return None

    async def process_all_inputs(self, inputs_list, num_generations=1, model=None, **kwargs):
        try:
            if num_generations > 1:
                logger.info(f"Using prefix reuse optimization for {num_generations} generations per input")
                # Group generations by prompt for better batching
                all_tasks = []
                for inputs in inputs_list:
                    # Create all generations for this prompt as a group
                    prompt_tasks = [
                        self.single_call(inputs, model=model, **kwargs) 
                        for _ in range(num_generations)
                    ]
                    all_tasks.append(asyncio.gather(*prompt_tasks))
                
                # Execute all prompt groups concurrently
                grouped_responses = await tqdm_asyncio.gather(*all_tasks, desc="Processing requests")
                return grouped_responses
            else:
                logger.info("Single generation per input")
                # Single generation per input - simple case
                all_tasks = [
                    self.single_call(inputs, model=model, **kwargs) 
                    for inputs in inputs_list
                ]
                responses = await tqdm_asyncio.gather(*all_tasks, desc="Processing requests")
                return [[response] for response in responses]  # Wrap in lists for consistency
        finally:
            pass

    def check_availability(self) -> Dict[str, Any]:
        """Check current availability status across all server instances
        
        Returns:
            Dictionary containing detailed availability information
        """
        return self.client_manager.check_availability()
    
    async def wait_for_availability(self, min_slots: int = 1, timeout_seconds: Optional[int] = None) -> bool:
        """Wait until at least min_slots are available across all instances
        
        Args:
            min_slots: Minimum number of slots to wait for
            timeout_seconds: Maximum time to wait (None for no timeout)
            
        Returns:
            True if slots became available, False if timeout
        """
        return await self.client_manager.wait_for_availability(min_slots, timeout_seconds)
    
    def is_available(self) -> bool:
        """Quick check if any slots are currently available
        
        Returns:
            True if at least one slot is available across all instances
        """
        availability = self.check_availability()
        return availability['has_available_slots']
    
    def get_capacity_info(self) -> Dict[str, int]:
        """Get basic capacity information
        
        Returns:
            Dictionary with total_capacity, total_available, total_used
        """
        availability = self.check_availability()
        return {
            'total_capacity': availability['total_capacity'],
            'total_available': availability['total_available'],
            'total_used': availability['total_used'],
            'utilization_percent': availability['overall_utilization_percent']
        }

class LLMProcessor:
    """
    A comprehensive processor that handles LLM inference with automatic batching,
    checkpointing, and progress tracking.
    """
    
    def __init__(self, 
                 model: str,
                 backend: str = None,
                 timeout_seconds: int = 24000,
                 max_window_size_per_instance: int = 256,
                 port: int = None,
                 api_key: str = "token-abc123",
                 batch_size: int = 50,
                 temperature: float = 1.0):
        """Initialize LLMProcessor with configurable parameters
        
        Args:
            model: Model name to find the correct server instances
            backend: Backend type ('vllm' or 'sglang'). If None, will auto-detect
            timeout_seconds: Request timeout in seconds
            max_window_size_per_instance: Maximum concurrent requests per server instance
            port: Port number for server instances (if None, will auto-detect from server info)
            api_key: API key for authentication
            batch_size: Number of inputs to process in each batch before saving
            temperature: Temperature for generation (default: 1.0)
        """
        self.batch_size = batch_size
        self.temperature = temperature
        
        # Initialize LLM client
        self.llm_client = LLMClient(
            model=model,
            backend=backend,
            timeout_seconds=timeout_seconds,
            max_window_size_per_instance=max_window_size_per_instance,
            port=port,
            api_key=api_key
        )
        
        backend_info = f" with backend '{backend}'" if backend else ""
        logger.info(f"Initialized LLMProcessor for model '{model}'{backend_info} with batch_size={batch_size}, temperature={temperature}")
    
    def generate_output_filename(self, input_file: str, model: str) -> str:
        """Generate output filename based on input file and model name, preserving file format."""
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        model_clean = model.replace('/', '_').replace('-', '_')
        input_dir = os.path.dirname(input_file)
        
        # Detect input file format and use the same for output
        try:
            input_format = detect_file_format(input_file)
            if input_format == 'parquet':
                extension = '.parquet'
            else:  # jsonl
                extension = '.jsonl'
        except ValueError:
            # Fallback to jsonl if format cannot be detected
            logger.warning(f"Could not detect format for {input_file}, defaulting to .jsonl")
            extension = '.jsonl'
        
        return os.path.join(input_dir, f"{base_name}_{model_clean}_responses{extension}")
    
    def load_existing_results(self, output_file: str, messages_column: str, num_generations: int, responses_column: str = 'responses') -> Tuple[pd.DataFrame, Set[str]]:
        """Load existing results and return processed messages set.
        
        Args:
            output_file: Path to output file
            messages_column: Column name containing messages
            num_generations: Expected number of generations per input
            responses_column: Column name for responses
            
        Returns:
            Tuple of (existing_df, processed_messages_set)
        """
        if not os.path.exists(output_file):
            logger.info(f"No existing output file found at: {output_file}")
            logger.info("Starting processing from the beginning")
            return pd.DataFrame(), set()
        
        try:
            logger.info(f"Found existing output file: {output_file}")
            logger.info("Reading existing results for checkpointing...")
            existing_df = read_data_file(output_file)
            logger.info(f"Successfully loaded existing output file with {len(existing_df)} entries")
            
            if responses_column in existing_df.columns:
                processed_messages = set()
                for idx, row in existing_df.iterrows():
                    responses = row.get(responses_column)
                    if responses is not None:
                        # Check if responses is sequence-like (has length and is iterable)
                        try:
                            responses_length = len(responses)
                            # Verify it's iterable
                            iter(responses)
                            is_sequence = True
                        except (TypeError, AttributeError):
                            is_sequence = False
                        
                        if is_sequence:
                            # Check if we have at least the required number of generations
                            if responses_length >= num_generations:
                                # Check if the first num_generations responses are all successful
                                first_n_responses = responses[:num_generations]
                                all_successful = all(response is not None for response in first_n_responses)
                                if all_successful:
                                    processed_messages.add(str(row[messages_column]))
                                    if responses_length > num_generations:
                                        logger.debug(f"Found message with {responses_length} generations (more than expected {num_generations}), keeping all existing responses")
                                else:
                                    logger.debug(f"Found incomplete responses for message (will retry): {len([r for r in first_n_responses if r is not None])}/{len(first_n_responses)} successful in first {num_generations} generations")
                            else:
                                logger.debug(f"Found message with {responses_length} generations, but {num_generations} expected (will retry)")
                        else:
                            # Handle single response case - only valid if num_generations is 1
                            if num_generations == 1 and responses is not None:
                                processed_messages.add(str(row[messages_column]))
                            else:
                                logger.debug(f"Found single response but {num_generations} generations expected (will retry)")
                
                logger.info(f"Found {len(processed_messages)} fully completed messages with at least {num_generations} successful generations each")
                logger.info(f"Will skip {len(processed_messages)} already processed inputs and continue with remaining work")
                return existing_df, processed_messages
            else:
                logger.info(f"No '{responses_column}' column found in existing file, starting fresh")
                return pd.DataFrame(), set()
        except Exception as e:
            logger.error(f"CRITICAL ERROR: Output file exists but cannot be read: {output_file}")
            logger.error(f"Error details: {e}")
            logger.error("This suggests the output file is corrupted. Please:")
            logger.error("1. Check if the file is corrupted and delete it if necessary")
            logger.error("2. Use force_restart=True to ignore the existing file")
            logger.error("3. Or fix the corruption issue before continuing")
            raise RuntimeError(f"Cannot read existing output file '{output_file}': {e}. "
                             f"File may be corrupted. Use force_restart=True to ignore existing file.") from e
    
    def save_batch_results(self, output_file: str, batch_results: List[Dict[str, Any]], all_existing_data: Optional[pd.DataFrame] = None, messages_column: str = 'messages'):
        """Save batch results to output file (replace mode to avoid duplicates)."""
        
        # Create DataFrame from batch results
        batch_df = pd.DataFrame(batch_results)
        
        if all_existing_data is not None and len(all_existing_data) > 0:
            # Get the messages being updated in this batch
            batch_messages = set(batch_df[messages_column].astype(str))
            
            # Remove existing records for messages being updated
            existing_filtered = all_existing_data[~all_existing_data[messages_column].astype(str).isin(batch_messages)]
            
            # Combine existing (non-updated) records with new batch results
            combined_df = pd.concat([existing_filtered, batch_df], ignore_index=True)
        else:
            combined_df = batch_df
        
        # Save the complete dataset using the appropriate format
        write_data_file(combined_df, output_file)
        
        logger.debug(f"Saved/updated {len(batch_results)} results in {output_file} (total records: {len(combined_df)})")
        
        return combined_df  # Return updated complete dataset
    
    def get_remaining_work(self, input_df: pd.DataFrame, processed_messages: Set[str], messages_column: str) -> pd.DataFrame:
        """Get the remaining work that needs to be processed."""
        if not processed_messages:
            return input_df.copy()
        
        remaining_df = input_df[~input_df[messages_column].astype(str).isin(processed_messages)].copy()
        logger.info(f"Remaining work: {len(remaining_df)} out of {len(input_df)} messages")
        return remaining_df
    
    async def process_batch(self, batch_df: pd.DataFrame, messages_column: str, 
                           num_generations: int, model: str, responses_column: str = 'responses', **kwargs) -> List[Dict[str, Any]]:
        """Process a batch of inputs and return results."""
        inputs_list = batch_df[messages_column].tolist()
        
        logger.debug(f"Processing batch of {len(inputs_list)} inputs with {num_generations} generations each")
        
        # Merge stored temperature with any additional kwargs
        generation_params = {'temperature': self.temperature}
        generation_params.update(kwargs)
        
        responses = await self.llm_client.process_all_inputs(
            inputs_list, 
            num_generations=num_generations,
            model=model,
            **generation_params
        )
        
        batch_results = []
        for idx, (_, row) in enumerate(batch_df.iterrows()):
            result = row.to_dict()
            result[responses_column] = responses[idx]
            batch_results.append(result)
        
        return batch_results
    
    async def process_dataset(self,
                            input_file: str,
                            model: str,
                            num_generations: int = 16,
                            output_file: Optional[str] = None,
                            messages_column: str = 'messages',
                            responses_column: str = 'responses',
                            force_restart: bool = False,
                            **kwargs) -> str:
        """
        Process a dataset of inputs through the LLM and save responses with checkpointing.
        Supports both JSONL and parquet file formats for input and output.
        
        Args:
            input_file: Path to input file (.jsonl or .parquet)
            model: Model name to use
            num_generations: Number of generations per input
            output_file: Output file path (auto-generated if None, preserves input format)
            messages_column: Column name containing the messages
            responses_column: Column name for storing the responses (default: responses)
            force_restart: Force restart from beginning, ignoring existing output file
            **kwargs: Additional parameters to pass to the OpenAI API (e.g., temperature, max_tokens, top_p)
            
        Returns:
            Path to the output file
            
        """
        # Generate output filename if not provided
        if output_file is None:
            output_file = self.generate_output_filename(input_file, model)
        
        logger.info(f"Processing dataset: {input_file}")
        logger.info(f"Output file: {output_file}")
        logger.info(f"Model: {model}")
        logger.info(f"Generations per input: {num_generations}")
        logger.info(f"Messages column: {messages_column}")
        logger.info(f"Responses column: {responses_column}")
        if kwargs:
            logger.info(f"Additional API parameters: {kwargs}")
        
        # Load existing results if not forcing restart
        if force_restart:
            logger.info("Force restart enabled - ignoring existing output file")
            existing_df = pd.DataFrame()
            processed_messages = set()
            if os.path.exists(output_file):
                os.remove(output_file)
                logger.info(f"Removed existing output file: {output_file}")
        else:
            existing_df, processed_messages = self.load_existing_results(output_file, messages_column, num_generations, responses_column)
        
        # Read input data
        logger.info(f"Reading input data from: {input_file}")
        input_df = read_data_file(input_file)
        
        # Check if messages column exists
        if messages_column not in input_df.columns:
            raise ValueError(f"Column '{messages_column}' not found in input data. Available columns: {list(input_df.columns)}")
        
        # Get remaining work
        remaining_df = self.get_remaining_work(input_df, processed_messages, messages_column)
        
        if len(remaining_df) == 0:
            logger.info("All messages have been processed! No remaining work.")
            return output_file
        
        # Check if any existing entries need to be retried (partial failures)
        if not force_restart and len(existing_df) > 0:
            partial_failures = len(input_df) - len(processed_messages) - len(remaining_df)
            if partial_failures > 0:
                logger.info(f"Found {partial_failures} messages with partial failures that will be retried")
        
        logger.info(f"Processing {len(remaining_df)} remaining inputs")
        logger.info(f"Batch size: {self.batch_size}")
        
        # Process in batches
        total_batches = (len(remaining_df) + self.batch_size - 1) // self.batch_size
        successful_generations = 0
        failed_generations = 0
        total_expected_generations = len(remaining_df) * num_generations
        
        # Load all existing data for replacement mode
        if os.path.exists(output_file) and not force_restart:
            try:
                all_existing_data = read_data_file(output_file)
                logger.debug(f"Loaded {len(all_existing_data)} existing records for update mode")
            except Exception as e:
                logger.warning(f"Could not load existing data for update mode: {e}")
                all_existing_data = pd.DataFrame()
        else:
            all_existing_data = pd.DataFrame()
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min((batch_idx + 1) * self.batch_size, len(remaining_df))
            
            batch_df = remaining_df.iloc[start_idx:end_idx].copy()
            
            logger.info(f"\n--- Processing batch {batch_idx + 1}/{total_batches} ({len(batch_df)} items) ---")
            
            try:
                # Process batch
                batch_results = await self.process_batch(
                    batch_df, messages_column, num_generations, model, responses_column, **kwargs
                )
                
                # Save batch results and get updated complete dataset
                all_existing_data = self.save_batch_results(output_file, batch_results, all_existing_data, messages_column)
                
                # Update success count
                batch_successful = sum(len([r for r in result[responses_column] if r is not None]) for result in batch_results)
                batch_failed = sum(len([r for r in result[responses_column] if r is None]) for result in batch_results)
                successful_generations += batch_successful
                failed_generations += batch_failed
                
                logger.info(f"Batch {batch_idx + 1} completed: {batch_successful} successful, {batch_failed} failed generations")
                logger.info(f"Overall progress: {successful_generations}/{total_expected_generations} successful ({successful_generations/total_expected_generations*100:.2f}%)")
                if failed_generations > 0:
                    logger.info(f"Total failures so far: {failed_generations} generations")
                
            except Exception as e:
                logger.error(f"Error processing batch {batch_idx + 1}: {e}")
                logger.error("Stopping processing. You can restart to continue from the next batch.")
                break
        
        logger.info(f"\nProcessing completed!")
        logger.info(f"Results saved to: {output_file}")
        logger.info(f"Total successful generations: {successful_generations}")
        logger.info(f"Success rate: {successful_generations/total_expected_generations*100:.2f}%")
        
        if failed_generations > 0:
            logger.info(f"Total failed generations: {failed_generations}")
            logger.info(f"Failed inputs will be retried when you run the processor again")
        
        return output_file 

    async def process_dataset_streaming(self,
                                      input_file: str,
                                      model: str,
                                      num_generations: int = 16,
                                      output_file: Optional[str] = None,
                                      messages_column: str = 'messages',
                                      responses_column: str = 'responses',
                                      force_restart: bool = False,
                                      save_interval: int = 50,
                                      **kwargs) -> str:
        """
        Process a dataset using streaming approach to eliminate long tail issues.
        Maintains constant GPU utilization by immediately starting new requests as others complete.
        Automatically uses prefix reuse optimization for better vLLM performance.
        
        Args:
            input_file: Path to input file (.jsonl or .parquet)
            model: Model name to use
            num_generations: Number of generations per input
            output_file: Output file path (auto-generated if None, preserves input format)
            messages_column: Column name containing the messages
            responses_column: Column name for storing the responses
            force_restart: Force restart from beginning, ignoring existing output file
            save_interval: Save results every N completed inputs (default: 50)
            **kwargs: Additional parameters to pass to the OpenAI API
            
        Returns:
            Path to the output file
        """
        # Generate output filename if not provided
        if output_file is None:
            output_file = self.generate_output_filename(input_file, model)
        
        logger.info(f"Processing dataset (STREAMING MODE): {input_file}")
        logger.info(f"Output file: {output_file}")
        logger.info(f"Model: {model}")
        logger.info(f"Generations per input: {num_generations}")
        logger.info(f"Save interval: {save_interval} completed inputs")
        if kwargs:
            logger.info(f"Additional API parameters: {kwargs}")
        
        # Load existing results if not forcing restart
        if force_restart:
            logger.info("Force restart enabled - ignoring existing output file")
            existing_df = pd.DataFrame()
            processed_messages = set()
            if os.path.exists(output_file):
                os.remove(output_file)
                logger.info(f"Removed existing output file: {output_file}")
        else:
            existing_df, processed_messages = self.load_existing_results(output_file, messages_column, num_generations, responses_column)
        
        # Read input data
        logger.info(f"Reading input data from: {input_file}")
        input_df = read_data_file(input_file)
        
        # Check if messages column exists
        if messages_column not in input_df.columns:
            raise ValueError(f"Column '{messages_column}' not found in input data. Available columns: {list(input_df.columns)}")
        
        # Get remaining work
        remaining_df = self.get_remaining_work(input_df, processed_messages, messages_column)
        
        if len(remaining_df) == 0:
            logger.info("All messages have been processed! No remaining work.")
            return output_file
        
        logger.info(f"Processing {len(remaining_df)} remaining inputs with streaming approach")
        
        # Prepare work queue
        work_queue = []
        for idx, (_, row) in enumerate(remaining_df.iterrows()):
            input_data = row[messages_column]
            for gen_idx in range(num_generations):
                work_item = {
                    'input_idx': idx,
                    'generation_idx': gen_idx,
                    'input_data': input_data,
                    'row_data': row.to_dict(),
                    'message_key': str(input_data)
                }
                work_queue.append(work_item)
        
        total_work_items = len(work_queue)
        logger.info(f"Created work queue with {total_work_items} items ({len(remaining_df)} inputs × {num_generations} generations)")
        
        # Streaming processing
        results_buffer = {}  # message_key -> {row_data: dict, responses: list}
        completed_items = 0
        successful_generations = 0
        failed_generations = 0
        completed_inputs_count = 0
        
        # Load existing data for incremental saving
        if os.path.exists(output_file) and not force_restart:
            try:
                all_existing_data = read_data_file(output_file)
                logger.debug(f"Loaded {len(all_existing_data)} existing records for incremental saves")
            except Exception as e:
                logger.warning(f"Could not load existing data: {e}")
                all_existing_data = pd.DataFrame()
        else:
            all_existing_data = pd.DataFrame()
        
        # Progress tracking
        progress_bar = tqdm_asyncio(total=total_work_items, desc="Processing requests")
        
        try:
            # Function to process a single work item
            async def process_work_item(work_item):
                nonlocal completed_items, successful_generations, failed_generations, completed_inputs_count
                
                try:
                    # Merge stored temperature with any additional kwargs
                    generation_params = {'temperature': self.temperature}
                    generation_params.update(kwargs)
                    
                    response = await self.llm_client.single_call(
                        work_item['input_data'], 
                        model=model, 
                        **generation_params
                    )
                    
                    # Store result
                    message_key = work_item['message_key']
                    if message_key not in results_buffer:
                        results_buffer[message_key] = {
                            'row_data': work_item['row_data'],
                            'responses': [None] * num_generations
                        }
                    
                    results_buffer[message_key]['responses'][work_item['generation_idx']] = response
                    
                    if response is not None:
                        successful_generations += 1
                    else:
                        failed_generations += 1
                        
                    completed_items += 1
                    progress_bar.update(1)
                    
                    # Update progress bar description
                    success_rate = (successful_generations / completed_items * 100) if completed_items > 0 else 0
                    progress_bar.set_description(f"Processing (Success: {success_rate:.1f}%)")
                    
                except Exception as e:
                    logger.error(f"Error processing work item: {e}")
                    failed_generations += 1
                    completed_items += 1
                    progress_bar.update(1)
            
            # Function to save completed inputs
            async def save_completed_inputs():
                nonlocal completed_inputs_count, all_existing_data
                
                # Find completed inputs (all generations done)
                completed_results = []
                completed_keys = []
                
                for message_key, result_data in results_buffer.items():
                    responses = result_data['responses']
                    if all(r is not None or r is None for r in responses):  # All slots filled (success or failure)
                        # Check if all generations for this input are done
                        expected_responses = num_generations
                        actual_responses = len([r for r in responses if r is not None or r is None])  # Count all non-empty slots
                        
                        if actual_responses == expected_responses:
                            result_dict = result_data['row_data'].copy()
                            result_dict[responses_column] = responses
                            completed_results.append(result_dict)
                            completed_keys.append(message_key)
                
                if completed_results:
                    logger.debug(f"Saving {len(completed_results)} completed inputs")
                    
                    # Remove completed items from buffer
                    for key in completed_keys:
                        del results_buffer[key]
                    
                    # Save to file
                    all_existing_data = self.save_batch_results(
                        output_file, completed_results, all_existing_data, messages_column
                    )
                    
                    completed_inputs_count += len(completed_results)
                    logger.info(f"Saved {len(completed_results)} completed inputs. "
                              f"Total completed: {completed_inputs_count}. "
                              f"Success rate: {successful_generations}/{completed_items} ({successful_generations/completed_items*100:.1f}%)")
            
            # Create tasks for all work items
            work_tasks = [process_work_item(item) for item in work_queue]
            
            # Process with periodic saving based on completed inputs
            last_saved_count = 0  # Track how many inputs were saved in last save operation
            for task in asyncio.as_completed(work_tasks):
                await task
                
                # Check if we have new completed inputs to save
                current_completed = sum(1 for result_data in results_buffer.values() 
                                      if all(r is not None or r is None for r in result_data['responses']) and 
                                         len([r for r in result_data['responses'] if r is not None or r is None]) == num_generations)
                
                # Save when we accumulate enough NEW completed inputs since last save
                new_completed_since_last_save = current_completed
                if new_completed_since_last_save >= save_interval:
                    await save_completed_inputs()
                    last_saved_count = current_completed  # Update after save
            
            # Final save for any remaining completed inputs
            await save_completed_inputs()
            
            # Save any partial results (inputs with some completed generations)
            if results_buffer:
                logger.info(f"Saving {len(results_buffer)} partial results")
                partial_results = []
                for message_key, result_data in results_buffer.items():
                    result_dict = result_data['row_data'].copy()
                    result_dict[responses_column] = result_data['responses']
                    partial_results.append(result_dict)
                
                all_existing_data = self.save_batch_results(
                    output_file, partial_results, all_existing_data, messages_column
                )
        
        finally:
            progress_bar.close()
        
        logger.info(f"\nStreaming processing completed!")
        logger.info(f"Results saved to: {output_file}")
        logger.info(f"Total successful generations: {successful_generations}")
        logger.info(f"Total failed generations: {failed_generations}")
        logger.info(f"Success rate: {successful_generations/(successful_generations + failed_generations)*100:.2f}%")
        
        return output_file 

    async def process_dataset_queue_streaming(self,
                                         input_file: str,
                                         model: str,
                                         num_generations: int = 16,
                                         output_file: Optional[str] = None,
                                         messages_column: str = 'messages',
                                         responses_column: str = 'responses',
                                         force_restart: bool = False,
                                         save_interval: int = 50,
                                         max_queue_size: Optional[int] = None,
                                         **kwargs) -> str:
        """
        Process a dataset using an advanced queue-based streaming approach for optimal GPU utilization.
        Uses asyncio.Queue to manage work items efficiently and minimize memory usage.
        
        This is the RECOMMENDED method for large datasets as it:
        - Eliminates long tail issues completely
        - Maintains constant GPU utilization
        - Uses minimal memory (queue-based vs loading all tasks)
        - Provides real-time progress and success rate tracking
        
        Args:
            input_file: Path to input file (.jsonl or .parquet)
            model: Model name to use
            num_generations: Number of generations per input
            output_file: Output file path (auto-generated if None, preserves input format)
            messages_column: Column name containing the messages
            responses_column: Column name for storing the responses
            force_restart: Force restart from beginning, ignoring existing output file
            save_interval: Save results every N completed inputs (default: 50)
            max_queue_size: Maximum queue size (None = unlimited, use total_capacity * 2 for optimal)
            **kwargs: Additional parameters to pass to the OpenAI API
            
        Returns:
            Path to the output file
        """
        # Generate output filename if not provided
        if output_file is None:
            output_file = self.generate_output_filename(input_file, model)
        
        logger.info(f"Processing dataset (QUEUE STREAMING MODE): {input_file}")
        logger.info(f"Output file: {output_file}")
        logger.info(f"Model: {model}")
        logger.info(f"Generations per input: {num_generations}")
        logger.info(f"Save interval: {save_interval} completed inputs")
        
        # Get optimal queue size based on server capacity
        capacity_info = self.llm_client.get_capacity_info()
        if max_queue_size is None:
            max_queue_size = capacity_info['total_capacity'] * 2  # 2x capacity for optimal throughput
        logger.info(f"Queue size: {max_queue_size} (Server capacity: {capacity_info['total_capacity']})")
        
        if kwargs:
            logger.info(f"Additional API parameters: {kwargs}")
        
        # Load existing results if not forcing restart
        if force_restart:
            logger.info("Force restart enabled - ignoring existing output file")
            existing_df = pd.DataFrame()
            processed_messages = set()
            if os.path.exists(output_file):
                os.remove(output_file)
                logger.info(f"Removed existing output file: {output_file}")
        else:
            existing_df, processed_messages = self.load_existing_results(output_file, messages_column, num_generations, responses_column)
        
        # Read input data
        logger.info(f"Reading input data from: {input_file}")
        input_df = read_data_file(input_file)
        
        # Check if messages column exists
        if messages_column not in input_df.columns:
            raise ValueError(f"Column '{messages_column}' not found in input data. Available columns: {list(input_df.columns)}")
        
        # Get remaining work
        remaining_df = self.get_remaining_work(input_df, processed_messages, messages_column)
        
        if len(remaining_df) == 0:
            logger.info("All messages have been processed! No remaining work.")
            return output_file
        
        total_inputs = len(remaining_df)
        total_generations = total_inputs * num_generations
        logger.info(f"Processing {total_inputs} inputs with {total_generations} total generations")
        
        # Initialize tracking variables
        work_queue = asyncio.Queue(maxsize=max_queue_size)
        results_buffer = {}  # message_key -> {row_data: dict, responses: list, completed_count: int}
        completed_generations = 0
        successful_generations = 0
        failed_generations = 0
        saved_inputs = 0
        
        # Load existing data for incremental saving
        if os.path.exists(output_file) and not force_restart:
            try:
                all_existing_data = read_data_file(output_file)
                logger.debug(f"Loaded {len(all_existing_data)} existing records for incremental saves")
            except Exception as e:
                logger.warning(f"Could not load existing data: {e}")
                all_existing_data = pd.DataFrame()
        else:
            all_existing_data = pd.DataFrame()
        
        # Progress tracking
        progress_bar = tqdm_asyncio(total=total_generations, desc="Processing")
        
        # Producer: Add work items to queue
        async def producer():
            for idx, (_, row) in enumerate(remaining_df.iterrows()):
                input_data = row[messages_column]
                message_key = str(input_data)
                
                # Initialize result buffer for this input
                if message_key not in results_buffer:
                    results_buffer[message_key] = {
                        'row_data': row.to_dict(),
                        'responses': [None] * num_generations,
                        'completed_count': 0
                    }
                
                # Add generation tasks to queue
                for gen_idx in range(num_generations):
                    work_item = {
                        'input_data': input_data,
                        'message_key': message_key,
                        'generation_idx': gen_idx
                    }
                    await work_queue.put(work_item)
            
            # Signal end of work
            for _ in range(capacity_info['total_capacity']):  # One sentinel per worker
                await work_queue.put(None)
        
        # Worker: Process items from queue
        async def worker(worker_id):
            nonlocal completed_generations, successful_generations, failed_generations
            
            while True:
                work_item = await work_queue.get()
                
                # Check for sentinel (end of work)
                if work_item is None:
                    work_queue.task_done()
                    break
                
                try:
                    # Merge stored temperature with any additional kwargs
                    generation_params = {'temperature': self.temperature}
                    generation_params.update(kwargs)
                    
                    # Process the request (automatically uses prefix reuse)
                    response = await self.llm_client.single_call(
                        work_item['input_data'], 
                        model=model, 
                        **generation_params
                    )
                    
                    # Store result
                    message_key = work_item['message_key']
                    gen_idx = work_item['generation_idx']
                    
                    results_buffer[message_key]['responses'][gen_idx] = response
                    results_buffer[message_key]['completed_count'] += 1
                    
                    # Update counters
                    completed_generations += 1
                    if response is not None:
                        successful_generations += 1
                    else:
                        failed_generations += 1
                    
                    # Update progress
                    success_rate = (successful_generations / completed_generations * 100) if completed_generations > 0 else 0
                    progress_bar.set_description(f"Success: {success_rate:.1f}% | Saved: {saved_inputs}/{total_inputs}")
                    progress_bar.update(1)
                    
                except Exception as e:
                    logger.error(f"Worker {worker_id} error: {e}")
                    failed_generations += 1
                    completed_generations += 1
                    progress_bar.update(1)
                finally:
                    work_queue.task_done()
        
        # Saver: Periodically save completed inputs
        async def saver():
            nonlocal saved_inputs, all_existing_data
            last_save_time = time.time()
            
            while completed_generations < total_generations:
                await asyncio.sleep(1)  # Check every second
                
                # Find completed inputs
                completed_results = []
                completed_keys = []
                
                for message_key, result_data in list(results_buffer.items()):
                    if result_data['completed_count'] == num_generations:
                        result_dict = result_data['row_data'].copy()
                        result_dict[responses_column] = result_data['responses']
                        completed_results.append(result_dict)
                        completed_keys.append(message_key)
                
                # Save if we have enough completed or it's been too long
                current_time = time.time()
                should_save = (
                    len(completed_results) >= save_interval or 
                    (completed_results and current_time - last_save_time > 3000)  # Safety save every 50 minutes (only if there are completed results)
                )
                
                if should_save and completed_results:
                    logger.debug(f"Saving {len(completed_results)} completed inputs")
                    
                    # Remove from buffer
                    for key in completed_keys:
                        del results_buffer[key]
                    
                    # Save to file
                    all_existing_data = self.save_batch_results(
                        output_file, completed_results, all_existing_data, messages_column
                    )
                    
                    saved_inputs += len(completed_results)
                    last_save_time = current_time
                    
                    logger.info(f"Saved {len(completed_results)} inputs. "
                              f"Total saved: {saved_inputs}/{total_inputs} "
                              f"({saved_inputs/total_inputs*100:.1f}%)")
        
        try:
            # Start all tasks
            producer_task = asyncio.create_task(producer())
            saver_task = asyncio.create_task(saver())
            
            # Start workers (one per server capacity for optimal utilization)
            worker_tasks = [
                asyncio.create_task(worker(i)) 
                for i in range(capacity_info['total_capacity'])
            ]
            
            # Wait for producer to finish
            await producer_task
            
            # Wait for all work to be done
            await work_queue.join()
            
            # Wait for workers to finish
            await asyncio.gather(*worker_tasks)
            
            # Cancel saver and do final save
            saver_task.cancel()
            try:
                await saver_task
            except asyncio.CancelledError:
                pass
            
            # Final save of any remaining results
            if results_buffer:
                logger.info(f"Final save of {len(results_buffer)} remaining inputs")
                final_results = []
                for message_key, result_data in results_buffer.items():
                    result_dict = result_data['row_data'].copy()
                    result_dict[responses_column] = result_data['responses']
                    final_results.append(result_dict)
                
                all_existing_data = self.save_batch_results(
                    output_file, final_results, all_existing_data, messages_column
                )
                saved_inputs += len(final_results)
        
        finally:
            progress_bar.close()
        
        logger.info(f"\nQueue streaming processing completed!")
        logger.info(f"Results saved to: {output_file}")
        logger.info(f"Total inputs processed: {saved_inputs}/{total_inputs}")
        logger.info(f"Total successful generations: {successful_generations}")
        logger.info(f"Total failed generations: {failed_generations}")
        logger.info(f"Success rate: {successful_generations/(successful_generations + failed_generations)*100:.2f}%")
        
        return output_file 