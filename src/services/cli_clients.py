"""
CLI Integration Clients for Claude Code and Gemini CLI

Direct integration with command-line AI tools without task-master dependency.
"""

import asyncio
import subprocess
import tempfile
import os
from typing import Dict, Any
from datetime import datetime
import logging

from .model_interface import (
    ModelClient, ModelConfig, ModelRequest, ModelResponse, 
    ModelProvider, ModelError
)


class ClaudeCodeClient(ModelClient):
    """Client for Claude Code CLI integration"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.logger = logging.getLogger(f"{__name__}.claude_code")
    
    async def generate(self, request: ModelRequest) -> ModelResponse:
        """Generate text using Claude Code CLI"""
        start_time = datetime.now()
        
        try:
            # Prepare the prompt
            full_prompt = self._prepare_prompt(request)
            
            # Create temporary file for the prompt
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(full_prompt)
                temp_file = f.name
            
            try:
                # Execute claude command
                result = await self._execute_claude_command(temp_file, request)
                
                # Calculate response time
                response_time = (datetime.now() - start_time).total_seconds()
                
                # Estimate usage
                estimated_input_tokens = len(full_prompt.split()) * 1.3
                estimated_output_tokens = len(result.split()) * 1.3
                
                usage = {
                    'prompt_tokens': int(estimated_input_tokens),
                    'completion_tokens': int(estimated_output_tokens),
                    'total_tokens': int(estimated_input_tokens + estimated_output_tokens)
                }
                
                return ModelResponse(
                    content=result.strip(),
                    model_id=self.config.model_id,
                    provider=self.config.provider,
                    usage=usage,
                    finish_reason="stop",
                    response_time=response_time,
                    estimated_cost=0.0,  # Claude Code is typically free
                    metadata={'claude_code_integration': True}
                )
                
            finally:
                # Clean up
                try:
                    os.unlink(temp_file)
                except:
                    pass
                    
        except Exception as e:
            self.logger.error(f"Claude Code generation failed: {str(e)}")
            raise ModelError(
                provider=self.config.provider,
                model_id=self.config.model_id,
                error_type="generation_error",
                message=str(e)
            )
    
    def _prepare_prompt(self, request: ModelRequest) -> str:
        """Prepare prompt for Claude Code"""
        parts = []
        
        if request.system_prompt:
            parts.append(f"System: {request.system_prompt}")
        
        parts.append(f"Human: {request.prompt}")
        parts.append("Assistant: ")
        
        return "\n\n".join(parts)
    
    async def _execute_claude_command(self, temp_file: str, request: ModelRequest) -> str:
        """Execute Claude Code command"""
        
        try:
            # Read the prompt from file and pass to claude
            with open(temp_file, 'r') as f:
                prompt_content = f.read()
            
            # Execute claude command with the prompt
            process = await asyncio.create_subprocess_exec(
                "claude",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate(input=prompt_content.encode())
            
            if process.returncode != 0:
                error_msg = f"Claude Code command failed: {stderr.decode()}"
                self.logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            return stdout.decode().strip()
            
        except Exception as e:
            self.logger.error(f"Failed to execute Claude Code command: {str(e)}")
            raise
    
    def is_available(self) -> bool:
        """Check if Claude Code CLI is available"""
        try:
            result = subprocess.run(
                ["claude", "--version"], 
                capture_output=True, 
                timeout=5
            )
            return result.returncode == 0
        except:
            return False
    
    def get_rate_limit_info(self) -> Dict[str, Any]:
        """Get rate limit information"""
        return {
            'requests_per_minute': 60,
            'requests_remaining': 60,
            'reset_time': None,
            'provider_limits': 'Depends on Claude Code configuration'
        }


class GeminiCLIClient(ModelClient):
    """Client for Gemini CLI integration"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.logger = logging.getLogger(f"{__name__}.gemini_cli")
    
    async def generate(self, request: ModelRequest) -> ModelResponse:
        """Generate text using Gemini CLI"""
        start_time = datetime.now()
        
        try:
            # Prepare the prompt
            full_prompt = self._prepare_prompt(request)
            
            # Execute gemini command
            result = await self._execute_gemini_command(full_prompt, request)
            
            # Calculate response time
            response_time = (datetime.now() - start_time).total_seconds()
            
            # Estimate usage
            estimated_input_tokens = len(full_prompt.split()) * 1.3
            estimated_output_tokens = len(result.split()) * 1.3
            
            usage = {
                'prompt_tokens': int(estimated_input_tokens),
                'completion_tokens': int(estimated_output_tokens),
                'total_tokens': int(estimated_input_tokens + estimated_output_tokens)
            }
            
            return ModelResponse(
                content=result.strip(),
                model_id=self.config.model_id,
                provider=self.config.provider,
                usage=usage,
                finish_reason="stop",
                response_time=response_time,
                estimated_cost=0.0,  # Gemini CLI is typically free
                metadata={'gemini_cli_integration': True}
            )
                
        except Exception as e:
            self.logger.error(f"Gemini CLI generation failed: {str(e)}")
            raise ModelError(
                provider=self.config.provider,
                model_id=self.config.model_id,
                error_type="generation_error",
                message=str(e)
            )
    
    def _prepare_prompt(self, request: ModelRequest) -> str:
        """Prepare prompt for Gemini CLI"""
        if request.system_prompt:
            return f"{request.system_prompt}\n\n{request.prompt}"
        return request.prompt
    
    async def _execute_gemini_command(self, prompt: str, request: ModelRequest) -> str:
        """Execute Gemini CLI command"""
        
        try:
            # Find gemini executable command
            gemini_cmd = self._find_gemini_path()
            if not gemini_cmd:
                raise RuntimeError("Gemini CLI executable not found")
            
            # Execute gemini command with prompt via stdin
            if gemini_cmd[0] == "bash":
                # For bash execution, construct the full command
                if "source" in gemini_cmd[2]:
                    cmd = ["bash", "-c", f"source ~/.nvm/nvm.sh && gemini -m {self.config.model_id}"]
                else:
                    cmd = ["bash", "-c", f"gemini -m {self.config.model_id}"]
            else:
                # For direct execution
                cmd = gemini_cmd + ["-m", self.config.model_id]
                
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            # Send prompt via stdin
            stdout, stderr = await process.communicate(input=prompt.encode())
            
            if process.returncode != 0:
                error_msg = f"Gemini CLI command failed: {stderr.decode()}"
                self.logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            return stdout.decode().strip()
                
        except Exception as e:
            self.logger.error(f"Failed to execute Gemini CLI command: {str(e)}")
            raise
    
    def _find_gemini_path(self) -> list:
        """Find the gemini executable path - returns command as list"""
        # Try different ways to run gemini
        possible_commands = [
            ["bash", "-c", "source ~/.nvm/nvm.sh && gemini"],  # Run via bash with nvm
            ["bash", "-c", "gemini"],  # Run via bash
            ["gemini"],  # Try PATH first
        ]
        
        for cmd in possible_commands:
            try:
                if cmd[0] == "bash":
                    # For bash command, test with --help
                    if "source" in cmd[2]:
                        test_cmd = ["bash", "-c", "source ~/.nvm/nvm.sh && gemini --help"]
                    else:
                        test_cmd = ["bash", "-c", "gemini --help"]
                else:
                    test_cmd = cmd + ["--help"]
                    
                result = subprocess.run(
                    test_cmd, 
                    capture_output=True, 
                    timeout=10  # Increased timeout
                )
                if result.returncode == 0:
                    return cmd
            except Exception as e:
                # Log the actual error for debugging
                if hasattr(self, 'logger'):
                    self.logger.debug(f"Command {cmd} failed: {e}")
                continue
            
        return None
    
    def is_available(self) -> bool:
        """Check if Gemini CLI is available"""
        return self._find_gemini_path() is not None
    
    def get_rate_limit_info(self) -> Dict[str, Any]:
        """Get rate limit information"""
        return {
            'requests_per_minute': 60,
            'requests_remaining': 60,
            'reset_time': None,
            'provider_limits': 'Depends on Gemini CLI configuration'
        }


def create_cli_client(config: ModelConfig) -> ModelClient:
    """Create appropriate CLI client based on provider"""
    
    if config.provider == ModelProvider.CLAUDE_CODE:
        return ClaudeCodeClient(config)
    elif config.provider == ModelProvider.GEMINI_CLI:
        return GeminiCLIClient(config)
    else:
        raise ValueError(f"No CLI client available for provider: {config.provider}")