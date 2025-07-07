#!/usr/bin/env python3

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / 'src'))
sys.path.append(str(Path(__file__).parent / 'src' / 'services'))

from services.model_interface import ModelConfig, ModelProvider
from services.cli_clients import GeminiCLIClient

def test_gemini_detection():
    print("=== Testing Gemini CLI Detection ===")
    
    config = ModelConfig(
        provider=ModelProvider.GEMINI_CLI,
        model_id="gemini-2.5-flash",
        display_name="Gemini CLI Test"
    )
    
    client = GeminiCLIClient(config)
    
    print(f"Gemini CLI available: {client.is_available()}")
    
    # Test the path detection method
    gemini_cmd = client._find_gemini_path()
    print(f"Detected command: {gemini_cmd}")
    
    return client.is_available()

if __name__ == "__main__":
    result = test_gemini_detection()
    print(f"Final result: {result}")