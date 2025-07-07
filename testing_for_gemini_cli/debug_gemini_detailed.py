#!/usr/bin/env python3

import subprocess
import sys
from pathlib import Path

def test_gemini_commands():
    print("=== Testing Gemini Commands Directly ===")
    
    test_commands = [
        ["bash", "-c", "source ~/.nvm/nvm.sh && gemini --help"],
        ["bash", "-c", "gemini --help"],
        ["gemini", "--help"],
        ["/home/lewbei/.nvm/versions/node/v24.3.0/bin/gemini", "--help"]
    ]
    
    for i, cmd in enumerate(test_commands, 1):
        print(f"\n--- Test {i}: {cmd} ---")
        try:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                timeout=10,
                text=True
            )
            print(f"Return code: {result.returncode}")
            print(f"Stdout: {result.stdout[:100]}...")
            print(f"Stderr: {result.stderr[:100]}...")
            
            if result.returncode == 0:
                print("✅ SUCCESS!")
                return cmd
            else:
                print("❌ FAILED")
                
        except Exception as e:
            print(f"❌ EXCEPTION: {e}")
    
    return None

if __name__ == "__main__":
    working_cmd = test_gemini_commands()
    print(f"\n=== RESULT ===")
    print(f"Working command: {working_cmd}")