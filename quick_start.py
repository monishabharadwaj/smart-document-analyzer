#!/usr/bin/env python3
"""
Quick Start Script for Smart Document Analyzer
==============================================

This script provides an interactive menu to run different components
without needing F5 or debug configurations.
"""

import os
import sys
import subprocess
from pathlib import Path

def print_banner():
    """Print the application banner."""
    print("=" * 60)
    print("🚀 Smart Document Analyzer - Quick Start Menu")
    print("=" * 60)

def print_menu():
    """Print the main menu options."""
    print("\nChoose an option:")
    print("1. 🔧 Start API Server")
    print("2. 🧪 Run Integration Demo")
    print("3. ✅ Run Final System Test")
    print("4. 🌐 Test API Endpoints")
    print("5. 📊 Install Dependencies")
    print("6. 📚 Open API Documentation")
    print("7. 🔍 Check System Status")
    print("8. ❌ Exit")
    print("-" * 40)

def run_command(command, description):
    """Run a command with proper error handling."""
    print(f"\n▶️ {description}")
    print(f"Running: {command}")
    print("-" * 40)
    
    try:
        result = subprocess.run(command, shell=True, cwd=Path(__file__).parent)
        if result.returncode == 0:
            print(f"✅ {description} completed successfully!")
        else:
            print(f"❌ {description} failed with exit code: {result.returncode}")
    except KeyboardInterrupt:
        print(f"\n⏹️ {description} was interrupted by user")
    except Exception as e:
        print(f"❌ Error running {description}: {e}")

def open_browser(url):
    """Open URL in default browser."""
    try:
        import webbrowser
        webbrowser.open(url)
        print(f"🌐 Opening {url} in your default browser...")
    except Exception as e:
        print(f"❌ Could not open browser: {e}")
        print(f"Please manually open: {url}")

def main():
    """Main interactive menu."""
    print_banner()
    
    while True:
        print_menu()
        
        try:
            choice = input("Enter your choice (1-8): ").strip()
            
            if choice == "1":
                print("🔧 Starting API Server...")
                print("📍 The server will run at: http://localhost:8000")
                print("📖 API docs will be at: http://localhost:8000/docs")
                print("⏹️ Press Ctrl+C to stop the server")
                run_command("python api_server.py", "API Server")
                
            elif choice == "2":
                run_command("python integration_demo.py", "Integration Demo")
                
            elif choice == "3":
                run_command("python final_test.py", "Final System Test")
                
            elif choice == "4":
                print("🌐 Testing API Endpoints...")
                print("⚠️ Make sure the API server is running first!")
                input("Press Enter when API server is ready, or Ctrl+C to cancel...")
                run_command("python test_api.py", "API Endpoint Tests")
                
            elif choice == "5":
                run_command("pip install -r requirements.txt", "Dependency Installation")
                
            elif choice == "6":
                print("📚 Opening API Documentation...")
                print("⚠️ Make sure the API server is running first!")
                open_browser("http://localhost:8000/docs")
                
            elif choice == "7":
                print("\n📊 System Status Check:")
                print("-" * 30)
                
                # Check Python version
                try:
                    python_version = subprocess.check_output([sys.executable, "--version"], text=True).strip()
                    print(f"✅ Python: {python_version}")
                except:
                    print("❌ Python: Not found or error")
                
                # Check if files exist
                important_files = [
                    "api_server.py",
                    "integration_demo.py", 
                    "final_test.py",
                    "src/semantic_search.py",
                    "requirements.txt"
                ]
                
                for file in important_files:
                    if Path(file).exists():
                        print(f"✅ File: {file}")
                    else:
                        print(f"❌ File: {file} (missing)")
                
                # Check some key dependencies
                dependencies = ["fastapi", "uvicorn", "transformers", "sentence-transformers", "faiss-cpu", "nltk"]
                for dep in dependencies:
                    try:
                        __import__(dep.replace("-", "_"))
                        print(f"✅ Package: {dep}")
                    except ImportError:
                        print(f"❌ Package: {dep} (not installed)")
                
            elif choice == "8":
                print("\n👋 Thank you for using Smart Document Analyzer!")
                print("🎯 Your AI-powered document processing system is ready!")
                break
                
            else:
                print("❌ Invalid choice. Please enter a number between 1-8.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except EOFError:
            print("\n\n👋 Goodbye!")
            break
            
        # Wait for user input before showing menu again
        if choice in ["1", "2", "3", "4", "5"]:
            input("\n⏸️ Press Enter to return to main menu...")

if __name__ == "__main__":
    main()
