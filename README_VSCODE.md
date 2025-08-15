# Smart Document Analyzer - VS Code Setup

## 🔧 Quick Start in VS Code

### 1. Open Project
```bash
cd C:\Users\Hp\smart-document-analyzer
code .
```

### 2. Install Extensions
Install these VS Code extensions:
- Python (Microsoft)
- Python Debugger (Microsoft) 
- Pylance (Microsoft)
- REST Client
- autoDocstring

### 3. Running the System

#### Method 1: Using Debug Panel (F5)
1. Press `F5` or click the Run icon
2. Select from these configurations:
   - **Start API Server** - Launch the REST API
   - **Run Integration Demo** - Test all components
   - **Run Final Test** - Comprehensive validation
   - **Test API Endpoints** - API connectivity test

#### Method 2: Using Tasks (Ctrl+Shift+P)
1. Press `Ctrl+Shift+P`
2. Type "Tasks: Run Task"
3. Choose from:
   - Start API Server
   - Run Integration Demo  
   - Run Final Test
   - Test API Endpoints
   - Install Dependencies
   - Format Code

#### Method 3: Using Integrated Terminal
1. Press `Ctrl+`` to open terminal
2. Run commands directly:
```bash
# Start API server
python api_server.py

# Run integration demo
python integration_demo.py

# Run final comprehensive test
python final_test.py

# Test API endpoints
python test_api.py
```

### 4. Testing API Endpoints in VS Code

#### Using REST Client Extension:
1. Open `api_tests.http` file
2. Start the API server first
3. Click "Send Request" above each HTTP request
4. View responses directly in VS Code

#### Using Browser:
1. Start API server: `python api_server.py`
2. Open: http://localhost:8000/docs
3. Interactive API documentation with test interface

### 5. File Structure Overview

```
smart-document-analyzer/
├── .vscode/              # VS Code configurations
│   ├── settings.json     # Workspace settings
│   ├── launch.json       # Debug configurations  
│   └── tasks.json        # Task definitions
├── src/                  # Core modules
│   └── semantic_search.py
├── api_server.py         # FastAPI REST server
├── integration_demo.py   # Integrated system demo
├── final_test.py         # Comprehensive validation
├── test_api.py           # API endpoint tests
├── api_tests.http        # REST Client test file
└── requirements.txt      # Python dependencies
```

### 6. Key Features in VS Code

#### IntelliSense & Code Completion
- Auto-complete for Python functions
- Type hints and parameter info
- Import suggestions

#### Debugging
- Set breakpoints by clicking line numbers
- Step through code with F10/F11
- Inspect variables and call stack
- Debug configurations for each component

#### Testing
- Run tests directly from editor
- View test results inline
- Coverage reports

#### Code Formatting
- Auto-format on save (Black formatter)
- PEP 8 compliance checking
- Import organization

### 7. Common Commands

| Command | Shortcut | Description |
|---------|----------|-------------|
| Run Python File | F5 | Execute current Python file |
| Open Terminal | Ctrl+` | Open integrated terminal |
| Command Palette | Ctrl+Shift+P | Access all VS Code commands |
| Quick Open | Ctrl+P | Quickly open files |
| Format Document | Shift+Alt+F | Format current file |
| Toggle Sidebar | Ctrl+B | Show/hide file explorer |

### 8. Debugging Tips

#### Debug the API Server:
1. Set breakpoints in `api_server.py`
2. Press F5 → "Start API Server"
3. Make API requests while debugging

#### Debug Integration:
1. Set breakpoints in `integration_demo.py`
2. Press F5 → "Run Integration Demo" 
3. Step through the analysis pipeline

### 9. API Testing Workflow

1. **Start Server**: F5 → "Start API Server"
2. **Test Endpoints**: Open `api_tests.http`
3. **Send Requests**: Click "Send Request" buttons
4. **View Results**: Responses appear inline
5. **Interactive Docs**: Visit http://localhost:8000/docs

### 10. Troubleshooting

#### Python Interpreter Issues:
1. Ctrl+Shift+P → "Python: Select Interpreter"
2. Choose Python 3.11 installation

#### Missing Dependencies:
1. Ctrl+Shift+P → "Tasks: Run Task" → "Install Dependencies"
2. Or run: `pip install -r requirements.txt`

#### Port Already in Use:
1. Change port in `api_server.py` (line with `port=8000`)
2. Update `@baseUrl` in `api_tests.http`

### 11. Performance Monitoring

#### View System Performance:
1. Run "Final Test" configuration
2. Monitor terminal output for metrics
3. Check processing speeds and accuracy

#### API Performance:
1. Use REST Client to measure response times
2. Monitor terminal logs during API server operation

## 🎯 Quick Actions

- **Start Everything**: F5 → "Start API Server"
- **Test System**: F5 → "Run Final Test"  
- **API Documentation**: http://localhost:8000/docs
- **Test APIs**: Open `api_tests.http` and click requests

Your Smart Document Analyzer is now fully integrated with VS Code! 🚀
