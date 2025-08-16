# Smart Document Analyzer - Setup Guide

This guide will help you set up and run the complete Smart Document Analyzer application with both backend and frontend components.

## Prerequisites

- **Python 3.8+** (for backend)
- **Node.js 16+** (for frontend)
- **VS Code** (optional but recommended)

## Backend Setup

### 1. Navigate to Backend Directory
```bash
cd C:\Users\Hp\smart-document-analyzer
```

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 3. Start the Backend Server
```bash
python api_server.py
```

The backend will start on `http://localhost:8000`

**Important API Endpoints:**
- `GET /` - Health check
- `GET /docs` - Interactive API documentation
- `POST /upload/document` - Upload text files
- `POST /classify/document` - Classify document content
- `POST /extract/entities` - Extract named entities
- `POST /summarize/text` - Generate summaries
- `POST /search/documents` - Semantic search

## Frontend Setup

### 1. Navigate to Frontend Directory
```bash
cd "C:\Users\Hp\OneDrive\Desktop\frontend dev\project"
```

### 2. Install Dependencies
```bash
npm install
```

### 3. Start the Frontend Development Server
```bash
npm run dev
```

The frontend will start on `http://localhost:5173`

## Testing the Application

### 1. Upload Test Documents
- Use the provided sample files in `sample-documents/` folder:
  - `sample-invoice.txt` - For testing invoice classification
  - `sample-contract.txt` - For testing contract classification

### 2. Test Each Feature

#### Document Upload
1. Go to the Upload tab
2. Drag and drop or select a `.txt` file
3. Verify successful upload

#### Document Classification
1. After uploading, the app will auto-switch to Classify tab
2. Click "Classify Document"
3. Should identify document type (invoice, contract, report, research paper)

#### Entity Extraction
1. Switch to Entities tab
2. Click "Extract Entities" 
3. Should identify people, organizations, dates, money amounts, etc.

#### Text Summarization
1. Switch to Summarize tab
2. Click "Generate Summary"
3. Should provide a condensed version with statistics

#### Semantic Search
1. Switch to Search tab
2. Upload multiple documents first using other tabs
3. Enter natural language queries like:
   - "contracts with payment terms"
   - "invoices from 2024"
   - "documents mentioning Chicago"

## Current Limitations

- **File Types**: Currently only supports `.txt` files (PDF/DOCX support planned)
- **File Size**: Maximum 10MB per file
- **Search**: Requires multiple documents to be uploaded first

## Troubleshooting

### Backend Issues
1. **Port 8000 already in use**: Kill existing processes or change port in `api_server.py`
2. **Dependencies missing**: Run `pip install -r requirements.txt`
3. **CORS errors**: The backend is configured to allow all origins for development

### Frontend Issues
1. **Connection refused**: Ensure backend is running on `http://localhost:8000`
2. **Upload fails**: Check file type (must be `.txt`) and size (under 10MB)
3. **API errors**: Check browser console and backend logs

### Network Issues
1. Ensure Windows Firewall allows Python and Node.js applications
2. Try accessing backend directly at `http://localhost:8000/docs`

## Architecture Overview

```
Frontend (React/TypeScript)     Backend (FastAPI/Python)
├── Upload Component       →    ├── File Upload Endpoint
├── Classification         →    ├── Document Classifier  
├── Entity Extraction      →    ├── NER Service
├── Summarization         →    ├── Text Summarizer
└── Search                →    └── Semantic Search Engine
```

## API Response Format

All backend endpoints return a standardized response:
```json
{
  "success": true,
  "message": "Operation completed",
  "data": { ... },
  "timestamp": "2024-01-15T10:30:00",
  "processing_time_ms": 1250.5
}
```

## Development Notes

- Backend uses CORS middleware to allow frontend access
- Frontend has proper error handling and loading states
- All components are responsive for desktop and mobile
- Real-time feedback for all operations

## Next Steps

1. **Add PDF/DOCX Support**: Implement document parsing for additional file types
2. **Batch Processing**: Allow multiple file uploads at once  
3. **User Authentication**: Add user accounts and document management
4. **Cloud Deployment**: Deploy to cloud platforms for production use
5. **Advanced Analytics**: Add document insights and trends

For issues or questions, check the backend logs and browser console for detailed error messages.
