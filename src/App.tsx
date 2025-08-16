import React, { useState } from 'react';
import { FileText, Upload, Search, BarChart3, Users, FileSearch } from 'lucide-react';
import FileUpload from './components/FileUpload';
import DocumentClassification from './components/DocumentClassification';
import EntityExtraction from './components/EntityExtraction';
import DocumentSummarization from './components/DocumentSummarization';
import SemanticSearch from './components/SemanticSearch';

type TabType = 'upload' | 'classify' | 'entities' | 'summarize' | 'search';

/**
 * Main application component for the Smart Document Analyzer
 * Provides a tabbed interface for all document processing features
 */
function App() {
  const [activeTab, setActiveTab] = useState<TabType>('upload');
  const [currentDocument, setCurrentDocument] = useState<{
    id: string;
    filename: string;
    content: string;
  } | null>(null);

  const handleUploadSuccess = (documentId: string, filename: string, content: string) => {
    setCurrentDocument({ id: documentId, filename, content });
    // Automatically switch to classification tab after successful upload
    setActiveTab('classify');
  };

  const tabs = [
    {
      id: 'upload' as TabType,
      label: 'Upload',
      icon: Upload,
      description: 'Upload PDF or DOCX documents',
    },
    {
      id: 'classify' as TabType,
      label: 'Classify',
      icon: BarChart3,
      description: 'Classify document types',
    },
    {
      id: 'entities' as TabType,
      label: 'Entities',
      icon: Users,
      description: 'Extract named entities',
    },
    {
      id: 'summarize' as TabType,
      label: 'Summarize',
      icon: FileText,
      description: 'Generate summaries',
    },
    {
      id: 'search' as TabType,
      label: 'Search',
      icon: Search,
      description: 'Semantic document search',
    },
  ];

  const renderTabContent = () => {
    switch (activeTab) {
      case 'upload':
        return <FileUpload onUploadSuccess={handleUploadSuccess} />;
      case 'classify':
        return (
          <DocumentClassification 
            documentId={currentDocument?.id || null} 
            filename={currentDocument?.filename || null}
            content={currentDocument?.content || null}
          />
        );
      case 'entities':
        return (
          <EntityExtraction 
            documentId={currentDocument?.id || null} 
            filename={currentDocument?.filename || null}
            content={currentDocument?.content || null}
          />
        );
      case 'summarize':
        return (
          <DocumentSummarization 
            documentId={currentDocument?.id || null} 
            filename={currentDocument?.filename || null}
            content={currentDocument?.content || null}
          />
        );
      case 'search':
        return <SemanticSearch />;
      default:
        return <FileUpload onUploadSuccess={handleUploadSuccess} />;
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center space-x-3">
            <FileSearch className="w-8 h-8 text-blue-600" />
            <div>
              <h1 className="text-2xl font-bold text-gray-900">Smart Document Analyzer</h1>
              <p className="text-gray-600 text-sm">
                Upload, analyze, and search your documents with AI-powered insights
              </p>
            </div>
          </div>
        </div>
      </header>

      {/* Current Document Status */}
      {currentDocument && (
        <div className="bg-blue-50 border-b border-blue-200">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-2">
            <div className="flex items-center text-sm text-blue-800">
              <FileText className="w-4 h-4 mr-2" />
              <span>Current document: <strong>{currentDocument.filename}</strong></span>
            </div>
          </div>
        </div>
      )}

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {/* Navigation Tabs */}
        <div className="mb-6">
          <nav className="flex space-x-1 bg-white rounded-lg shadow-sm p-1">
            {tabs.map((tab) => {
              const IconComponent = tab.icon;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex-1 flex items-center justify-center px-3 py-2 text-sm font-medium rounded-md transition-colors ${
                    activeTab === tab.id
                      ? 'bg-blue-600 text-white shadow-sm'
                      : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
                  }`}
                  title={tab.description}
                >
                  <IconComponent className="w-4 h-4 mr-2" />
                  <span className="hidden sm:inline">{tab.label}</span>
                </button>
              );
            })}
          </nav>
        </div>

        {/* Tab Content */}
        <div className="transition-all duration-200">
          {renderTabContent()}
        </div>
      </div>

      {/* Footer */}
      <footer className="bg-white border-t border-gray-200 mt-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="text-center text-gray-500 text-sm">
            <p>Smart Document Analyzer - Powered by AI for intelligent document processing</p>
            <p className="mt-1">
              Backend API: <code className="bg-gray-100 px-1 rounded text-xs">
                https://github.com/monishabharadwaj/smart-document-analyzer
              </code>
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;