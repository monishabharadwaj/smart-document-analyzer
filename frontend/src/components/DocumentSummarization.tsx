import React, { useState } from 'react';
import { FileText, AlertCircle, ChevronDown, ChevronUp } from 'lucide-react';
import { SummaryResult } from '../types';
import apiService from '../services/api';
import LoadingSpinner from './LoadingSpinner';
import ErrorMessage from './ErrorMessage';

interface DocumentSummarizationProps {
  documentId: string | null;
  filename: string | null;
  content: string | null;
}

/**
 * Document summarization component
 * Generates and displays document summaries with word count statistics
 */
export const DocumentSummarization: React.FC<DocumentSummarizationProps> = ({ 
  documentId, 
  filename,
  content 
}) => {
  const [isSummarizing, setIsSummarizing] = useState(false);
  const [result, setResult] = useState<SummaryResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isExpanded, setIsExpanded] = useState(true);

  const handleSummarize = async () => {
    if (!content) {
      setError('Please upload a document first');
      return;
    }

    setError(null);
    setIsSummarizing(true);

    try {
      const response = await apiService.summarizeText(content, 0.3);
      
      if (response.success && response.data) {
        setResult(response.data);
        setIsExpanded(true);
      } else {
        setError(response.error || 'Summarization failed');
      }
    } catch (err) {
      setError('An unexpected error occurred during summarization');
    } finally {
      setIsSummarizing(false);
    }
  };

  const calculateCompressionRatio = (original: number, summary: number) => {
    return ((1 - summary / original) * 100).toFixed(1);
  };

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <h2 className="text-xl font-semibold text-gray-800 mb-4 flex items-center">
        <FileText className="w-5 h-5 mr-2" />
        Document Summarization
      </h2>

      {error && (
        <ErrorMessage 
          message={error} 
          onDismiss={() => setError(null)} 
        />
      )}

      {!content && (
        <div className="text-center py-8">
          <AlertCircle className="w-12 h-12 text-gray-400 mx-auto mb-4" />
          <p className="text-gray-500">Please upload a document first to generate a summary</p>
        </div>
      )}

      {content && !result && !isSummarizing && (
        <div className="text-center">
          <p className="text-gray-600 mb-4">
            Ready to summarize: <span className="font-medium">{filename}</span>
          </p>
          <button
            onClick={handleSummarize}
            className="bg-purple-600 text-white px-6 py-2 rounded-lg font-medium hover:bg-purple-700 transition-colors"
          >
            Generate Summary
          </button>
        </div>
      )}

      {isSummarizing && (
        <LoadingSpinner message="Analyzing document and generating summary..." />
      )}

      {result && (
        <div className="space-y-4">
          <div className="bg-gray-50 rounded-lg p-4">
            <div className="flex items-center justify-between mb-3">
              <h3 className="font-medium text-gray-800">Document Summary</h3>
              <button
                onClick={() => setIsExpanded(!isExpanded)}
                className="text-gray-500 hover:text-gray-700 transition-colors"
              >
                {isExpanded ? (
                  <ChevronUp className="w-5 h-5" />
                ) : (
                  <ChevronDown className="w-5 h-5" />
                )}
              </button>
            </div>

            {isExpanded && (
              <div className="prose prose-sm max-w-none">
                <p className="text-gray-700 leading-relaxed whitespace-pre-wrap">
                  {result.summary}
                </p>
              </div>
            )}

            <div className="mt-4 pt-3 border-t border-gray-200">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                <div className="text-center">
                  <div className="text-2xl font-bold text-blue-600">
                    {result.word_count.original.toLocaleString()}
                  </div>
                  <div className="text-gray-500">Original Words</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-green-600">
                    {result.word_count.summary.toLocaleString()}
                  </div>
                  <div className="text-gray-500">Summary Words</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-purple-600">
                    {calculateCompressionRatio(result.word_count.original, result.word_count.summary)}%
                  </div>
                  <div className="text-gray-500">Compression</div>
                </div>
              </div>
            </div>
          </div>

          <div className="text-xs text-gray-500">
            Processed on {new Date(result.processed_at).toLocaleString()}
          </div>

          <button
            onClick={() => setResult(null)}
            className="text-purple-600 hover:text-purple-700 font-medium text-sm"
          >
            Summarize Another Document
          </button>
        </div>
      )}
    </div>
  );
};

export default DocumentSummarization;