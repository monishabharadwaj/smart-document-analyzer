import React, { useState } from 'react';
import { FileText, AlertCircle } from 'lucide-react';
import { ClassificationResult } from '../types';
import apiService from '../services/api';
import LoadingSpinner from './LoadingSpinner';
import ErrorMessage from './ErrorMessage';

interface DocumentClassificationProps {
  documentId: string | null;
  filename: string | null;
  content: string | null;
}

/**
 * Document classification component
 * Classifies documents as invoice, report, contract, or research paper
 */
export const DocumentClassification: React.FC<DocumentClassificationProps> = ({ 
  documentId, 
  filename,
  content 
}) => {
  const [isClassifying, setIsClassifying] = useState(false);
  const [result, setResult] = useState<ClassificationResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const getClassificationColor = (type: string) => {
    const colors = {
      invoice: 'bg-green-100 text-green-800 border-green-200',
      report: 'bg-blue-100 text-blue-800 border-blue-200',
      contract: 'bg-purple-100 text-purple-800 border-purple-200',
      'research paper': 'bg-amber-100 text-amber-800 border-amber-200',
      default: 'bg-gray-100 text-gray-800 border-gray-200',
    };
    return colors[type as keyof typeof colors] || colors.default;
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'text-green-600';
    if (confidence >= 0.6) return 'text-amber-600';
    return 'text-red-600';
  };

  const handleClassify = async () => {
    if (!content) {
      setError('Please upload a document first');
      return;
    }

    setError(null);
    setIsClassifying(true);

    try {
      const response = await apiService.classifyText(content);
      
      if (response.success && response.data) {
        setResult(response.data);
      } else {
        setError(response.error || 'Classification failed');
      }
    } catch (err) {
      setError('An unexpected error occurred during classification');
    } finally {
      setIsClassifying(false);
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <h2 className="text-xl font-semibold text-gray-800 mb-4 flex items-center">
        <FileText className="w-5 h-5 mr-2" />
        Document Classification
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
          <p className="text-gray-500">Please upload a document first to enable classification</p>
        </div>
      )}

      {content && !result && !isClassifying && (
        <div className="text-center">
          <p className="text-gray-600 mb-4">
            Ready to classify: <span className="font-medium">{filename}</span>
          </p>
          <button
            onClick={handleClassify}
            className="bg-blue-600 text-white px-6 py-2 rounded-lg font-medium hover:bg-blue-700 transition-colors"
          >
            Classify Document
          </button>
        </div>
      )}

      {isClassifying && (
        <LoadingSpinner message="Analyzing document structure and content..." />
      )}

      {result && (
        <div className="space-y-4">
          <div className="bg-gray-50 rounded-lg p-4">
            <h3 className="font-medium text-gray-800 mb-2">Classification Result</h3>
            <div className="flex items-center justify-between mb-2">
              <span className={`px-3 py-1 rounded-full text-sm font-medium border ${getClassificationColor(result.classification.type)}`}>
                {result.classification.type.charAt(0).toUpperCase() + result.classification.type.slice(1)}
              </span>
              <span className={`font-medium ${getConfidenceColor(result.classification.confidence)}`}>
                {(result.classification.confidence * 100).toFixed(1)}% confident
              </span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2 mb-2">
              <div
                className={`h-2 rounded-full transition-all duration-500 ${
                  result.classification.confidence >= 0.8 ? 'bg-green-500' :
                  result.classification.confidence >= 0.6 ? 'bg-amber-500' : 'bg-red-500'
                }`}
                style={{ width: `${result.classification.confidence * 100}%` }}
              />
            </div>
          </div>

          <div className="text-xs text-gray-500">
            Processed on {new Date(result.processed_at).toLocaleString()}
          </div>

          <button
            onClick={() => setResult(null)}
            className="text-blue-600 hover:text-blue-700 font-medium text-sm"
          >
            Classify Another Document
          </button>
        </div>
      )}
    </div>
  );
};

export default DocumentClassification;