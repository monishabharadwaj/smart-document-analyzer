import React, { useState } from 'react';
import { Users, AlertCircle } from 'lucide-react';
import { EntityResult } from '../types';
import apiService from '../services/api';
import LoadingSpinner from './LoadingSpinner';
import ErrorMessage from './ErrorMessage';

interface EntityExtractionProps {
  documentId: string | null;
  filename: string | null;
  content: string | null;
}

/**
 * Named Entity Recognition (NER) component
 * Extracts and displays entities like Person, Organization, Date, Money, etc.
 */
export const EntityExtraction: React.FC<EntityExtractionProps> = ({ 
  documentId, 
  filename,
  content 
}) => {
  const [isExtracting, setIsExtracting] = useState(false);
  const [result, setResult] = useState<EntityResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const getEntityColor = (label: string) => {
    const colors = {
      PERSON: 'bg-blue-100 text-blue-800 border-blue-200',
      ORG: 'bg-green-100 text-green-800 border-green-200',
      ORGANIZATION: 'bg-green-100 text-green-800 border-green-200',
      DATE: 'bg-purple-100 text-purple-800 border-purple-200',
      MONEY: 'bg-yellow-100 text-yellow-800 border-yellow-200',
      GPE: 'bg-pink-100 text-pink-800 border-pink-200',
      LOC: 'bg-pink-100 text-pink-800 border-pink-200',
      LOCATION: 'bg-pink-100 text-pink-800 border-pink-200',
      MISC: 'bg-gray-100 text-gray-800 border-gray-200',
      default: 'bg-gray-100 text-gray-800 border-gray-200',
    };
    return colors[label.toUpperCase() as keyof typeof colors] || colors.default;
  };

  const groupEntitiesByLabel = (entities: EntityResult['entities']) => {
    return entities.reduce((groups, entity) => {
      const label = entity.label.toUpperCase();
      if (!groups[label]) {
        groups[label] = [];
      }
      groups[label].push(entity);
      return groups;
    }, {} as Record<string, EntityResult['entities']>);
  };

  const handleExtractEntities = async () => {
    if (!content) {
      setError('Please upload a document first');
      return;
    }

    setError(null);
    setIsExtracting(true);

    try {
      const response = await apiService.extractEntitiesFromText(content);
      
      if (response.success && response.data) {
        setResult(response.data);
      } else {
        setError(response.error || 'Entity extraction failed');
      }
    } catch (err) {
      setError('An unexpected error occurred during entity extraction');
    } finally {
      setIsExtracting(false);
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <h2 className="text-xl font-semibold text-gray-800 mb-4 flex items-center">
        <Users className="w-5 h-5 mr-2" />
        Named Entity Recognition
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
          <p className="text-gray-500">Please upload a document first to extract entities</p>
        </div>
      )}

      {content && !result && !isExtracting && (
        <div className="text-center">
          <p className="text-gray-600 mb-4">
            Ready to extract entities from: <span className="font-medium">{filename}</span>
          </p>
          <button
            onClick={handleExtractEntities}
            className="bg-green-600 text-white px-6 py-2 rounded-lg font-medium hover:bg-green-700 transition-colors"
          >
            Extract Entities
          </button>
        </div>
      )}

      {isExtracting && (
        <LoadingSpinner message="Identifying and extracting named entities..." />
      )}

      {result && (
        <div className="space-y-4">
          <div className="bg-gray-50 rounded-lg p-4">
            <h3 className="font-medium text-gray-800 mb-4">
              Found {result.entities.length} entities
            </h3>
            
            {result.entities.length === 0 ? (
              <p className="text-gray-500 text-center py-4">No entities found in this document</p>
            ) : (
              <div className="space-y-4">
                {Object.entries(groupEntitiesByLabel(result.entities)).map(([label, entities]) => (
                  <div key={label} className="border border-gray-200 rounded-lg p-3">
                    <h4 className="font-medium text-gray-700 mb-2 flex items-center">
                      <span className={`px-2 py-1 rounded text-xs font-medium border mr-2 ${getEntityColor(label)}`}>
                        {label}
                      </span>
                      <span className="text-sm text-gray-500">({entities.length})</span>
                    </h4>
                    <div className="flex flex-wrap gap-2">
                      {entities.map((entity, index) => (
                        <span
                          key={`${entity.text}-${index}`}
                          className="bg-white border border-gray-200 rounded px-2 py-1 text-sm text-gray-700 hover:bg-gray-50 transition-colors"
                          title={entity.confidence ? `Confidence: ${(entity.confidence * 100).toFixed(1)}%` : undefined}
                        >
                          {entity.text}
                        </span>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          <div className="text-xs text-gray-500">
            Processed on {new Date(result.processed_at).toLocaleString()}
          </div>

          <button
            onClick={() => setResult(null)}
            className="text-green-600 hover:text-green-700 font-medium text-sm"
          >
            Extract from Another Document
          </button>
        </div>
      )}
    </div>
  );
};

export default EntityExtraction;