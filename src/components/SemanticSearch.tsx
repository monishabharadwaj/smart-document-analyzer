import React, { useState } from 'react';
import { Search, FileText, Star } from 'lucide-react';
import { SearchResult } from '../types';
import apiService from '../services/api';
import LoadingSpinner from './LoadingSpinner';
import ErrorMessage from './ErrorMessage';

/**
 * Semantic search component
 * Allows users to search across all uploaded documents using natural language queries
 */
export const SemanticSearch: React.FC = () => {
  const [query, setQuery] = useState('');
  const [isSearching, setIsSearching] = useState(false);
  const [results, setResults] = useState<SearchResult[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [hasSearched, setHasSearched] = useState(false);

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!query.trim()) {
      setError('Please enter a search query');
      return;
    }

    setError(null);
    setIsSearching(true);
    setHasSearched(true);

    try {
      const response = await apiService.searchDocuments(query.trim());
      
      if (response.success && response.data) {
        setResults(response.data);
      } else {
        setError(response.error || 'Search failed');
        setResults([]);
      }
    } catch (err) {
      setError('An unexpected error occurred during search');
      setResults([]);
    } finally {
      setIsSearching(false);
    }
  };

  const getRelevanceColor = (score: number) => {
    if (score >= 0.8) return 'text-green-600';
    if (score >= 0.6) return 'text-amber-600';
    return 'text-red-600';
  };

  const getRelevanceStars = (score: number) => {
    const stars = Math.round(score * 5);
    return Array.from({ length: 5 }, (_, i) => (
      <Star
        key={i}
        className={`w-3 h-3 ${i < stars ? 'text-yellow-400 fill-current' : 'text-gray-300'}`}
      />
    ));
  };

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <h2 className="text-xl font-semibold text-gray-800 mb-4 flex items-center">
        <Search className="w-5 h-5 mr-2" />
        Semantic Search
      </h2>

      {error && (
        <ErrorMessage 
          message={error} 
          onDismiss={() => setError(null)} 
        />
      )}

      <form onSubmit={handleSearch} className="mb-6">
        <div className="flex gap-2">
          <div className="flex-1">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search across all documents (e.g., 'contracts signed in 2023', 'financial reports')"
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              disabled={isSearching}
            />
          </div>
          <button
            type="submit"
            disabled={isSearching || !query.trim()}
            className="bg-blue-600 text-white px-6 py-2 rounded-lg font-medium hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {isSearching ? 'Searching...' : 'Search'}
          </button>
        </div>
      </form>

      {isSearching && (
        <LoadingSpinner message="Performing semantic search across documents..." />
      )}

      {hasSearched && !isSearching && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="font-medium text-gray-800">
              Search Results {results.length > 0 && `(${results.length})`}
            </h3>
            {query && (
              <span className="text-sm text-gray-500">
                for "{query}"
              </span>
            )}
          </div>

          {results.length === 0 ? (
            <div className="text-center py-8">
              <FileText className="w-12 h-12 text-gray-400 mx-auto mb-4" />
              <p className="text-gray-500">No documents found matching your search query</p>
              <p className="text-gray-400 text-sm mt-1">Try different keywords or phrases</p>
            </div>
          ) : (
            <div className="space-y-3">
              {results.map((result, index) => (
                <div key={`${result.document_id}-${index}`} className="border border-gray-200 rounded-lg p-4 hover:bg-gray-50 transition-colors">
                  <div className="flex items-start justify-between mb-2">
                    <h4 className="font-medium text-gray-800 flex items-center">
                      <FileText className="w-4 h-4 mr-2 text-gray-500" />
                      {result.filename}
                    </h4>
                    <div className="flex items-center space-x-2">
                      <div className="flex">
                        {getRelevanceStars(result.relevance_score)}
                      </div>
                      <span className={`text-sm font-medium ${getRelevanceColor(result.relevance_score)}`}>
                        {(result.relevance_score * 100).toFixed(0)}%
                      </span>
                    </div>
                  </div>
                  
                  <p className="text-gray-600 text-sm leading-relaxed mb-2">
                    {result.snippet}
                  </p>
                  
                  {result.metadata && (
                    <div className="flex items-center text-xs text-gray-500 space-x-4">
                      {result.metadata.page && (
                        <span>Page {result.metadata.page}</span>
                      )}
                      {result.metadata.section && (
                        <span>Section: {result.metadata.section}</span>
                      )}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default SemanticSearch;