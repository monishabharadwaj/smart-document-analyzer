// API service functions for interacting with the Smart Document Analyzer backend

import { ClassificationResult, EntityResult, SummaryResult, SearchResult, ApiResponse } from '../types';

const API_BASE_URL = 'http://localhost:8000'; // Local backend URL

// Backend response format
interface BackendResponse {
  success: boolean;
  message: string;
  data?: any;
  timestamp: string;
  processing_time_ms?: number;
}

class ApiService {
  private baseURL: string;

  constructor(baseURL: string = API_BASE_URL) {
    this.baseURL = baseURL;
  }

  /**
   * Check if backend is healthy
   */
  async healthCheck(): Promise<boolean> {
    try {
      const response = await fetch(`${this.baseURL}/health`);
      return response.ok;
    } catch {
      return false;
    }
  }

  /**
   * Upload a document to the backend
   * @param file - The file to upload (supports text files for now)
   * @returns Promise with upload result
   */
  async uploadDocument(file: File): Promise<ApiResponse<{ document_id: string; filename: string }>> {
    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch(`${this.baseURL}/upload/document`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ message: response.statusText }));
        throw new Error(errorData.detail || errorData.message || `Upload failed: ${response.statusText}`);
      }

      const backendResponse: BackendResponse = await response.json();
      
      if (backendResponse.success && backendResponse.data) {
        // Extract document ID from backend response
        const documentId = backendResponse.data.doc_id || backendResponse.data.document_id;
        const filename = backendResponse.data.metadata?.filename || file.name;
        
        return { 
          success: true, 
          data: { 
            document_id: documentId,
            filename: filename
          }
        };
      } else {
        throw new Error(backendResponse.message || 'Upload failed');
      }
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Upload failed',
      };
    }
  }

  /**
   * Classify a document using text content
   * @param text - The text content to classify
   * @returns Promise with classification result
   */
  async classifyText(text: string): Promise<ApiResponse<ClassificationResult>> {
    try {
      const response = await fetch(`${this.baseURL}/classify/document`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ message: response.statusText }));
        throw new Error(errorData.detail || errorData.message || 'Classification failed');
      }

      const backendResponse: BackendResponse = await response.json();
      
      if (backendResponse.success && backendResponse.data) {
        return { 
          success: true, 
          data: {
            document_id: 'temp_id',
            classification: {
              type: backendResponse.data.predicted_class,
              confidence: backendResponse.data.confidence
            },
            processed_at: backendResponse.timestamp
          }
        };
      } else {
        throw new Error(backendResponse.message || 'Classification failed');
      }
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Classification failed',
      };
    }
  }

  /**
   * Extract entities from text
   * @param text - The text content to process
   * @returns Promise with entity extraction result
   */
  async extractEntitiesFromText(text: string): Promise<ApiResponse<EntityResult>> {
    try {
      const response = await fetch(`${this.baseURL}/extract/entities`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ message: response.statusText }));
        throw new Error(errorData.detail || errorData.message || 'Entity extraction failed');
      }

      const backendResponse: BackendResponse = await response.json();
      
      if (backendResponse.success && backendResponse.data) {
        return { 
          success: true, 
          data: {
            document_id: 'temp_id',
            entities: backendResponse.data.entities.map((entity: any) => ({
              text: entity.text,
              label: entity.label,
              start: entity.start || 0,
              end: entity.end || 0,
              confidence: entity.confidence
            })),
            processed_at: backendResponse.timestamp
          }
        };
      } else {
        throw new Error(backendResponse.message || 'Entity extraction failed');
      }
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Entity extraction failed',
      };
    }
  }

  /**
   * Generate a summary of text
   * @param text - The text to summarize
   * @param ratio - Summary ratio (0.1-0.9)
   * @returns Promise with summary result
   */
  async summarizeText(text: string, ratio: number = 0.3): Promise<ApiResponse<SummaryResult>> {
    try {
      const response = await fetch(`${this.baseURL}/summarize/text`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
          text, 
          ratio,
          summary_type: 'extractive'
        }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ message: response.statusText }));
        throw new Error(errorData.detail || errorData.message || 'Summarization failed');
      }

      const backendResponse: BackendResponse = await response.json();
      
      if (backendResponse.success && backendResponse.data) {
        return { 
          success: true, 
          data: {
            document_id: 'temp_id',
            summary: backendResponse.data.summary,
            word_count: {
              original: backendResponse.data.statistics?.original_words || 0,
              summary: backendResponse.data.statistics?.summary_words || 0
            },
            processed_at: backendResponse.timestamp
          }
        };
      } else {
        throw new Error(backendResponse.message || 'Summarization failed');
      }
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Summarization failed',
      };
    }
  }

  /**
   * Perform semantic search across documents
   * @param query - The search query
   * @param top_k - Maximum number of results (optional)
   * @returns Promise with search results
   */
  async searchDocuments(query: string, top_k: number = 5): Promise<ApiResponse<SearchResult[]>> {
    try {
      const response = await fetch(`${this.baseURL}/search/documents`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query, top_k }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ message: response.statusText }));
        throw new Error(errorData.detail || errorData.message || 'Search failed');
      }

      const backendResponse: BackendResponse = await response.json();
      
      if (backendResponse.success && backendResponse.data) {
        const results = backendResponse.data.results || [];
        const searchResults = results.map((result: any) => ({
          document_id: result.doc_id || result.document_id,
          filename: result.metadata?.filename || 'Unknown',
          relevance_score: result.similarity || result.score || 0,
          snippet: result.text || result.content || '',
          metadata: result.metadata
        }));
        
        return { 
          success: true, 
          data: searchResults
        };
      } else {
        throw new Error(backendResponse.message || 'Search failed');
      }
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Search failed',
      };
    }
  }

  /**
   * Get collection insights and analytics
   */
  async getCollectionInsights(): Promise<ApiResponse<any>> {
    try {
      const response = await fetch(`${this.baseURL}/collection/insights`);
      
      if (!response.ok) {
        throw new Error('Failed to fetch insights');
      }

      const backendResponse: BackendResponse = await response.json();
      
      if (backendResponse.success) {
        return { success: true, data: backendResponse.data };
      } else {
        throw new Error(backendResponse.message || 'Failed to get insights');
      }
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to get insights',
      };
    }
  }
}

export default new ApiService();
