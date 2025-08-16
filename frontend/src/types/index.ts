// Type definitions for the Smart Document Analyzer application

export interface Document {
  id: string;
  filename: string;
  uploadedAt: string;
  size: number;
  type: string;
}

export interface ClassificationResult {
  document_id: string;
  classification: {
    type: string;
    confidence: number;
  };
  processed_at: string;
}

export interface EntityResult {
  document_id: string;
  entities: {
    text: string;
    label: string;
    start: number;
    end: number;
    confidence?: number;
  }[];
  processed_at: string;
}

export interface SummaryResult {
  document_id: string;
  summary: string;
  word_count: {
    original: number;
    summary: number;
  };
  processed_at: string;
}

export interface SearchResult {
  document_id: string;
  filename: string;
  relevance_score: number;
  snippet: string;
  metadata?: {
    page?: number;
    section?: string;
  };
}

export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
}

export interface LoadingState {
  isLoading: boolean;
  message?: string;
}

export interface ErrorState {
  hasError: boolean;
  message?: string;
}