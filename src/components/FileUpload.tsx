import React, { useState, useRef } from 'react';
import { Upload, File, CheckCircle } from 'lucide-react';
import apiService from '../services/api';
import LoadingSpinner from './LoadingSpinner';
import ErrorMessage from './ErrorMessage';

interface FileUploadProps {
  onUploadSuccess: (documentId: string, filename: string, content: string) => void;
}

/**
 * File upload component with drag-and-drop support
 * Handles PDF and DOCX file uploads with validation
 */
export const FileUpload: React.FC<FileUploadProps> = ({ onUploadSuccess }) => {
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadedFile, setUploadedFile] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const acceptedTypes = '.txt,.text';
  const maxFileSize = 10 * 1024 * 1024; // 10MB

  const validateFile = (file: File): string | null => {
    if (!file.type.includes('text/plain') && !file.name.toLowerCase().endsWith('.txt')) {
      return 'Please upload a text file (.txt). PDF/DOCX support coming soon!';
    }
    if (file.size > maxFileSize) {
      return 'File size must be less than 10MB';
    }
    return null;
  };

  const handleFileUpload = async (file: File) => {
    const validationError = validateFile(file);
    if (validationError) {
      setError(validationError);
      return;
    }

    setError(null);
    setIsUploading(true);

    try {
      // Read file content for processing
      const fileContent = await file.text();
      
      const result = await apiService.uploadDocument(file);
      
      if (result.success && result.data) {
        setUploadedFile(file.name);
        onUploadSuccess(result.data.document_id, result.data.filename, fileContent);
      } else {
        setError(result.error || 'Upload failed');
      }
    } catch (err) {
      setError('An unexpected error occurred during upload');
    } finally {
      setIsUploading(false);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    
    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0) {
      handleFileUpload(files[0]);
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      handleFileUpload(files[0]);
    }
  };

  const triggerFileSelect = () => {
    fileInputRef.current?.click();
  };

  if (isUploading) {
    return (
      <div className="bg-white rounded-lg shadow-md p-6">
        <LoadingSpinner message="Uploading document..." />
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <h2 className="text-xl font-semibold text-gray-800 mb-4">Upload Document</h2>
      
      {error && (
        <ErrorMessage 
          message={error} 
          onDismiss={() => setError(null)} 
        />
      )}

      {uploadedFile ? (
        <div className="text-center py-8">
          <CheckCircle className="w-12 h-12 text-green-500 mx-auto mb-4" />
          <p className="text-lg font-medium text-gray-800 mb-2">Upload Successful!</p>
          <p className="text-gray-600 mb-4">{uploadedFile}</p>
          <button
            onClick={() => {
              setUploadedFile(null);
              if (fileInputRef.current) {
                fileInputRef.current.value = '';
              }
            }}
            className="text-blue-600 hover:text-blue-700 font-medium"
          >
            Upload Another Document
          </button>
        </div>
      ) : (
        <>
          <div
            className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors
              ${isDragging 
                ? 'border-blue-500 bg-blue-50' 
                : 'border-gray-300 hover:border-blue-400 hover:bg-gray-50'
              }`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={triggerFileSelect}
          >
            <Upload className="w-12 h-12 text-gray-400 mx-auto mb-4" />
            <p className="text-lg font-medium text-gray-700 mb-2">
              Drop your document here or click to browse
            </p>
            <p className="text-gray-500 text-sm mb-4">
              Supports text files (.txt) up to 10MB. PDF/DOCX support coming soon!
            </p>
            <button
              type="button"
              className="bg-blue-600 text-white px-6 py-2 rounded-lg font-medium hover:bg-blue-700 transition-colors"
            >
              Choose File
            </button>
          </div>

          <input
            ref={fileInputRef}
            type="file"
            accept={acceptedTypes}
            onChange={handleFileSelect}
            className="hidden"
          />
        </>
      )}
    </div>
  );
};

export default FileUpload;