import React from 'react';
import { Loader2 } from 'lucide-react';

interface LoadingSpinnerProps {
  message?: string;
  size?: 'sm' | 'md' | 'lg';
}

/**
 * Reusable loading spinner component with optional message
 * Used throughout the app to indicate processing states
 */
export const LoadingSpinner: React.FC<LoadingSpinnerProps> = ({ 
  message = 'Processing...', 
  size = 'md' 
}) => {
  const sizeClasses = {
    sm: 'w-4 h-4',
    md: 'w-6 h-6',
    lg: 'w-8 h-8',
  };

  return (
    <div className="flex items-center justify-center p-4">
      <div className="text-center">
        <Loader2 className={`${sizeClasses[size]} animate-spin text-blue-600 mx-auto mb-2`} />
        <p className="text-gray-600 text-sm">{message}</p>
      </div>
    </div>
  );
};

export default LoadingSpinner;