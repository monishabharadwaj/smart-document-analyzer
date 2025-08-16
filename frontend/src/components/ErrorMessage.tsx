import React from 'react';
import { AlertCircle, X } from 'lucide-react';

interface ErrorMessageProps {
  message: string;
  onDismiss?: () => void;
  variant?: 'error' | 'warning';
}

/**
 * Error message component for displaying API errors and validation messages
 * Includes dismiss functionality and different styling variants
 */
export const ErrorMessage: React.FC<ErrorMessageProps> = ({ 
  message, 
  onDismiss, 
  variant = 'error' 
}) => {
  const bgColor = variant === 'error' ? 'bg-red-50' : 'bg-amber-50';
  const textColor = variant === 'error' ? 'text-red-800' : 'text-amber-800';
  const iconColor = variant === 'error' ? 'text-red-500' : 'text-amber-500';

  return (
    <div className={`${bgColor} border border-opacity-20 rounded-lg p-4 mb-4`}>
      <div className="flex items-center">
        <AlertCircle className={`w-5 h-5 ${iconColor} mr-2 flex-shrink-0`} />
        <p className={`${textColor} text-sm flex-1`}>{message}</p>
        {onDismiss && (
          <button
            onClick={onDismiss}
            className={`${iconColor} hover:opacity-75 ml-2`}
          >
            <X className="w-4 h-4" />
          </button>
        )}
      </div>
    </div>
  );
};

export default ErrorMessage;