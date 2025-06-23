import React from 'react';
import { motion } from 'framer-motion';
import { CheckCircleIcon, XCircleIcon } from '@heroicons/react/24/outline';

interface FeedbackMessageProps {
  message: string;
  isCorrect: boolean | null;
}

const FeedbackMessage = ({ message, isCorrect }: FeedbackMessageProps) => {
  const variants = {
    hidden: { opacity: 0, y: 20 },
    visible: { opacity: 1, y: 0 },
  };

  return (
    <motion.div
      variants={variants}
      initial="hidden"
      animate="visible"
      className={`
        rounded-lg p-4
        ${isCorrect
          ? 'bg-emerald-500/10 border border-emerald-500/20'
          : 'bg-rose-500/10 border border-rose-500/20'
        }
      `}
    >
      <div className="flex items-start space-x-3">
        <div className="flex-shrink-0">
          {isCorrect ? (
            <CheckCircleIcon
              className="h-6 w-6 text-emerald-400"
              aria-hidden="true"
            />
          ) : (
            <XCircleIcon
              className="h-6 w-6 text-rose-400"
              aria-hidden="true"
            />
          )}
        </div>
        <div className="flex-1 pt-0.5">
          <p className={`
            text-sm font-medium
            ${isCorrect ? 'text-emerald-400' : 'text-rose-400'}
          `}>
            {isCorrect ? 'Correct!' : 'Not quite!'}
          </p>
          <p className="mt-1 text-sm text-gray-300">
            {message}
          </p>
        </div>
      </div>
    </motion.div>
  );
};

export default FeedbackMessage; 