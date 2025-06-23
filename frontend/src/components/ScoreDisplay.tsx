import React from 'react';
import { motion } from 'framer-motion';
import { StarIcon } from '@heroicons/react/24/outline';

interface ScoreDisplayProps {
  score: number;
  totalQuestions: number; // Or rounds played, etc.
}

const ScoreDisplay = ({ score, totalQuestions }: ScoreDisplayProps) => {
  const percentage = totalQuestions > 0 ? Math.round((score / totalQuestions) * 100) : 0;

  return (
    <div className="relative">
      {/* Score card */}
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-lg font-semibold text-white flex items-center gap-2">
            <StarIcon className="w-5 h-5 text-primary-400" />
            Your Score
          </h3>
          <p className="text-sm text-gray-400 mt-1">
            {totalQuestions} questions answered
          </p>
        </div>
        <div className="text-right">
          <motion.p
            key={score}
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-3xl font-display font-bold text-white"
          >
            {score}
            <span className="text-gray-400 text-xl">/{totalQuestions}</span>
          </motion.p>
          <p className="text-sm text-gray-400">
            {percentage}% correct
          </p>
        </div>
      </div>

      {/* Progress bar */}
      <div className="mt-4 h-2 bg-dark-600 rounded-full overflow-hidden">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${percentage}%` }}
          transition={{ duration: 0.5, ease: "easeOut" }}
          className="h-full bg-gradient-to-r from-primary-500 to-blue-500 rounded-full"
        />
      </div>

      {/* Decorative elements */}
      <div className="absolute -inset-4 -z-10 bg-gradient-to-r from-primary-500/10 to-blue-500/10 rounded-lg blur-lg opacity-50" />
    </div>
  );
};

export default ScoreDisplay; 