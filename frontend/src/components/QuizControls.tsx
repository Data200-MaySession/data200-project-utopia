import React from 'react';
import { motion } from 'framer-motion';
import { TrophyIcon, XCircleIcon } from '@heroicons/react/24/outline';

interface QuizControlsProps {
  onGuess: (guess: 'Hit' | 'Flop') => void;
  disabled?: boolean; // Optional: to disable buttons during loading/feedback
}

const buttonVariants = {
  initial: { scale: 1 },
  hover: { 
    scale: 1.02,
    transition: { 
      type: "spring",
      stiffness: 400,
      damping: 10
    }
  },
  tap: { 
    scale: 0.98,
    transition: {
      type: "spring",
      stiffness: 400,
      damping: 10
    }
  },
  disabled: {
    scale: 1,
    opacity: 0.5
  }
};

const QuizControls = ({ onGuess, disabled = false }: QuizControlsProps) => {
  return (
    <div className="grid grid-cols-2 gap-4">
      <motion.button
        variants={buttonVariants}
        initial="initial"
        whileHover="hover"
        whileTap="tap"
        animate={disabled ? "disabled" : "initial"}
        onClick={() => onGuess('Hit')}
        disabled={disabled}
        className={`
          relative group overflow-hidden rounded-lg px-6 py-4
          bg-gradient-to-br from-emerald-500 to-emerald-600
          disabled:from-emerald-500/50 disabled:to-emerald-600/50
          disabled:cursor-not-allowed
          transition-all duration-300
          shadow-lg hover:shadow-emerald-500/25
        `}
      >
        <motion.div
          initial={{ opacity: 0, scale: 0.8 }}
          whileHover={{ opacity: 1, scale: 1 }}
          className="absolute inset-0 bg-gradient-to-t from-white/20 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300"
        />
        <div className="relative flex items-center justify-center space-x-3">
          <motion.div
            initial={{ rotate: 0 }}
            whileHover={{ rotate: 360 }}
            transition={{ duration: 0.5 }}
          >
            <TrophyIcon className="w-6 h-6" />
          </motion.div>
          <span className="font-medium text-lg">Hit</span>
        </div>
      </motion.button>

      <motion.button
        variants={buttonVariants}
        initial="initial"
        whileHover="hover"
        whileTap="tap"
        animate={disabled ? "disabled" : "initial"}
        onClick={() => onGuess('Flop')}
        disabled={disabled}
        className={`
          relative group overflow-hidden rounded-lg px-6 py-4
          bg-gradient-to-br from-rose-500 to-rose-600
          disabled:from-rose-500/50 disabled:to-rose-600/50
          disabled:cursor-not-allowed
          transition-all duration-300
          shadow-lg hover:shadow-rose-500/25
        `}
      >
        <motion.div
          initial={{ opacity: 0, scale: 0.8 }}
          whileHover={{ opacity: 1, scale: 1 }}
          className="absolute inset-0 bg-gradient-to-t from-white/20 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300"
        />
        <div className="relative flex items-center justify-center space-x-3">
          <motion.div
            initial={{ rotate: 0 }}
            whileHover={{ rotate: 180 }}
            transition={{ duration: 0.3 }}
          >
            <XCircleIcon className="w-6 h-6" />
          </motion.div>
          <span className="font-medium text-lg">Flop</span>
        </div>
      </motion.button>
    </div>
  );
};

export default QuizControls; 