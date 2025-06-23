import React, { useState, useEffect, useCallback, useRef } from 'react';
import Confetti from 'react-confetti';
import useWindowSize from 'react-use/lib/useWindowSize';
import MovieCard from '../components/MovieCard';
import QuizControls from '../components/QuizControls';
import ScoreDisplay from '../components/ScoreDisplay';
import FeedbackMessage from '../components/FeedbackMessage';
import FilterControls from '../components/FilterControls';
import {
  getMovieToGuess,
  submitGuess,
  getFilterOptions,
} from '../services/backendService';
import type {
  QuizMovieFromBackend,
  GuessResponse,
  FilterOptions,
  GetMovieParams,
} from '../services/backendService';
import { getPosterUrl } from '../services/tmdbService';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Cog6ToothIcon,
  PlayIcon,
  ArrowPathIcon,
  BoltIcon,
  FireIcon,
  TrophyIcon,
  CloudIcon,
  FilmIcon,
  ClockIcon,
  SparklesIcon,
  VideoCameraIcon,
  HeartIcon,
  RocketLaunchIcon,
  FaceSmileIcon,
  TicketIcon,
  ArrowLeftIcon
} from '@heroicons/react/24/outline';

// Types
type QuizState = 'MODE_SELECT' | 'CATEGORY_SELECT' | 'PLAYING' | 'GAME_OVER' | 'TIME_UP';

interface GameMode {
  id: string;
  name: string;
  description: string;
  totalQuestions?: number;
  timeLimit?: number; // in seconds
  icon?: React.ElementType;
  color?: string;
  difficulty: 'Easy' | 'Medium' | 'Hard';
}

// Game Modes Definition
const GAME_MODES: GameMode[] = [
  {
    id: 'chill_mode',
    name: 'Chill Mode',
    description: 'Take your time, no pressure! Perfect for beginners.',
    icon: CloudIcon,
    color: 'from-sky-500 to-blue-600',
    difficulty: 'Easy'
  },
  {
    id: 'quick_five',
    name: 'Quick Five',
    description: 'Five movies, 30 seconds each. Can you handle it?',
    totalQuestions: 5,
    timeLimit: 30,
    icon: BoltIcon,
    color: 'from-amber-500 to-orange-600',
    difficulty: 'Medium'
  },
  {
    id: 'speed_demon',
    name: 'Speed Demon',
    description: '10 movies, just 15 seconds each! Lightning fast!',
    totalQuestions: 10,
    timeLimit: 15,
    icon: FireIcon,
    color: 'from-rose-500 to-red-600',
    difficulty: 'Hard'
  },
  {
    id: 'movie_master',
    name: 'Movie Master',
    description: '20 movies, 10 seconds each. Only for the elite!',
    totalQuestions: 20,
    timeLimit: 10,
    icon: SparklesIcon,
    color: 'from-violet-500 to-purple-600',
    difficulty: 'Hard'
  }
];

// Time's up messages with humor
const TIME_UP_MESSAGES = [
  "Oops! Time flies when you're overthinking! 🐌",
  "Even Internet Explorer loads faster than your decision! 😅",
  "Breaking News: Snails Are Faster Than Your Choices! 🐌",
  "Did you fall asleep? Wake up, movie star! ⏰",
  "Plot twist: The timer won, you lost! 🎬",
  "Your decision-making needs a software update! 🤖",
  "Error 404: Quick Decisions Not Found! 🤔",
  "Loading... Still loading... Maybe next time! 💭"
];

const getRandomTimeUpMessage = () => {
  return TIME_UP_MESSAGES[Math.floor(Math.random() * TIME_UP_MESSAGES.length)];
};

// Simple component for wrong guess visual effect
const WrongGuessEffect: React.FC<{ active: boolean }> = ({ active }) => {
  if (!active) return null;
  return (
    <div 
      className="fixed inset-0 bg-red-600/30 backdrop-blur-[2px] z-[1000] wrong-guess-flash"
      style={{
        backgroundImage: `
          radial-gradient(circle at 20% 20%, rgba(220, 38, 38, 0.4) 0%, transparent 50%),
          radial-gradient(circle at 80% 80%, rgba(220, 38, 38, 0.4) 0%, transparent 50%)
        `
      }}
    />
  );
};

// Add this helper function at the top level
const getScoreMessage = (score: number, total: number): string => {
  const percentage = (score / total) * 100;
  if (percentage === 100) return "Perfect Score! You're a Movie Oracle! 🏆";
  if (percentage >= 80) return "Amazing Performance! You're a Movie Expert! 🌟";
  if (percentage >= 60) return "Great Job! You've Got the Movie Magic! ✨";
  if (percentage >= 40) return "Not Bad! Keep Practicing! 🎬";
  return "Room for Improvement! Try Again! 🎯";
};

// Genre card data with enhanced visuals
const GENRE_CARDS = {
  'Action': {
    icon: FireIcon,
    gradient: 'from-rose-500 to-orange-500',
    hoverGradient: 'from-rose-600 to-orange-600',
    description: 'High-octane thrills and epic adventures',
    pattern: "url(\"data:image/svg+xml,%3Csvg width='20' height='20' viewBox='0 0 20 20' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='%23ffffff' fill-opacity='0.05' fill-rule='evenodd'%3E%3Ccircle cx='3' cy='3' r='3'/%3E%3Ccircle cx='13' cy='13' r='3'/%3E%3C/g%3E%3C/svg%3E\")"
  },
  'Comedy': {
    icon: FaceSmileIcon,
    gradient: 'from-amber-400 to-yellow-500',
    hoverGradient: 'from-amber-500 to-yellow-600',
    description: 'Laugh-out-loud moments and witty humor',
    pattern: "url(\"data:image/svg+xml,%3Csvg width='20' height='20' viewBox='0 0 20 20' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M0 0h20L0 20z' fill='%23ffffff' fill-opacity='0.05'/%3E%3C/svg%3E\")"
  },
  'Drama': {
    icon: HeartIcon,
    gradient: 'from-purple-500 to-indigo-500',
    hoverGradient: 'from-purple-600 to-indigo-600',
    description: 'Emotional journeys and compelling stories',
    pattern: "url(\"data:image/svg+xml,%3Csvg width='20' height='20' viewBox='0 0 20 20' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M10 0l10 20H0z' fill='%23ffffff' fill-opacity='0.05'/%3E%3C/svg%3E\")"
  },
  'Sci-Fi': {
    icon: RocketLaunchIcon,
    gradient: 'from-cyan-400 to-blue-500',
    hoverGradient: 'from-cyan-500 to-blue-600',
    description: 'Mind-bending futures and space odysseys',
    pattern: "url(\"data:image/svg+xml,%3Csvg width='20' height='20' viewBox='0 0 20 20' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='%23ffffff' fill-opacity='0.05' fill-rule='evenodd'%3E%3Cpath d='M0 0h20v20H0z'/%3E%3Cpath d='M10 10l10-10v10z'/%3E%3C/g%3E%3C/svg%3E\")"
  },
  'Horror': {
    icon: VideoCameraIcon,
    gradient: 'from-red-900 to-red-700',
    hoverGradient: 'from-red-800 to-red-600',
    description: 'Spine-chilling tales and terrifying thrills',
    pattern: "url(\"data:image/svg+xml,%3Csvg width='20' height='20' viewBox='0 0 20 20' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M0 0h20v20H0z' fill='%23ffffff' fill-opacity='0.05'/%3E%3C/svg%3E\")"
  },
  'Romance': {
    icon: HeartIcon,
    gradient: 'from-pink-400 to-rose-500',
    hoverGradient: 'from-pink-500 to-rose-600',
    description: 'Love stories that touch the heart',
    pattern: "url(\"data:image/svg+xml,%3Csvg width='20' height='20' viewBox='0 0 20 20' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M10 0l10 20H0z' fill='%23ffffff' fill-opacity='0.05'/%3E%3C/svg%3E\")"
  },
  'Animation': {
    icon: SparklesIcon,
    gradient: 'from-green-400 to-teal-500',
    hoverGradient: 'from-green-500 to-teal-600',
    description: 'Magical worlds of animated wonder',
    pattern: "url(\"data:image/svg+xml,%3Csvg width='20' height='20' viewBox='0 0 20 20' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M0 0h20L0 20z' fill='%23ffffff' fill-opacity='0.05'/%3E%3C/svg%3E\")"
  },
  'Documentary': {
    icon: FilmIcon,
    gradient: 'from-slate-500 to-slate-700',
    hoverGradient: 'from-slate-600 to-slate-800',
    description: 'Real stories that inspire and inform',
    pattern: "url(\"data:image/svg+xml,%3Csvg width='20' height='20' viewBox='0 0 20 20' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M0 0h20v20H0z' fill='%23ffffff' fill-opacity='0.05'/%3E%3C/svg%3E\")"
  }
};

const QuizPage: React.FC = () => {
  const [currentMovie, setCurrentMovie] = useState<QuizMovieFromBackend | null>(null);
  const [score, setScore] = useState(0);
  const [totalQuestionsAnswered, setTotalQuestionsAnswered] = useState(0);
  const [feedback, setFeedback] = useState<{ message: string; isCorrect: boolean | null }>({
    message: '',
    isCorrect: null,
  });
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // State for filters/categories
  const [filterOptions, setFilterOptions] = useState<FilterOptions | null>(null);
  const [selectedCategories, setSelectedCategories] = useState<GetMovieParams>({});

  // State for confetti and effects
  const [showConfetti, setShowConfetti] = useState(false);
  const [showWrongGuessEffect, setShowWrongGuessEffect] = useState(false);
  const { width, height } = useWindowSize();

  // New Quiz State Management
  const [quizState, setQuizState] = useState<QuizState>('MODE_SELECT');
  const [selectedGameMode, setSelectedGameMode] = useState<GameMode>(GAME_MODES[0]);
  const [questionsRemaining, setQuestionsRemaining] = useState<number | undefined>(
    GAME_MODES[0].totalQuestions
  );
  const questionsRemainingRef = useRef(questionsRemaining);

  // Audio Refs for sound effects
  const correctSoundRef = useRef<HTMLAudioElement>(null);
  const wrongSoundRef = useRef<HTMLAudioElement>(null);
  const gameStartSoundRef = useRef<HTMLAudioElement>(null);
  const gameOverSoundRef = useRef<HTMLAudioElement>(null);

  // Add new state variables for timer
  const [timeLeft, setTimeLeft] = useState<number | null>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const [timeUpMessage, setTimeUpMessage] = useState<string>("");

  // Fetch filter options on mount (used for category selection)
  useEffect(() => {
    const fetchOptions = async () => {
      try {
        const options = await getFilterOptions();
        setFilterOptions(options);
      } catch (err) {
        console.error("Error fetching filter options:", err);
        setError("Could not load filter options.");
      }
    };
    fetchOptions();
  }, []);

  // Update ref whenever questionsRemaining state changes
  useEffect(() => {
    questionsRemainingRef.current = questionsRemaining;
  }, [questionsRemaining]);

  // Clear timer on component unmount
  useEffect(() => {
    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current);
      }
    };
  }, []);

  // Timer effect
  useEffect(() => {
    // Clear any existing timer
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }

    // Only start timer if we're in PLAYING state and have a time limit
    if (quizState === 'PLAYING' && selectedGameMode.timeLimit && timeLeft !== null) {
      if (timeLeft <= 0) {
        handleTimeUp();
      } else {
        timerRef.current = setInterval(() => {
          setTimeLeft(prev => {
            if (prev === null || prev <= 0) {
              if (timerRef.current) {
                clearInterval(timerRef.current);
              }
              handleTimeUp();
              return 0;
            }
            return prev - 1;
          });
        }, 1000);
      }
    }

    // Cleanup function
    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
    };
  }, [timeLeft, quizState, selectedGameMode.timeLimit]);

  // Add this at the beginning of the component
  useEffect(() => {
    // Hide footer when component mounts
    document.body.classList.add('hide-footer');
    
    // Show footer when component unmounts
    return () => {
      document.body.classList.remove('hide-footer');
    };
  }, []);

  const resetQuizState = () => {
    setScore(0);
    setTotalQuestionsAnswered(0);
    setCurrentMovie(null);
    setFeedback({ message: '', isCorrect: null });
    setIsLoading(false);
    setError(null);
    setShowConfetti(false);
    setShowWrongGuessEffect(false);
    if (selectedGameMode.totalQuestions) {
      setQuestionsRemaining(selectedGameMode.totalQuestions);
      questionsRemainingRef.current = selectedGameMode.totalQuestions; // Also update ref
    } else {
      setQuestionsRemaining(undefined);
      questionsRemainingRef.current = undefined; // Also update ref
    }
  };
  
  const startGame = () => {
    resetQuizState();
    setQuizState('PLAYING');
    if (selectedGameMode.timeLimit) {
      setTimeLeft(selectedGameMode.timeLimit);
    }
    loadNextMovie(selectedCategories);
    if (gameStartSoundRef.current) {
      gameStartSoundRef.current.play().catch(e => console.log("Error playing start sound:", e));
    }
  };

  const loadNextMovie = useCallback(async (filtersToApply?: GetMovieParams) => {
    setIsLoading(true);
    setError(null);
    setFeedback({ message: '', isCorrect: null });
    setCurrentMovie(null); // Clear current movie before loading next
    setShowConfetti(false);
    setShowWrongGuessEffect(false);
    
    // Reset timer for each new movie
    if (selectedGameMode.timeLimit) {
      if (timerRef.current) {
        clearInterval(timerRef.current);
      }
      setTimeLeft(selectedGameMode.timeLimit);
    }

    try {
      await new Promise(resolve => setTimeout(resolve, 500)); // Simulate loading
      const movie = await getMovieToGuess(filtersToApply || selectedCategories);
      setCurrentMovie(movie);
    } catch (err: any) {
      console.error("Error loading next movie:", err);
      setError(err.message || "Oops! We couldn't load the next movie. Please try again.");
    } finally {
      setIsLoading(false);
    }
  }, [selectedCategories, selectedGameMode.timeLimit]);

  const handleCategoryChange = (filterType: keyof GetMovieParams, value: string) => {
    setSelectedCategories(prev => ({ ...prev, [filterType]: value || undefined }));
  };

  const handleGuess = async (guess: 'Hit' | 'Flop') => {
    if (!currentMovie || quizState !== 'PLAYING') return;
    
    // Clear current timer
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
    
    setIsLoading(true);
    setError(null);
    
    try {
      await new Promise(resolve => setTimeout(resolve, 750));
      const result: GuessResponse = await submitGuess(currentMovie.id, guess);
      setFeedback({
        message: result.feedbackMessage,
        isCorrect: result.isCorrect,
      });
      setScore((prev) => (result.isCorrect ? prev + 1 : prev));
      setTotalQuestionsAnswered((prev) => prev + 1);
      
      let newQuestionsRemaining = questionsRemainingRef.current;
      if (selectedGameMode.totalQuestions && newQuestionsRemaining !== undefined) {
        newQuestionsRemaining = newQuestionsRemaining - 1;
        setQuestionsRemaining(newQuestionsRemaining);
      }

      if (result.isCorrect) {
        // Correct guess effects
        setShowConfetti(true);
        if (correctSoundRef.current) {
          correctSoundRef.current.currentTime = 0; // Reset sound to start
          correctSoundRef.current.volume = 0.5; // Set volume to 50%
          correctSoundRef.current.play().catch(e => console.log("Error playing correct sound:", e));
        }
        setTimeout(() => setShowConfetti(false), 5000);
      } else {
        // Wrong guess effects
        setShowWrongGuessEffect(true);
        if (wrongSoundRef.current) {
          wrongSoundRef.current.currentTime = 0; // Reset sound to start
          wrongSoundRef.current.volume = 0.5; // Set volume to 50%
          wrongSoundRef.current.play().catch(e => console.log("Error playing wrong sound:", e));
        }
        // Add screen shake effect
        document.body.classList.add('shake');
        setTimeout(() => {
          document.body.classList.remove('shake');
          setShowWrongGuessEffect(false);
        }, 820); // Slightly longer than the animation to ensure smooth transition
      }

      const nextQuestionDelay = result.isCorrect ? 4500 : 3000;
      
      setTimeout(() => {
        if (selectedGameMode.totalQuestions && newQuestionsRemaining !== undefined && newQuestionsRemaining <= 0) {
          setQuizState('GAME_OVER');
          if (gameOverSoundRef.current) {
            gameOverSoundRef.current.currentTime = 0;
            gameOverSoundRef.current.volume = 0.5;
            gameOverSoundRef.current.play().catch(e => console.log("Error playing game over sound:", e));
          }
        } else {
          loadNextMovie(selectedCategories);
        }
      }, nextQuestionDelay);

    } catch (err: any) {
      console.error("Error submitting guess:", err);
      setError(err.message || "Oh no! Your guess couldn't be submitted. Let's try again.");
    }
  };

  const handleNextMovieAfterFeedback = () => {
    if (selectedGameMode.totalQuestions && questionsRemainingRef.current !== undefined && questionsRemainingRef.current <= 0) {
      setQuizState('GAME_OVER');
      if (gameOverSoundRef.current) {
        gameOverSoundRef.current.play().catch(e => console.log("Error playing game over sound:", e));
      }
    } else {
      loadNextMovie(selectedCategories);
    }
  };

  const handleTimeUp = () => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
    }
    setTimeUpMessage(getRandomTimeUpMessage());
    setQuizState('TIME_UP');
    if (wrongSoundRef.current) {
      wrongSoundRef.current.play().catch(e => console.log("Error playing wrong sound:", e));
    }
  };

  let posterUrlForCard = '';
  if (currentMovie && currentMovie.posterUrl) {
    const fullPosterPath = getPosterUrl(currentMovie.posterUrl);
    if (fullPosterPath) {
      posterUrlForCard = fullPosterPath;
    }
    // If fullPosterPath is null, posterUrlForCard remains '', which is a valid string for MovieCard
  }

  // Conditional Rendering based on QuizState

  if (quizState === 'MODE_SELECT') {
    return (
      <div className="min-h-screen bg-dark-900 flex flex-col items-center justify-center relative overflow-hidden">
        {/* Animated background gradients */}
        <div className="absolute inset-0 z-0">
          <div className="absolute top-0 left-0 w-full h-full bg-gradient-to-br from-primary-900/30 via-dark-900 to-dark-900" />
          <div className="absolute -top-40 -left-40 w-80 h-80 bg-primary-500/10 rounded-full filter blur-3xl animate-pulse" />
          <div className="absolute -bottom-40 -right-40 w-80 h-80 bg-primary-500/10 rounded-full filter blur-3xl animate-pulse delay-1000" />
        </div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="relative z-10 w-full max-w-4xl px-4 py-8"
        >
          <div className="text-center mb-12">
            <motion.h1
              initial={{ opacity: 0, y: -20 }}
              animate={{ opacity: 1, y: 0 }}
              className="text-5xl sm:text-6xl font-display font-bold text-white mb-4"
            >
              Choose Your Challenge
            </motion.h1>
            <motion.p
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 }}
              className="text-xl text-gray-300"
            >
              Select your preferred game mode and difficulty
            </motion.p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {GAME_MODES.map((mode, index) => {
              const Icon = mode.icon;
              return (
                <motion.button
                  key={mode.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.1 }}
                  onClick={() => {
                    setSelectedGameMode(mode);
                    setQuizState('CATEGORY_SELECT');
                  }}
                  className={`
                    relative group p-6 rounded-xl border-2 transition-all duration-300
                    ${selectedGameMode.id === mode.id
                      ? `bg-gradient-to-br ${mode.color} border-transparent`
                      : 'bg-dark-800/50 border-dark-600 hover:border-primary-500/50'
                    }
                    backdrop-blur-sm hover:scale-[1.02]
                  `}
                >
                  {/* Difficulty Badge */}
                  <div className={`
                    absolute top-4 right-4 px-3 py-1 rounded-full text-sm font-medium
                    ${mode.difficulty === 'Easy' ? 'bg-emerald-500/20 text-emerald-300' :
                      mode.difficulty === 'Medium' ? 'bg-amber-500/20 text-amber-300' :
                      'bg-rose-500/20 text-rose-300'}
                  `}>
                    {mode.difficulty}
                  </div>

                  <div className="flex flex-col items-center text-center">
                    {Icon && (
                      <div className={`
                        p-4 rounded-full mb-4
                        ${selectedGameMode.id === mode.id
                          ? 'bg-white/20'
                          : 'bg-dark-700 group-hover:bg-dark-600'
                        }
                      `}>
                        <Icon className="w-8 h-8 text-white" />
                      </div>
                    )}
                    <h3 className="text-xl font-bold text-white mb-2">{mode.name}</h3>
                    <p className="text-gray-300 mb-4">{mode.description}</p>
                    {(mode.totalQuestions || mode.timeLimit) && (
                      <div className="flex items-center justify-center space-x-4 text-sm text-gray-400">
                        {mode.totalQuestions && (
                          <span className="flex items-center">
                            <FilmIcon className="w-4 h-4 mr-1" />
                            {mode.totalQuestions} Movies
                          </span>
                        )}
                        {mode.timeLimit && (
                          <span className="flex items-center">
                            <ClockIcon className="w-4 h-4 mr-1" />
                            {mode.timeLimit}s Each
                          </span>
                        )}
                      </div>
                    )}
                  </div>

                  {/* Hover effect overlay */}
                  <div className="absolute inset-0 bg-gradient-to-t from-white/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300 rounded-xl" />
                </motion.button>
              );
            })}
          </div>
        </motion.div>
      </div>
    );
  }

  if (quizState === 'CATEGORY_SELECT') {
    return (
      <div className="min-h-screen bg-gradient-to-b from-dark-900 via-dark-800 to-dark-900 flex flex-col items-center justify-center relative overflow-hidden">
        {/* Premium animated background */}
        <div className="fixed inset-0 z-0">
          {/* Base gradient layer */}
          <div className="absolute inset-0 bg-gradient-to-br from-dark-800/80 via-dark-900/90 to-dark-800/80" />
          
          {/* Subtle color tint layer */}
          <div className="absolute inset-0 bg-gradient-to-tr from-primary-900/5 via-transparent to-blue-900/5" />
          
          {/* Animated gradient orbs */}
          <div className="absolute top-0 left-0 w-full h-full overflow-hidden">
            {/* Primary orb */}
            <motion.div
              animate={{
                scale: [1, 1.2, 1],
                opacity: [0.15, 0.25, 0.15],
                y: [0, -20, 0],
              }}
              transition={{
                duration: 10,
                repeat: Infinity,
                ease: "easeInOut"
              }}
              className="absolute -top-40 -left-40 w-[600px] h-[600px] rounded-full"
              style={{
                background: 'radial-gradient(circle at center, rgba(var(--primary-500-rgb), 0.15) 0%, rgba(var(--primary-500-rgb), 0.05) 50%, transparent 70%)',
                filter: 'blur(40px)'
              }}
            />

            {/* Secondary orb */}
            <motion.div
              animate={{
                scale: [1.2, 1, 1.2],
                opacity: [0.15, 0.25, 0.15],
                y: [0, 20, 0],
              }}
              transition={{
                duration: 12,
                repeat: Infinity,
                ease: "easeInOut",
                delay: 1
              }}
              className="absolute -bottom-40 -right-40 w-[600px] h-[600px] rounded-full"
              style={{
                background: 'radial-gradient(circle at center, rgba(59, 130, 246, 0.15) 0%, rgba(59, 130, 246, 0.05) 50%, transparent 70%)',
                filter: 'blur(40px)'
              }}
            />

            {/* Accent orb */}
            <motion.div
              animate={{
                scale: [1, 1.3, 1],
                opacity: [0.1, 0.2, 0.1],
                x: [0, -20, 0],
              }}
              transition={{
                duration: 15,
                repeat: Infinity,
                ease: "easeInOut",
                delay: 2
              }}
              className="absolute top-1/2 right-1/4 w-[400px] h-[400px] rounded-full"
              style={{
                background: 'radial-gradient(circle at center, rgba(124, 58, 237, 0.15) 0%, rgba(124, 58, 237, 0.05) 50%, transparent 70%)',
                filter: 'blur(40px)'
              }}
            />
          </div>

          {/* Premium grid pattern */}
          <div className="absolute inset-0">
            <div 
              className="absolute inset-0 opacity-[0.07]"
              style={{
                backgroundImage: `
                  linear-gradient(to right, rgba(255,255,255,0.1) 1px, transparent 1px),
                  linear-gradient(to bottom, rgba(255,255,255,0.1) 1px, transparent 1px)
                `,
                backgroundSize: '4rem 4rem',
                maskImage: 'radial-gradient(circle at center, black, transparent 80%)'
              }}
            />
          </div>

          {/* Subtle noise texture */}
          <div 
            className="absolute inset-0 opacity-[0.02]" 
            style={{ 
              backgroundImage: `url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.65' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%' height='100%' filter='url(%23noise)' fill='white'/%3E%3C/svg%3E")`,
              filter: 'contrast(300%) brightness(100%)'
            }}
          />
        </div>

        <div className="relative z-10 w-full max-w-7xl px-4 py-12">
          {/* Header Section */}
          <motion.div 
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-center mb-16"
          >
            <motion.div
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              transition={{ type: "spring", duration: 0.5 }}
              className="inline-flex items-center justify-center w-20 h-20 rounded-2xl bg-gradient-to-br from-primary-500/20 to-primary-500/10 backdrop-blur-sm border border-primary-500/20 mb-6"
            >
              <TicketIcon className="w-10 h-10 text-primary-400" />
            </motion.div>
            
            <h1 className="text-5xl sm:text-6xl font-display font-bold bg-clip-text text-transparent bg-gradient-to-r from-white to-gray-400 mb-4">
              Choose Your Genre
            </h1>
            <p className="text-xl text-gray-400">
              Playing <span className="text-primary-400 font-semibold">{selectedGameMode.name}</span>
            </p>
          </motion.div>

          {/* Genre Grid */}
          {filterOptions && (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-2 gap-6 max-w-5xl mx-auto px-4">
              <AnimatePresence>
                {filterOptions.genres.map((genre, index) => {
                  const genreData = GENRE_CARDS[genre as keyof typeof GENRE_CARDS] || {
                    icon: FilmIcon,
                    gradient: 'from-gray-500 to-gray-600',
                    hoverGradient: 'from-gray-600 to-gray-700',
                    description: 'Explore amazing films',
                    pattern: ''
                  };
                  const Icon = genreData.icon;

                  return (
                    <motion.div
                      key={genre}
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, scale: 0.95 }}
                      transition={{ delay: index * 0.1 }}
                      onClick={() => handleCategoryChange('genre', genre)}
                      className={`
                        group cursor-pointer relative overflow-hidden p-6
                        rounded-2xl border-2 transition-all duration-500 backdrop-blur-sm
                        ${selectedCategories.genre === genre
                          ? `bg-gradient-to-br ${genreData.gradient} bg-opacity-30 border-primary-500 shadow-lg shadow-primary-500/20`
                          : `bg-dark-800/20 border-white/5 hover:border-primary-500/50`
                        }
                      `}
                    >
                      {/* Background Gradient */}
                      <div className={`
                        absolute inset-0 transition-all duration-500
                        ${selectedCategories.genre === genre
                          ? `bg-gradient-to-br ${genreData.gradient} opacity-30`
                          : `bg-gradient-to-br ${genreData.gradient} opacity-0 group-hover:opacity-20`
                        }
                      `} />

                      {/* Hover Gradient */}
                      <div className={`
                        absolute inset-0 transition-all duration-500
                        bg-gradient-to-br ${genreData.hoverGradient} opacity-0 group-hover:opacity-20
                      `} />

                      {/* Content Container */}
                      <div className="relative z-10">
                        {/* Icon Container */}
                        <div className={`
                          w-16 h-16 rounded-xl flex items-center justify-center mb-6
                          transition-all duration-500
                          ${selectedCategories.genre === genre
                            ? `bg-gradient-to-br ${genreData.gradient}`
                            : `bg-dark-700 group-hover:bg-gradient-to-br group-hover:${genreData.hoverGradient}`
                          }
                        `}>
                          <Icon className={`
                            w-8 h-8 transition-all duration-500
                            ${selectedCategories.genre === genre
                              ? 'text-white'
                              : 'text-gray-400 group-hover:text-white'
                            }
                          `} />
                        </div>

                        {/* Content */}
                        <div className="space-y-2">
                          <h3 className="text-2xl font-bold text-white">{genre}</h3>
                          <p className="text-gray-400 group-hover:text-gray-300 transition-colors">
                            {genreData.description}
                          </p>
                        </div>

                        {/* Selection Indicator */}
                        {selectedCategories.genre === genre && (
                          <motion.div
                            initial={{ scale: 0 }}
                            animate={{ scale: 1 }}
                            className="absolute top-4 right-4"
                          >
                            <div className={`
                              w-8 h-8 rounded-full flex items-center justify-center
                              bg-gradient-to-br ${genreData.gradient}
                            `}>
                              <SparklesIcon className="w-5 h-5 text-white" />
                            </div>
                          </motion.div>
                        )}
                      </div>
                    </motion.div>
                  );
                })}
              </AnimatePresence>
            </div>
          )}

          {/* Navigation Buttons */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
            className="flex justify-center items-center gap-6 mt-12"
          >
            {/* Back Button */}
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => setQuizState('MODE_SELECT')}
              className="group flex items-center gap-2 px-6 py-3 rounded-xl bg-dark-800/50 border-2 border-dark-700 hover:border-primary-500/30 backdrop-blur-sm transition-all duration-300"
            >
              <ArrowLeftIcon className="w-5 h-5 text-gray-400 group-hover:text-primary-400 transition-colors" />
              <span className="text-gray-400 group-hover:text-white transition-colors">Back</span>
            </motion.button>

            {/* Start Button */}
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={startGame}
              disabled={!selectedCategories.genre}
              className={`
                relative px-8 py-3 rounded-xl font-medium text-lg
                transition-all duration-300 overflow-hidden
                ${selectedCategories.genre
                  ? 'bg-gradient-to-r from-primary-500 to-primary-600 text-white shadow-lg shadow-primary-500/25'
                  : 'bg-dark-700/50 text-gray-500 cursor-not-allowed'
                }
              `}
            >
              <span className="relative z-10">
                {selectedCategories.genre ? 'Start Game' : 'Select a Genre'}
              </span>
              {selectedCategories.genre && (
                <motion.div
                  className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent"
                  initial={{ x: '-100%' }}
                  animate={{ x: '100%' }}
                  transition={{ repeat: Infinity, duration: 1.5, ease: "linear" }}
                />
              )}
            </motion.button>
          </motion.div>
        </div>
      </div>
    );
  }

  if (quizState === 'TIME_UP') {
    return (
      <div className="min-h-screen bg-dark-900 flex flex-col items-center justify-center py-12 px-4">
        <motion.div
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          className="w-full max-w-lg bg-dark-800/70 backdrop-blur-md p-8 rounded-xl shadow-2xl border border-primary-500/20 text-center"
        >
          <motion.div
            initial={{ rotate: 0 }}
            animate={{ rotate: 360 }}
            transition={{ duration: 0.5 }}
            className="w-20 h-20 mx-auto mb-6 text-rose-500"
          >
            <ClockIcon className="w-full h-full" />
          </motion.div>

          <motion.h1
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-4xl font-display font-bold text-white mb-4"
          >
            Time's Up!
          </motion.h1>

          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="text-xl text-gray-300 mb-8"
          >
            {timeUpMessage}
          </motion.p>

          <div className="space-y-4">
            <motion.button
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4 }}
              onClick={() => {
                resetQuizState();
                startGame();
              }}
              className="w-full btn-primary py-4"
            >
              Try Again
            </motion.button>

            <motion.button
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.5 }}
              onClick={() => {
                resetQuizState();
                setQuizState('MODE_SELECT');
              }}
              className="w-full btn-secondary py-4"
            >
              Change Mode
            </motion.button>
          </div>
        </motion.div>
      </div>
    );
  }

  if (quizState === 'GAME_OVER') {
    const scorePercentage = (score / totalQuestionsAnswered) * 100;
    const scoreMessage = getScoreMessage(score, totalQuestionsAnswered);

    return (
      <div className="min-h-screen bg-dark-900 flex flex-col items-center justify-center py-12 px-4 sm:px-6 lg:px-8">
        <motion.div
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          className="w-full max-w-lg bg-dark-800/70 backdrop-blur-md p-8 rounded-xl shadow-2xl border border-primary-500/20 text-center"
        >
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
          >
            <h1 className="text-5xl font-display font-bold text-white mb-2">Game Over!</h1>
            <p className="text-xl text-primary-400 mb-6">{selectedGameMode.name}</p>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.4 }}
            className="relative mb-8"
          >
            <div className="absolute inset-0 bg-gradient-to-r from-primary-500/20 to-primary-600/20 rounded-lg blur-xl" />
            <div className="relative bg-dark-700/50 rounded-lg p-6 border border-primary-500/30">
              <div className="text-4xl font-bold mb-2">
                <span className="text-green-400">{score}</span>
                <span className="text-gray-400">/{totalQuestionsAnswered}</span>
              </div>
              <p className="text-2xl font-medium text-primary-400 mb-4">
                {scorePercentage.toFixed(1)}% Accuracy
              </p>
              <p className="text-lg text-gray-300">{scoreMessage}</p>
            </div>
          </motion.div>

          <div className="space-y-4">
            <motion.button
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.6 }}
              onClick={() => {
                resetQuizState();
                startGame(); // Restart with same settings
              }}
              className="w-full btn-primary py-4 flex items-center justify-center gap-3 relative overflow-hidden group"
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
            >
              <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/10 to-transparent -translate-x-full group-hover:translate-x-full transition-transform duration-1000" />
              <ArrowPathIcon className="w-6 h-6" />
              Play Again (Same Settings)
            </motion.button>

            <motion.button
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.7 }}
              onClick={() => {
                resetQuizState();
                setSelectedGameMode(GAME_MODES[0]);
                setQuestionsRemaining(GAME_MODES[0].totalQuestions);
                setSelectedCategories({});
                setQuizState('MODE_SELECT');
              }}
              className="w-full btn-secondary py-4 flex items-center justify-center gap-3 relative overflow-hidden group"
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
            >
              <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/10 to-transparent -translate-x-full group-hover:translate-x-full transition-transform duration-1000" />
              <Cog6ToothIcon className="w-6 h-6" />
              Change Settings & Play Again
            </motion.button>
          </div>
        </motion.div>

        {scorePercentage >= 80 && <Confetti width={width} height={height} numberOfPieces={150} recycle={false} gravity={0.3} tweenDuration={4000} onConfettiComplete={(confettiInstance: any) => {
          setShowConfetti(false);
          if (confettiInstance) {
            confettiInstance.reset();
          }
        }} />}
      </div>
    );
  }

  // PLAYING State (existing UI, slightly modified)
  return (
    <div className="min-h-screen bg-dark-900 text-white relative overflow-x-hidden pt-24">
      {/* Audio elements */}
      <audio src="/sounds/correct.mp3" ref={correctSoundRef} preload="auto" />
      <audio src="/sounds/wrong.mp3" ref={wrongSoundRef} preload="auto" />
      <audio src="/sounds/game-start.mp3" ref={gameStartSoundRef} preload="auto" />
      <audio src="/sounds/game-over.mp3" ref={gameOverSoundRef} preload="auto" />

      {/* Wrong guess effect overlay */}
      <WrongGuessEffect active={showWrongGuessEffect} />

      {/* Confetti effect */}
      {showConfetti && (
        <Confetti
          width={width}
          height={height}
          numberOfPieces={150}
          recycle={false}
          gravity={0.3}
          tweenDuration={4000}
          onConfettiComplete={(confettiInstance: any) => {
            setShowConfetti(false);
            if (confettiInstance) {
              confettiInstance.reset();
            }
          }}
        />
      )}

      {/* Background Elements */}
      <div className="absolute inset-0 z-0">
        <div className="absolute top-0 left-0 w-full h-full bg-gradient-to-br from-primary-900/30 via-dark-900 to-dark-900" />
        <div className="absolute -top-40 -left-40 w-80 h-80 bg-primary-500/10 rounded-full filter blur-3xl animate-pulse" />
        <div className="absolute -bottom-40 -right-40 w-80 h-80 bg-primary-500/10 rounded-full filter blur-3xl animate-pulse delay-1000" />
      </div>

      {/* Main Content Container */}
      <div className="relative z-10 container mx-auto px-4">
        {/* Header Section with Title and Mode */}
        <div className="flex flex-col items-center justify-center mb-8 relative">
          {/* Title */}
          <motion.h1
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-4xl sm:text-5xl font-display font-bold text-white mb-4"
          >
            Movie Success Quiz
          </motion.h1>

          {/* Game Mode Display */}
          {selectedGameMode && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="flex items-center justify-center gap-2 text-primary-400 mb-4"
            >
              {selectedGameMode.icon && (
                <selectedGameMode.icon className="w-5 h-5" />
              )}
              <span>{selectedGameMode.name}</span>
            </motion.div>
          )}

          {/* Timer - Now below game mode */}
          {quizState === 'PLAYING' && selectedGameMode.timeLimit && (
            <motion.div
              initial={{ opacity: 0, y: -20 }}
              animate={{ opacity: 1, y: 0 }}
            >
              <div className={`
                inline-flex items-center gap-2 px-5 py-2 rounded-lg
                ${timeLeft && timeLeft <= 5 
                  ? 'bg-red-500/90 shadow-sm shadow-red-500/20' 
                  : 'bg-dark-800/90 shadow-sm shadow-primary-500/10'
                }
                backdrop-blur-md border border-white/10
                transition-all duration-300
              `}>
                <ClockIcon className={`
                  w-4 h-4
                  ${timeLeft && timeLeft <= 5 ? 'animate-pulse text-red-200' : 'text-primary-400'}
                `} />
                <span className={`
                  font-mono text-lg font-bold
                  ${timeLeft && timeLeft <= 5 ? 'text-white' : 'text-primary-300'}
                `}>
                  {timeLeft}s
                </span>
              </div>
            </motion.div>
          )}
        </div>

        {/* Conditional content: Error, Loading, or Movie Grid */}
        {error && !currentMovie && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-center my-8 p-6 bg-red-800/30 text-red-300 rounded-xl shadow-lg border border-red-700/50 max-w-md mx-auto"
          >
            <div className="flex justify-center mb-4">
              <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-12 h-12 text-red-400">
                <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m0-10.036A11.959 11.959 0 013.598 6H4v.005A12.06 12.06 0 0012 21.75M21 12a9 9 0 11-18 0 9 9 0 0118 0zM12 12H3.356A11.96 11.96 0 0112 2.25v9.75z" />
              </svg>
            </div>
            <p className="font-bold text-xl text-white mb-2">Oops! Movie Night Glitch</p>
            <p className="text-red-300 mb-6">{error}</p>
            <div className="flex flex-col sm:flex-row justify-center gap-3">
              <button
                onClick={() => loadNextMovie(selectedCategories)} // Retry
                className="btn-secondary py-3 px-6 bg-red-500/80 hover:bg-red-500 text-white border-red-500/0"
              >
                <ArrowPathIcon className="w-5 h-5 mr-2 inline-block" />
                Try Again
              </button>
              <button
                onClick={() => {
                  resetQuizState();
                  //setSelectedCategories({}); // Keep categories for now, user might want to retry same mode/genre if it was a temp error
                  setQuizState('CATEGORY_SELECT'); 
                }}
                className="btn-secondary py-3 px-6"
              >
                Change Genre
              </button>
            </div>
          </motion.div>
        )}

        {!error && !currentMovie && isLoading && (
          <div className="text-center my-12">
            <div className="inline-flex items-center justify-center p-4 bg-primary-500/10 rounded-full mb-4">
              <svg className="animate-spin h-10 w-10 text-primary-400" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
            </div>
            <p className="text-2xl font-display text-gray-300 animate-pulse">Fetching your movie...</p>
            <p className="text-sm text-gray-500">Get ready for the challenge!</p>
          </div>
        )}

        {!error && currentMovie && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 max-w-6xl mx-auto">
            {/* Left column: Movie Card */}
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.2 }}
              className="flex justify-center"
            >
              <MovieCard
                title={currentMovie?.title || 'Loading...'}
                posterUrl={posterUrlForCard}
              />
            </motion.div>

            {/* Right column: Controls and Score */}
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.3 }}
              className="flex flex-col justify-center space-y-8"
            >
              <div className="card">
                <h2 className="text-2xl font-display font-bold text-white mb-6">
                  Make Your Prediction
                </h2>
                <QuizControls
                  onGuess={handleGuess}
                  disabled={isLoading || !!feedback.message}
                />
              </div>

              <AnimatePresence mode="wait">
                {feedback.message && (
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -20 }}
                    className="card"
                  >
                    <FeedbackMessage
                      message={feedback.message}
                      isCorrect={feedback.isCorrect}
                    />
                    <motion.button
                      whileHover={{ scale: 1.02 }}
                      whileTap={{ scale: 0.98 }}
                      onClick={handleNextMovieAfterFeedback}
                      className="btn-primary w-full mt-4"
                      disabled={isLoading}
                    >
                      Next Movie
                    </motion.button>
                  </motion.div>
                )}
              </AnimatePresence>

              <div className="card">
                <ScoreDisplay
                  score={score}
                  totalQuestions={totalQuestionsAnswered}
                />
              </div>
            </motion.div>
          </div>
        )}
      </div>
    </div>
  );
};

export default QuizPage; 