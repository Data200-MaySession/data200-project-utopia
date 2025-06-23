import { Link } from 'react-router-dom';
import {
  FilmIcon,
  AcademicCapIcon,
  DocumentTextIcon,
  UserGroupIcon,
  HeartIcon,
  GlobeAltIcon,
  SparklesIcon,
  InformationCircleIcon,
  RocketLaunchIcon
} from '@heroicons/react/24/outline';

const Footer = () => {
  return (
    <footer className="bg-gradient-to-b from-dark-900 to-black text-gray-400 relative overflow-hidden">
      {/* Background Elements */}
      <div className="absolute inset-0 z-0">
        <div className="absolute inset-0 bg-gradient-to-b from-primary-900/10 to-dark-900/30" />
        <div className="absolute -top-40 -left-40 w-80 h-80 bg-primary-500/5 rounded-full filter blur-3xl" />
        <div className="absolute -bottom-40 -right-40 w-80 h-80 bg-primary-500/5 rounded-full filter blur-3xl" />
      </div>

      {/* Content */}
      <div className="relative z-10 max-w-7xl mx-auto pt-16 pb-8 px-4 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-12 mb-12">
          {/* Brand section */}
          <div className="col-span-1 lg:col-span-1">
            <Link 
              to="/" 
              className="inline-flex items-center space-x-2 text-2xl font-display text-white"
            >
              <FilmIcon className="h-8 w-8 text-primary-500" />
              <span>MoviePredictor</span>
            </Link>
            <p className="mt-4 text-sm leading-6 text-gray-300">
              Experience the future of movie success prediction through advanced AI. 
              Challenge our machine learning model and discover what makes a movie truly successful.
            </p>
            <div className="mt-6 flex space-x-4">
              <a 
                href="https://github.com/prashantkoirala/Movie-Success-Predictor" 
                target="_blank" 
                rel="noopener noreferrer"
                className="text-gray-400 hover:text-white transition-colors duration-200"
              >
                <span className="sr-only">GitHub</span>
                <svg className="h-6 w-6" fill="currentColor" viewBox="0 0 24 24">
                  <path fillRule="evenodd" d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z" clipRule="evenodd" />
                </svg>
              </a>
            </div>
          </div>

          {/* Features */}
          <div>
            <h3 className="text-sm font-semibold text-white tracking-wider uppercase mb-4">
              Features
            </h3>
            <ul className="space-y-3">
              <li>
                <Link to="/quiz" className="group flex items-center text-sm hover:text-white transition-colors duration-200">
                  <SparklesIcon className="h-5 w-5 mr-2 text-primary-500 group-hover:text-primary-400" />
                  Movie Quiz
                </Link>
              </li>
              <li>
                <Link to="/features" className="group flex items-center text-sm hover:text-white transition-colors duration-200">
                  <RocketLaunchIcon className="h-5 w-5 mr-2 text-primary-500 group-hover:text-primary-400" />
                  Features
                </Link>
              </li>
              <li>
                <Link to="/about" className="group flex items-center text-sm hover:text-white transition-colors duration-200">
                  <InformationCircleIcon className="h-5 w-5 mr-2 text-primary-500 group-hover:text-primary-400" />
                  About
                </Link>
              </li>
              <li>
                <Link to="/team" className="group flex items-center text-sm hover:text-white transition-colors duration-200">
                  <UserGroupIcon className="h-5 w-5 mr-2 text-primary-500 group-hover:text-primary-400" />
                  Team
                </Link>
              </li>
            </ul>
          </div>

          {/* Resources */}
          <div>
            <h3 className="text-sm font-semibold text-white tracking-wider uppercase mb-4">
              Resources
            </h3>
            <ul className="space-y-3">
              <li>
                <a 
                  href="https://www.themoviedb.org/" 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="group flex items-center text-sm hover:text-white transition-colors duration-200"
                >
                  <GlobeAltIcon className="h-5 w-5 mr-2 text-primary-500 group-hover:text-primary-400" />
                  TMDB API
                </a>
              </li>
              <li>
                <Link to="/docs" className="group flex items-center text-sm hover:text-white transition-colors duration-200">
                  <DocumentTextIcon className="h-5 w-5 mr-2 text-primary-500 group-hover:text-primary-400" />
                  Documentation
                </Link>
              </li>
              <li>
                <a 
                  href="https://scikit-learn.org/" 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="group flex items-center text-sm hover:text-white transition-colors duration-200"
                >
                  <AcademicCapIcon className="h-5 w-5 mr-2 text-primary-500 group-hover:text-primary-400" />
                  scikit-learn
                </a>
              </li>
            </ul>
          </div>

          {/* Project Info */}
          <div>
            <h3 className="text-sm font-semibold text-white tracking-wider uppercase mb-4">
              Project Info
            </h3>
            <ul className="space-y-3">
              <li>
                <p className="text-sm">
                  DATA 200 Final Project
                </p>
              </li>
              <li>
                <p className="text-sm">
                  Model Accuracy: 70.47%
                </p>
              </li>
              <li>
                <p className="text-sm">
                  Dataset: 375,377 movies
                </p>
              </li>
            </ul>
          </div>
        </div>

        {/* Bottom section with gradient border */}
        <div className="relative">
          <div className="absolute inset-0 flex items-center" aria-hidden="true">
            <div className="w-full border-t border-gray-800" />
          </div>
          <div className="relative flex justify-center">
            <span className="px-2 bg-gradient-to-b from-dark-900 to-black text-sm text-gray-500">
              MoviePredictor
            </span>
          </div>
        </div>

        {/* Copyright */}
        <div className="mt-8 flex flex-col items-center">
          <p className="text-sm text-gray-400">
            © {new Date().getFullYear()} MoviePredictor. All rights reserved.
          </p>
          <p className="mt-2 text-sm text-gray-500">
            Made with <HeartIcon className="inline-block h-4 w-4 text-red-500" /> by Prashant, Aaska & Aishmita
          </p>
        </div>
      </div>
    </footer>
  );
};

export default Footer; 