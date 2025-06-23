import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
  BeakerIcon,
  CodeBracketIcon,
  CommandLineIcon,
  CpuChipIcon,
  DocumentTextIcon,
  FolderIcon,
  LightBulbIcon,
  PuzzlePieceIcon,
  RocketLaunchIcon,
  ServerIcon,
  WrenchScrewdriverIcon,
} from '@heroicons/react/24/outline';

const sections = [
  {
    id: 'introduction',
    title: 'Introduction',
    icon: RocketLaunchIcon,
  },
  {
    id: 'getting-started',
    title: 'Getting Started',
    icon: LightBulbIcon,
  },
  {
    id: 'architecture',
    title: 'Architecture',
    icon: PuzzlePieceIcon,
  },
  {
    id: 'frontend',
    title: 'Frontend',
    icon: CodeBracketIcon,
  },
  {
    id: 'backend',
    title: 'Backend',
    icon: ServerIcon,
  },
  {
    id: 'ml-model',
    title: 'ML Model',
    icon: CpuChipIcon,
  },
  {
    id: 'api',
    title: 'API Reference',
    icon: BeakerIcon,
  },
  {
    id: 'deployment',
    title: 'Deployment',
    icon: WrenchScrewdriverIcon,
  },
  {
    id: 'cli',
    title: 'CLI Tools',
    icon: CommandLineIcon,
  },
  {
    id: 'project-structure',
    title: 'Project Structure',
    icon: FolderIcon,
  },
  {
    id: 'contributing',
    title: 'Contributing',
    icon: DocumentTextIcon,
  },
];

const Documentation = () => {
  const [activeSection, setActiveSection] = useState('introduction');
  const [showBackToTop, setShowBackToTop] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      // Show back to top button when scrolled down 500px
      setShowBackToTop(window.scrollY > 500);

      // Update active section based on scroll position
      const sectionElements = sections.map(section => ({
        id: section.id,
        element: document.getElementById(section.id),
      }));

      const currentSection = sectionElements.find(({ element }) => {
        if (!element) return false;
        const rect = element.getBoundingClientRect();
        return rect.top <= 100 && rect.bottom >= 100;
      });

      if (currentSection) {
        setActiveSection(currentSection.id);
      }
    };

    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const scrollToTop = () => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const scrollToSection = (sectionId: string) => {
    const element = document.getElementById(sectionId);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth' });
    }
  };

  return (
    <div className="min-h-screen bg-dark-900">
      {/* Documentation Header */}
      <div className="h-16" /> {/* Navbar spacer */}
      <header className="bg-gradient-to-b from-dark-900 to-dark-800 border-b border-dark-700">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
          <motion.h1
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-4xl sm:text-5xl font-bold text-white mb-4"
          >
            Documentation
          </motion.h1>
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="text-xl text-gray-400 max-w-3xl"
          >
            Everything you need to know about the Movie Success Predictor project,
            from setup to deployment.
          </motion.p>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="lg:grid lg:grid-cols-12 lg:gap-8">
          {/* Sidebar Navigation */}
          <div className="hidden lg:block lg:col-span-3">
            <nav className="sticky top-20 space-y-1">
              {sections.map((section) => (
                <button
                  key={section.id}
                  onClick={() => scrollToSection(section.id)}
                  className={`w-full flex items-center px-3 py-2 text-sm font-medium rounded-lg transition-all duration-200 ${
                    activeSection === section.id
                      ? 'text-white bg-primary-500/10 border border-primary-500/20'
                      : 'text-gray-400 hover:text-white hover:bg-dark-800'
                  }`}
                >
                  <section.icon className="w-5 h-5 mr-2" />
                  {section.title}
                </button>
              ))}
            </nav>
          </div>

          {/* Main Content */}
          <main className="lg:col-span-9 prose prose-invert max-w-none">
            <section id="introduction" className="mb-16">
              <h2 className="flex items-center text-3xl font-bold text-white mb-6">
                <RocketLaunchIcon className="w-8 h-8 mr-3 text-primary-500" />
                Introduction
              </h2>
              <p className="text-gray-300 mb-6">
                The Movie Success Predictor is a sophisticated machine learning project that combines
                data science with web technology to predict whether a movie will be a hit or a flop.
                This project was developed as part of DATA 200 Final Project, demonstrating the
                practical application of machine learning in the entertainment industry.
              </p>
              <div className="bg-dark-800 rounded-lg p-6 border border-dark-700">
                <h3 className="text-xl font-semibold text-white mb-4">Key Features</h3>
                <ul className="space-y-3 text-gray-300">
                  <li className="flex items-start">
                    <span className="flex-shrink-0 w-1.5 h-1.5 rounded-full bg-primary-500 mt-2 mr-3" />
                    Interactive quiz interface for testing movie success predictions
                  </li>
                  <li className="flex items-start">
                    <span className="flex-shrink-0 w-1.5 h-1.5 rounded-full bg-primary-500 mt-2 mr-3" />
                    Machine learning model with 70.47% accuracy
                  </li>
                  <li className="flex items-start">
                    <span className="flex-shrink-0 w-1.5 h-1.5 rounded-full bg-primary-500 mt-2 mr-3" />
                    Analysis of 375,377 movies from 1970 to 2018
                  </li>
                  <li className="flex items-start">
                    <span className="flex-shrink-0 w-1.5 h-1.5 rounded-full bg-primary-500 mt-2 mr-3" />
                    Real-time integration with TMDB API
                  </li>
                </ul>
              </div>
            </section>

            <section id="getting-started" className="mb-16">
              <h2 className="flex items-center text-3xl font-bold text-white mb-6">
                <LightBulbIcon className="w-8 h-8 mr-3 text-primary-500" />
                Getting Started
              </h2>
              <div className="space-y-6">
                <div className="bg-dark-800 rounded-lg p-6 border border-dark-700">
                  <h3 className="text-xl font-semibold text-white mb-4">Prerequisites</h3>
                  <ul className="space-y-3 text-gray-300">
                    <li className="flex items-start">
                      <span className="flex-shrink-0 w-1.5 h-1.5 rounded-full bg-primary-500 mt-2 mr-3" />
                      Node.js (v18 or later)
                    </li>
                    <li className="flex items-start">
                      <span className="flex-shrink-0 w-1.5 h-1.5 rounded-full bg-primary-500 mt-2 mr-3" />
                      Python (3.8 or later)
                    </li>
                    <li className="flex items-start">
                      <span className="flex-shrink-0 w-1.5 h-1.5 rounded-full bg-primary-500 mt-2 mr-3" />
                      TMDB API key
                    </li>
                  </ul>
                </div>

                <div className="space-y-4">
                  <h3 className="text-xl font-semibold text-white">Frontend Setup</h3>
                  <div className="bg-dark-800/50 rounded-lg p-4 font-mono text-sm">
                    <p className="text-gray-300">cd frontend</p>
                    <p className="text-gray-300">npm install</p>
                    <p className="text-gray-300">npm run dev</p>
                  </div>
                </div>

                <div className="space-y-4">
                  <h3 className="text-xl font-semibold text-white">Backend Setup</h3>
                  <div className="bg-dark-800/50 rounded-lg p-4 font-mono text-sm">
                    <p className="text-gray-300">cd backend</p>
                    <p className="text-gray-300">python -m venv venv</p>
                    <p className="text-gray-300">source venv/bin/activate  # On Windows: venv\Scripts\activate</p>
                    <p className="text-gray-300">pip install -r requirements.txt</p>
                    <p className="text-gray-300">python app.py</p>
                  </div>
                </div>
              </div>
            </section>

            <section id="architecture" className="mb-16">
              <h2 className="flex items-center text-3xl font-bold text-white mb-6">
                <PuzzlePieceIcon className="w-8 h-8 mr-3 text-primary-500" />
                Architecture
              </h2>
              <div className="space-y-6">
                <p className="text-gray-300">
                  The Movie Success Predictor follows a modern, scalable architecture that separates
                  concerns between the frontend, backend, and machine learning components:
                </p>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  <div className="bg-dark-800 rounded-lg p-6 border border-dark-700">
                    <h3 className="text-xl font-semibold text-white mb-4">Frontend Layer</h3>
                    <ul className="space-y-2 text-gray-300">
                      <li>React + Vite</li>
                      <li>TypeScript</li>
                      <li>Tailwind CSS</li>
                      <li>Framer Motion</li>
                    </ul>
                  </div>
                  <div className="bg-dark-800 rounded-lg p-6 border border-dark-700">
                    <h3 className="text-xl font-semibold text-white mb-4">Backend Layer</h3>
                    <ul className="space-y-2 text-gray-300">
                      <li>Flask API</li>
                      <li>SQLite Database</li>
                      <li>TMDB Integration</li>
                      <li>JWT Authentication</li>
                    </ul>
                  </div>
                  <div className="bg-dark-800 rounded-lg p-6 border border-dark-700">
                    <h3 className="text-xl font-semibold text-white mb-4">ML Layer</h3>
                    <ul className="space-y-2 text-gray-300">
                      <li>scikit-learn</li>
                      <li>pandas</li>
                      <li>numpy</li>
                      <li>Feature Engineering</li>
                    </ul>
                  </div>
                </div>
              </div>
            </section>

            <section id="frontend" className="mb-16">
              <h2 className="flex items-center text-3xl font-bold text-white mb-6">
                <CodeBracketIcon className="w-8 h-8 mr-3 text-primary-500" />
                Frontend
              </h2>
              <div className="space-y-6">
                <p className="text-gray-300">
                  The frontend is built with React and TypeScript, utilizing modern web technologies
                  to create a responsive and interactive user interface.
                </p>
                <div className="bg-dark-800 rounded-lg p-6 border border-dark-700">
                  <h3 className="text-xl font-semibold text-white mb-4">Key Components</h3>
                  <div className="space-y-4">
                    <div>
                      <h4 className="text-lg font-medium text-white">MovieCard</h4>
                      <p className="text-gray-300">Displays movie information and poster</p>
                    </div>
                    <div>
                      <h4 className="text-lg font-medium text-white">QuizControls</h4>
                      <p className="text-gray-300">Handles user predictions</p>
                    </div>
                    <div>
                      <h4 className="text-lg font-medium text-white">ScoreDisplay</h4>
                      <p className="text-gray-300">Shows user performance</p>
                    </div>
                  </div>
                </div>
              </div>
            </section>

            <section id="backend" className="mb-16">
              <h2 className="flex items-center text-3xl font-bold text-white mb-6">
                <ServerIcon className="w-8 h-8 mr-3 text-primary-500" />
                Backend
              </h2>
              <div className="space-y-6">
                <p className="text-gray-300">
                  The backend is powered by Flask, providing RESTful APIs for the frontend and
                  handling machine learning model predictions.
                </p>
                <div className="bg-dark-800 rounded-lg p-6 border border-dark-700">
                  <h3 className="text-xl font-semibold text-white mb-4">API Endpoints</h3>
                  <div className="space-y-4">
                    <div className="p-4 bg-dark-900 rounded-lg">
                      <code className="text-primary-400">GET /api/quiz/next-movie</code>
                      <p className="text-gray-300 mt-2">Fetches the next movie for prediction</p>
                    </div>
                    <div className="p-4 bg-dark-900 rounded-lg">
                      <code className="text-primary-400">POST /api/quiz/submit-guess</code>
                      <p className="text-gray-300 mt-2">Submits user's prediction and returns result</p>
                    </div>
                  </div>
                </div>
              </div>
            </section>

            <section id="ml-model" className="mb-16">
              <h2 className="flex items-center text-3xl font-bold text-white mb-6">
                <CpuChipIcon className="w-8 h-8 mr-3 text-primary-500" />
                ML Model
              </h2>
              <div className="space-y-6">
                <p className="text-gray-300">
                  The machine learning model uses various algorithms to predict movie success,
                  trained on a dataset of over 375,000 movies.
                </p>
                <div className="bg-dark-800 rounded-lg p-6 border border-dark-700">
                  <h3 className="text-xl font-semibold text-white mb-4">Model Performance</h3>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                    <div className="p-4 bg-dark-900 rounded-lg">
                      <p className="text-lg font-medium text-white">Accuracy</p>
                      <p className="text-2xl font-bold text-primary-400">70.47%</p>
                    </div>
                    <div className="p-4 bg-dark-900 rounded-lg">
                      <p className="text-lg font-medium text-white">Training Data</p>
                      <p className="text-2xl font-bold text-primary-400">375,377 Movies</p>
                    </div>
                  </div>
                </div>
              </div>
            </section>

            <section id="api" className="mb-16">
              <h2 className="flex items-center text-3xl font-bold text-white mb-6">
                <BeakerIcon className="w-8 h-8 mr-3 text-primary-500" />
                API Reference
              </h2>
              
              <div className="space-y-8">
                {/* Introduction */}
                <div>
                  <p className="text-gray-300 mb-6">
                    The Movie Success Predictor API provides endpoints for movie prediction, quiz functionality,
                    and movie data retrieval. All endpoints are RESTful and return JSON responses.
                  </p>
                  
                  <div className="bg-dark-800 rounded-lg p-6 border border-dark-700 mb-8">
                    <h3 className="text-xl font-semibold text-white mb-4">Base URL</h3>
                    <code className="block bg-dark-900 p-3 rounded text-primary-400">
                      http://localhost:5001/api
                    </code>
                  </div>
                </div>

                {/* Authentication */}
                <div className="mb-8">
                  <h3 className="text-2xl font-bold text-white mb-4">Authentication</h3>
                  <div className="bg-dark-800 rounded-lg p-6 border border-dark-700">
                    <p className="text-gray-300 mb-4">
                      API requests require a valid TMDB API key to be included in the environment variables.
                      Set up your authentication by creating a <code className="text-primary-400">.env</code> file:
                    </p>
                    <code className="block bg-dark-900 p-3 rounded text-primary-400 mb-4">
                      TMDB_API_KEY=your_api_key_here
                    </code>
                    <div className="flex items-start mt-4">
                      <div className="flex-shrink-0 w-6 h-6 rounded-full bg-yellow-500/20 p-1 mr-3">
                        <svg className="w-4 h-4 text-yellow-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                        </svg>
                      </div>
                      <p className="text-yellow-500/90 text-sm">
                        Never commit your API key to version control. Always use environment variables.
                      </p>
                    </div>
                  </div>
                </div>

                {/* Endpoints */}
                <div>
                  <h3 className="text-2xl font-bold text-white mb-6">Endpoints</h3>
                  
                  {/* Get Next Movie */}
                  <div className="mb-8">
                    <div className="flex items-center space-x-3 mb-4">
                      <span className="bg-green-600/20 text-green-400 px-3 py-1 rounded-full text-sm font-mono">GET</span>
                      <code className="text-lg text-primary-400">/quiz/next-movie</code>
                    </div>
                    <div className="bg-dark-800 rounded-lg p-6 border border-dark-700">
                      <p className="text-gray-300 mb-4">
                        Retrieves the next movie for the quiz, with optional filtering by genre and certification.
                      </p>
                      
                      <h4 className="text-white font-semibold mb-2">Query Parameters</h4>
                      <div className="space-y-2 mb-4">
                        <p className="text-gray-300"><code className="text-primary-400">genre</code> (optional) - Filter by movie genre</p>
                        <p className="text-gray-300"><code className="text-primary-400">certification</code> (optional) - Filter by movie certification</p>
                      </div>

                      <h4 className="text-white font-semibold mb-2">Response</h4>
                      <code className="block bg-dark-900 p-3 rounded text-primary-400 mb-4">
{`{
  "id": "123",
  "title": "Movie Title",
  "posterUrl": "/path/to/poster.jpg",
  "year": 2023,
  "budget": 1000000,
  "runtime": 120,
  "genre": "Action",
  "certification": "PG-13"
}`}
                      </code>
                    </div>
                  </div>

                  {/* Submit Guess */}
                  <div className="mb-8">
                    <div className="flex items-center space-x-3 mb-4">
                      <span className="bg-blue-600/20 text-blue-400 px-3 py-1 rounded-full text-sm font-mono">POST</span>
                      <code className="text-lg text-primary-400">/quiz/submit-guess</code>
                    </div>
                    <div className="bg-dark-800 rounded-lg p-6 border border-dark-700">
                      <p className="text-gray-300 mb-4">
                        Submit a prediction for whether a movie will be a hit or flop.
                      </p>
                      
                      <h4 className="text-white font-semibold mb-2">Request Body</h4>
                      <code className="block bg-dark-900 p-3 rounded text-primary-400 mb-4">
{`{
  "movieId": "123",
  "guess": "Hit" | "Flop"
}`}
                      </code>

                      <h4 className="text-white font-semibold mb-2">Response</h4>
                      <code className="block bg-dark-900 p-3 rounded text-primary-400 mb-4">
{`{
  "isCorrect": true,
  "actualResult": "Hit",
  "feedbackMessage": "Correct! This movie was indeed a hit!",
  "confidence": 0.85
}`}
                      </code>
                    </div>
                  </div>

                  {/* Get Filter Options */}
                  <div className="mb-8">
                    <div className="flex items-center space-x-3 mb-4">
                      <span className="bg-green-600/20 text-green-400 px-3 py-1 rounded-full text-sm font-mono">GET</span>
                      <code className="text-lg text-primary-400">/quiz/filter-options</code>
                    </div>
                    <div className="bg-dark-800 rounded-lg p-6 border border-dark-700">
                      <p className="text-gray-300 mb-4">
                        Retrieves available filter options for the quiz.
                      </p>

                      <h4 className="text-white font-semibold mb-2">Response</h4>
                      <code className="block bg-dark-900 p-3 rounded text-primary-400 mb-4">
{`{
  "genres": ["Action", "Comedy", "Drama", ...],
  "certifications": ["G", "PG", "PG-13", "R"],
  "years": [1970, 1971, ..., 2023]
}`}
                      </code>
                    </div>
                  </div>

                  {/* Error Handling */}
                  <div className="mb-8">
                    <h3 className="text-2xl font-bold text-white mb-4">Error Handling</h3>
                    <div className="bg-dark-800 rounded-lg p-6 border border-dark-700">
                      <p className="text-gray-300 mb-4">
                        The API uses standard HTTP response codes and returns error messages in a consistent format:
                      </p>
                      <code className="block bg-dark-900 p-3 rounded text-primary-400 mb-4">
{`{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable error message",
    "details": { /* Additional error details */ }
  }
}`}
                      </code>
                      
                      <h4 className="text-white font-semibold mt-4 mb-2">Common Error Codes</h4>
                      <div className="space-y-2">
                        <div className="flex items-center">
                          <span className="w-16 text-red-400">400</span>
                          <span className="text-gray-300">Bad Request - Invalid parameters</span>
                        </div>
                        <div className="flex items-center">
                          <span className="w-16 text-red-400">401</span>
                          <span className="text-gray-300">Unauthorized - Invalid or missing API key</span>
                        </div>
                        <div className="flex items-center">
                          <span className="w-16 text-red-400">404</span>
                          <span className="text-gray-300">Not Found - Resource not found</span>
                        </div>
                        <div className="flex items-center">
                          <span className="w-16 text-red-400">429</span>
                          <span className="text-gray-300">Too Many Requests - Rate limit exceeded</span>
                        </div>
                        <div className="flex items-center">
                          <span className="w-16 text-red-400">500</span>
                          <span className="text-gray-300">Internal Server Error</span>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Rate Limiting */}
                  <div className="mb-8">
                    <h3 className="text-2xl font-bold text-white mb-4">Rate Limiting</h3>
                    <div className="bg-dark-800 rounded-lg p-6 border border-dark-700">
                      <p className="text-gray-300 mb-4">
                        The API implements rate limiting to ensure fair usage:
                      </p>
                      <ul className="space-y-2 text-gray-300">
                        <li>• 100 requests per minute per IP address</li>
                        <li>• 1000 requests per hour per IP address</li>
                      </ul>
                      <div className="flex items-start mt-4">
                        <div className="flex-shrink-0 w-6 h-6 rounded-full bg-blue-500/20 p-1 mr-3">
                          <svg className="w-4 h-4 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                          </svg>
                        </div>
                        <p className="text-blue-500/90 text-sm">
                          Rate limit headers are included in all responses to help track your usage.
                        </p>
                      </div>
                    </div>
                  </div>

                  {/* Example Usage */}
                  <div>
                    <h3 className="text-2xl font-bold text-white mb-4">Example Usage</h3>
                    <div className="bg-dark-800 rounded-lg p-6 border border-dark-700">
                      <p className="text-gray-300 mb-4">Here's a complete example of using the API with fetch:</p>
                      <code className="block bg-dark-900 p-3 rounded text-primary-400">
{`// Get next movie with filters
const response = await fetch('/api/quiz/next-movie?genre=Action&certification=PG-13');
const movie = await response.json();

// Submit a guess
const result = await fetch('/api/quiz/submit-guess', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    movieId: movie.id,
    guess: 'Hit'
  })
});
const feedback = await result.json();`}
                      </code>
                    </div>
                  </div>
                </div>
              </div>
            </section>

            <section id="deployment" className="mb-16">
              <h2 className="flex items-center text-3xl font-bold text-white mb-6">
                <WrenchScrewdriverIcon className="w-8 h-8 mr-3 text-primary-500" />
                Deployment
              </h2>
              <div className="space-y-6">
                <p className="text-gray-300 mb-6">
                  The Movie Success Predictor can be deployed to various platforms. Here are detailed instructions
                  for the most common deployment options.
                </p>

                {/* Docker Deployment */}
                <div className="bg-dark-800 rounded-lg p-6 border border-dark-700 mb-8">
                  <h3 className="text-xl font-semibold text-white mb-4">Docker Deployment</h3>
                  <p className="text-gray-300 mb-4">
                    The project includes Docker configuration for containerized deployment:
                  </p>
                  <code className="block bg-dark-900 p-3 rounded text-primary-400 mb-4">
{`# Build the images
docker-compose build

# Start the services
docker-compose up -d

# View logs
docker-compose logs -f`}
                  </code>
                  <div className="flex items-start mt-4">
                    <div className="flex-shrink-0 w-6 h-6 rounded-full bg-blue-500/20 p-1 mr-3">
                      <svg className="w-4 h-4 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                      </svg>
                    </div>
                    <p className="text-blue-500/90 text-sm">
                      Make sure to set environment variables in docker-compose.yml or use a .env file
                    </p>
                  </div>
                </div>

                {/* Cloud Platforms */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {/* AWS */}
                  <div className="bg-dark-800 rounded-lg p-6 border border-dark-700">
                    <h3 className="text-xl font-semibold text-white mb-4">AWS Deployment</h3>
                    <ol className="space-y-2 text-gray-300 list-decimal list-inside">
                      <li>Create an Elastic Beanstalk application</li>
                      <li>Configure environment variables</li>
                      <li>Deploy using the EB CLI:</li>
                    </ol>
                    <code className="block bg-dark-900 p-3 rounded text-primary-400 mt-3">
{`eb init movie-predictor
eb create production
eb deploy`}
                    </code>
                  </div>

                  {/* Heroku */}
                  <div className="bg-dark-800 rounded-lg p-6 border border-dark-700">
                    <h3 className="text-xl font-semibold text-white mb-4">Heroku Deployment</h3>
                    <ol className="space-y-2 text-gray-300 list-decimal list-inside">
                      <li>Install Heroku CLI</li>
                      <li>Login and create app</li>
                      <li>Deploy using Git:</li>
                    </ol>
                    <code className="block bg-dark-900 p-3 rounded text-primary-400 mt-3">
{`heroku login
heroku create
git push heroku main`}
                    </code>
                  </div>
                </div>

                {/* Environment Variables */}
                <div className="bg-dark-800 rounded-lg p-6 border border-dark-700">
                  <h3 className="text-xl font-semibold text-white mb-4">Environment Variables</h3>
                  <p className="text-gray-300 mb-4">
                    Required environment variables for deployment:
                  </p>
                  <div className="space-y-2">
                    <div className="flex items-start">
                      <code className="text-primary-400 w-48">TMDB_API_KEY</code>
                      <span className="text-gray-300">Your TMDB API key</span>
                    </div>
                    <div className="flex items-start">
                      <code className="text-primary-400 w-48">DATABASE_URL</code>
                      <span className="text-gray-300">Database connection string</span>
                    </div>
                    <div className="flex items-start">
                      <code className="text-primary-400 w-48">NODE_ENV</code>
                      <span className="text-gray-300">production/development</span>
                    </div>
                    <div className="flex items-start">
                      <code className="text-primary-400 w-48">PORT</code>
                      <span className="text-gray-300">Port number (default: 5001)</span>
                    </div>
                  </div>
                </div>
              </div>
            </section>

            <section id="cli" className="mb-16">
              <h2 className="flex items-center text-3xl font-bold text-white mb-6">
                <CommandLineIcon className="w-8 h-8 mr-3 text-primary-500" />
                CLI Tools
              </h2>
              <div className="space-y-6">
                <p className="text-gray-300 mb-6">
                  The project includes several command-line tools for data collection, model training,
                  and maintenance tasks.
                </p>

                {/* Data Collection */}
                <div className="bg-dark-800 rounded-lg p-6 border border-dark-700 mb-8">
                  <h3 className="text-xl font-semibold text-white mb-4">Data Collection</h3>
                  <p className="text-gray-300 mb-4">
                    Tools for collecting movie data from TMDB:
                  </p>
                  <code className="block bg-dark-900 p-3 rounded text-primary-400 mb-4">
{`# Scrape movies from TMDB
python scrape-movies.py

# Process raw data
python preprocessing.py`}
                  </code>
                  <p className="text-gray-300 mt-4">
                    The scraper collects movie data from 1970 to present, including budget, revenue,
                    and other features used for prediction.
                  </p>
                </div>

                {/* Model Training */}
                <div className="bg-dark-800 rounded-lg p-6 border border-dark-700 mb-8">
                  <h3 className="text-xl font-semibold text-white mb-4">Model Training</h3>
                  <p className="text-gray-300 mb-4">
                    Commands for training and evaluating the ML model:
                  </p>
                  <code className="block bg-dark-900 p-3 rounded text-primary-400 mb-4">
{`# Train the model
python classification.py

# Generate performance graphs
python graph.py`}
                  </code>
                </div>

                {/* Database Management */}
                <div className="bg-dark-800 rounded-lg p-6 border border-dark-700">
                  <h3 className="text-xl font-semibold text-white mb-4">Database Management</h3>
                  <p className="text-gray-300 mb-4">
                    Tools for managing the movie database:
                  </p>
                  <div className="space-y-4">
                    <code className="block bg-dark-900 p-3 rounded text-primary-400">
{`# Backup database
python tools/backup_db.py

# Update movie data
python tools/update_movies.py

# Clean database
python tools/clean_db.py`}
                    </code>
                  </div>
                </div>
              </div>
            </section>

            <section id="project-structure" className="mb-16">
              <h2 className="flex items-center text-3xl font-bold text-white mb-6">
                <FolderIcon className="w-8 h-8 mr-3 text-primary-500" />
                Project Structure
              </h2>
              <div className="space-y-6">
                <p className="text-gray-300 mb-6">
                  The project follows a modular structure separating frontend, backend, and ML components.
                </p>

                {/* Directory Structure */}
                <div className="bg-dark-800 rounded-lg p-6 border border-dark-700">
                  <h3 className="text-xl font-semibold text-white mb-4">Directory Structure</h3>
                  <code className="block bg-dark-900 p-3 rounded text-primary-400 whitespace-pre">
{`movie-success-predictor/
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── services/
│   │   ├── hooks/
│   │   └── types/
│   ├── public/
│   └── package.json
├── backend/
│   ├── app.py
│   ├── models/
│   ├── routes/
│   ├── services/
│   └── requirements.txt
├── ml/
│   ├── classification.py
│   ├── preprocessing.py
│   ├── graph.py
│   └── models/
├── tools/
│   ├── scrape-movies.py
│   └── update_movies.py
├── data/
│   └── moviesDb.csv
└── docker-compose.yml`}
                  </code>
                </div>

                {/* Key Files */}
                <div className="bg-dark-800 rounded-lg p-6 border border-dark-700">
                  <h3 className="text-xl font-semibold text-white mb-4">Key Files</h3>
                  <div className="space-y-4">
                    <div>
                      <h4 className="text-lg font-medium text-white">Frontend</h4>
                      <ul className="list-disc list-inside text-gray-300 ml-4">
                        <li>components/* - Reusable UI components</li>
                        <li>pages/* - Page components and routing</li>
                        <li>services/* - API integration and utilities</li>
                      </ul>
                    </div>
                    <div>
                      <h4 className="text-lg font-medium text-white">Backend</h4>
                      <ul className="list-disc list-inside text-gray-300 ml-4">
                        <li>app.py - Main Flask application</li>
                        <li>models/* - Database models</li>
                        <li>routes/* - API endpoint definitions</li>
                      </ul>
                    </div>
                    <div>
                      <h4 className="text-lg font-medium text-white">ML</h4>
                      <ul className="list-disc list-inside text-gray-300 ml-4">
                        <li>classification.py - Model training</li>
                        <li>preprocessing.py - Data preprocessing</li>
                        <li>graph.py - Visualization generation</li>
                      </ul>
                    </div>
                  </div>
                </div>
              </div>
            </section>

            <section id="contributing" className="mb-16">
              <h2 className="flex items-center text-3xl font-bold text-white mb-6">
                <DocumentTextIcon className="w-8 h-8 mr-3 text-primary-500" />
                Contributing
              </h2>
              <div className="space-y-6">
                <p className="text-gray-300 mb-6">
                  We welcome contributions to the Movie Success Predictor project! Here's how you can help.
                </p>

                {/* Getting Started */}
                <div className="bg-dark-800 rounded-lg p-6 border border-dark-700 mb-8">
                  <h3 className="text-xl font-semibold text-white mb-4">Getting Started</h3>
                  <ol className="space-y-2 text-gray-300 list-decimal list-inside">
                    <li>Fork the repository</li>
                    <li>Create a feature branch</li>
                    <li>Make your changes</li>
                    <li>Submit a pull request</li>
                  </ol>
                  <code className="block bg-dark-900 p-3 rounded text-primary-400 mt-4">
{`git clone https://github.com/your-username/movie-success-predictor.git
cd movie-success-predictor
git checkout -b feature/your-feature-name`}
                  </code>
                </div>

                {/* Development Guidelines */}
                <div className="bg-dark-800 rounded-lg p-6 border border-dark-700 mb-8">
                  <h3 className="text-xl font-semibold text-white mb-4">Development Guidelines</h3>
                  <ul className="space-y-3 text-gray-300">
                    <li className="flex items-start">
                      <span className="flex-shrink-0 w-1.5 h-1.5 rounded-full bg-primary-500 mt-2 mr-3" />
                      Follow the existing code style and conventions
                    </li>
                    <li className="flex items-start">
                      <span className="flex-shrink-0 w-1.5 h-1.5 rounded-full bg-primary-500 mt-2 mr-3" />
                      Write clear commit messages
                    </li>
                    <li className="flex items-start">
                      <span className="flex-shrink-0 w-1.5 h-1.5 rounded-full bg-primary-500 mt-2 mr-3" />
                      Add tests for new features
                    </li>
                    <li className="flex items-start">
                      <span className="flex-shrink-0 w-1.5 h-1.5 rounded-full bg-primary-500 mt-2 mr-3" />
                      Update documentation as needed
                    </li>
                  </ul>
                </div>

                {/* Code Review Process */}
                <div className="bg-dark-800 rounded-lg p-6 border border-dark-700">
                  <h3 className="text-xl font-semibold text-white mb-4">Code Review Process</h3>
                  <div className="space-y-4 text-gray-300">
                    <p>
                      All contributions go through our code review process:
                    </p>
                    <ul className="space-y-2 list-disc list-inside ml-4">
                      <li>Automated tests must pass</li>
                      <li>Code review by at least one maintainer</li>
                      <li>Documentation updates if applicable</li>
                      <li>Changes must be backward compatible</li>
                    </ul>
                    <div className="flex items-start mt-4">
                      <div className="flex-shrink-0 w-6 h-6 rounded-full bg-yellow-500/20 p-1 mr-3">
                        <svg className="w-4 h-4 text-yellow-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                        </svg>
                      </div>
                      <p className="text-yellow-500/90 text-sm">
                        Please ensure your PR description clearly explains the changes and motivation.
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            </section>
          </main>
        </div>
      </div>

      {/* Back to Top Button */}
      <motion.button
        initial={{ opacity: 0 }}
        animate={{ opacity: showBackToTop ? 1 : 0 }}
        onClick={scrollToTop}
        className={`fixed bottom-8 right-8 p-3 rounded-full bg-primary-500 text-white shadow-lg hover:bg-primary-400 transition-all duration-200 ${
          showBackToTop ? 'pointer-events-auto' : 'pointer-events-none'
        }`}
      >
        <svg
          className="w-6 h-6"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
          xmlns="http://www.w3.org/2000/svg"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M5 10l7-7m0 0l7 7m-7-7v18"
          />
        </svg>
      </motion.button>
    </div>
  );
};

export default Documentation; 