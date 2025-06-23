import React, { useEffect, useRef } from 'react';
import { Link } from 'react-router-dom';
import { motion, useScroll, useTransform } from 'framer-motion';
import {
  ChartBarIcon,
  LightBulbIcon,
  TrophyIcon,
  BeakerIcon,
  CodeBracketIcon,
  PresentationChartLineIcon,
  UserGroupIcon,
  AcademicCapIcon,
  RocketLaunchIcon,
  SparklesIcon,
  ChartPieIcon,
  CursorArrowRaysIcon,
} from '@heroicons/react/24/outline';

// Import social media icons
import {
  FaGithub,
  FaLinkedin,
} from 'react-icons/fa';

const features = [
  {
    name: 'AI-Powered Predictions',
    description: 'Our machine learning model analyzes multiple factors to predict movie success with high accuracy.',
    icon: LightBulbIcon,
  },
  {
    name: 'Test Your Intuition',
    description: 'Challenge yourself against our AI model and see if you can predict which movies will be hits.',
    icon: TrophyIcon,
  },
  {
    name: 'Learn From Data',
    description: 'Understand the patterns and factors that contribute to a movie\'s success in the box office.',
    icon: ChartBarIcon,
  },
];

const teamMembers = [
  {
    name: 'Prashant Koirala',
    role: 'Project Leader',
    description: 'Leading the development and implementation of the Movie Success Predictor project, coordinating team efforts and ensuring project milestones are met.',
    icon: RocketLaunchIcon,
    image: '/team/prashant.jpg', // Add your image path here
    socials: {
      github: 'https://github.com/prashantkoirala',
      linkedin: 'https://linkedin.com/in/prashantkoirala',
    },
  },
  {
    name: 'Aaska Koirala',
    role: 'Data Scientist',
    description: 'Specializing in data analysis and machine learning model development, focusing on improving prediction accuracy.',
    icon: BeakerIcon,
    image: '/team/aaska.jpg', // Add your image path here
    socials: {
      github: 'https://github.com/aaskakoirala',
      linkedin: 'https://linkedin.com/in/aaskakoirala',
    },
  },
  {
    name: 'Aishmita Yonzan',
    role: 'Full Stack Developer',
    description: 'Implementing the frontend and backend integration, ensuring seamless user experience and system functionality.',
    icon: CodeBracketIcon,
    image: '/team/aishmita.jpg', // Add your image path here
    socials: {
      github: 'https://github.com/aishmitayonzan',
      linkedin: 'https://linkedin.com/in/aishmitayonzan',
    },
  },
];

const stats = [
  { label: 'Model Accuracy', value: '70.47%', icon: ChartPieIcon },
  { label: 'Movies Analyzed', value: '375,377', icon: SparklesIcon },
  { label: 'Years of Data', value: '1970-2018', icon: CursorArrowRaysIcon },
];

const LandingPage = () => {
  const { scrollYProgress } = useScroll();
  const heroRef = useRef<HTMLDivElement>(null);
  
  const y = useTransform(scrollYProgress, [0, 0.5], ['0%', '50%']);
  const opacity = useTransform(scrollYProgress, [0, 0.5], [1, 0]);

  return (
    <div className="min-h-screen bg-dark-900">
      {/* Hero Section with Parallax */}
      <section className="relative h-screen overflow-hidden" ref={heroRef}>
        {/* Animated background elements */}
        <motion.div 
          className="absolute inset-0 z-0"
          style={{ y, opacity }}
        >
          <div className="absolute inset-0 bg-gradient-to-br from-primary-900/30 via-dark-900 to-dark-900" />
          <div className="absolute top-0 left-0 w-full h-full">
            <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-primary-500/10 rounded-full filter blur-3xl animate-pulse" />
            <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-primary-500/10 rounded-full filter blur-3xl animate-pulse delay-1000" />
          </div>
        </motion.div>
        
        {/* Hero Content */}
        <div className="relative z-10 h-full flex items-center">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-32">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8 }}
              className="text-center"
            >
              <h1 className="text-5xl sm:text-7xl font-bold text-white mb-6 tracking-tight">
                Predict Movie Success
                <span className="block text-transparent bg-clip-text bg-gradient-to-r from-primary-400 to-primary-600 mt-2">
                  With AI Precision
                </span>
              </h1>
              <p className="text-xl sm:text-2xl text-gray-300 max-w-3xl mx-auto mb-12 leading-relaxed">
                Experience the future of movie success prediction through our advanced AI model.
                Challenge your intuition against machine learning algorithms.
              </p>
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.5 }}
                className="flex flex-col sm:flex-row gap-4 justify-center"
              >
                <Link
                  to="/quiz"
                  className="inline-flex items-center px-8 py-4 border-2 border-primary-500 text-lg font-medium rounded-lg text-white bg-primary-500 hover:bg-primary-400 hover:border-primary-400 transition-all duration-300 shadow-lg hover:shadow-primary-500/20 transform hover:-translate-y-1"
                >
                  Start Predicting
                </Link>
                <a
                  href="#about"
                  className="inline-flex items-center px-8 py-4 border-2 border-primary-500 text-lg font-medium rounded-lg text-primary-400 hover:text-white hover:bg-primary-500 transition-all duration-300 transform hover:-translate-y-1"
                >
                  Learn More
                </a>
              </motion.div>
            </motion.div>
          </div>
        </div>

        {/* Scroll Indicator */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1 }}
          className="absolute bottom-8 left-1/2 transform -translate-x-1/2"
        >
          <div className="w-6 h-10 border-2 border-primary-500 rounded-full p-1">
            <div className="w-1.5 h-3 bg-primary-500 rounded-full animate-bounce mx-auto" />
          </div>
        </motion.div>
      </section>

      {/* Stats Section */}
      <section className="py-20 relative">
        <div className="absolute inset-0 bg-gradient-to-b from-dark-900 via-primary-900/5 to-dark-900" />
        <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {stats.map((stat, index) => (
              <motion.div
                key={stat.label}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: index * 0.2 }}
                viewport={{ once: true }}
                className="relative p-8 rounded-xl glass border border-primary-500/10 hover:border-primary-500/20 transition-all duration-300 group text-center"
              >
                <div className="absolute -top-4 left-1/2 transform -translate-x-1/2">
                  <span className="inline-flex items-center justify-center p-3 bg-gradient-to-br from-primary-500 to-primary-600 rounded-lg shadow-lg group-hover:shadow-primary-500/20 transition-all duration-300">
                    <stat.icon className="h-6 w-6 text-white" aria-hidden="true" />
                  </span>
                </div>
                <p className="mt-8 text-4xl font-bold text-white mb-2 group-hover:text-primary-400 transition-colors duration-300">
                  {stat.value}
                </p>
                <p className="text-gray-400 font-medium">
                  {stat.label}
                </p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* About Project Section */}
      <section id="about" className="py-20 relative">
        <div className="absolute inset-0 bg-gradient-to-b from-dark-900 via-primary-900/5 to-dark-900" />
        <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-3xl sm:text-5xl font-bold text-white mb-4">
              About the <span className="text-transparent bg-clip-text bg-gradient-to-r from-primary-400 to-primary-600">Project</span>
            </h2>
            <p className="text-gray-300 max-w-3xl mx-auto text-lg mb-12">
              The Movie Success Predictor is a DATA 200 Final Project that combines machine learning with 
              interactive web technology to create an engaging platform for predicting movie success.
            </p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8 max-w-4xl mx-auto">
              <div className="bg-dark-800/50 p-8 rounded-xl border border-primary-500/10 hover:border-primary-500/20 transition-all duration-300">
                <h3 className="text-xl font-bold text-white mb-4">Our Approach</h3>
                <p className="text-gray-300 text-md leading-relaxed">
                  Using advanced machine learning algorithms, we analyze various factors including budget, 
                  runtime, genre, and more to predict whether a movie will be a hit or a flop.
                </p>
              </div>
              <div className="bg-dark-800/50 p-8 rounded-xl border border-primary-500/10 hover:border-primary-500/20 transition-all duration-300">
                <h3 className="text-xl font-bold text-white mb-4">The Challenge</h3>
                <p className="text-gray-300 text-md leading-relaxed">
                  Test your movie industry knowledge against our AI model and learn about the factors 
                  that contribute to a movie's success in the box office.
                </p>
              </div>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Team Section */}
      <section className="py-20 relative">
        <div className="absolute inset-0 bg-gradient-to-b from-dark-900 via-primary-900/5 to-dark-900" />
        <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-3xl sm:text-5xl font-bold text-white mb-4">
              Meet the <span className="text-transparent bg-clip-text bg-gradient-to-r from-primary-400 to-primary-600">Team</span>
            </h2>
            <p className="text-gray-300 max-w-2xl mx-auto text-lg mb-12">
              Our dedicated team of students bringing together expertise in data science,
              machine learning, and web development.
            </p>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {teamMembers.map((member, index) => (
              <motion.div
                key={member.name}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: index * 0.2 }}
                viewport={{ once: true }}
                className="relative p-8 rounded-xl glass border border-primary-500/10 hover:border-primary-500/20 transition-all duration-300 group"
              >
                <div className="relative w-48 h-48 mx-auto mb-6 rounded-full overflow-hidden group-hover:ring-4 ring-primary-500/50 transition-all duration-300">
                  <img
                    src={member.image}
                    alt={member.name}
                    className="w-full h-full object-cover transition-transform duration-300 group-hover:scale-110"
                  />
                  <div className="absolute inset-0 bg-gradient-to-b from-transparent to-primary-900/90 opacity-0 group-hover:opacity-100 transition-opacity duration-300 flex items-end justify-center pb-4">
                    <div className="flex gap-4">
                      {member.socials.github && (
                        <a
                          href={member.socials.github}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-white hover:text-primary-400 transform hover:scale-110 transition-all duration-200"
                        >
                          <FaGithub className="w-6 h-6" />
                        </a>
                      )}
                      {member.socials.linkedin && (
                        <a
                          href={member.socials.linkedin}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-white hover:text-primary-400 transform hover:scale-110 transition-all duration-200"
                        >
                          <FaLinkedin className="w-6 h-6" />
                        </a>
                      )}
                    </div>
                  </div>
                </div>

                <div className="absolute -top-4 left-6">
                  <span className="inline-flex items-center justify-center p-3 bg-gradient-to-br from-primary-500 to-primary-600 rounded-lg shadow-lg group-hover:shadow-primary-500/20 transition-all duration-300">
                    <member.icon className="h-6 w-6 text-white" aria-hidden="true" />
                  </span>
                </div>

                <h3 className="text-xl font-bold text-white mb-2 group-hover:text-primary-400 transition-colors duration-300">
                  {member.name}
                </h3>
                <div className="text-primary-400 text-sm font-semibold mb-3">
                  {member.role}
                </div>
                <p className="text-gray-300 leading-relaxed">
                  {member.description}
                </p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-20 relative">
        <div className="absolute inset-0 bg-gradient-to-b from-dark-900 via-primary-900/5 to-dark-900" />
        <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-3xl sm:text-5xl font-bold text-white mb-4">
              Why Use <span className="text-transparent bg-clip-text bg-gradient-to-r from-primary-400 to-primary-600">MoviePredictor</span>?
            </h2>
            <p className="text-gray-300 max-w-2xl mx-auto text-lg">
              Our platform combines machine learning with your intuition to create
              an engaging experience in movie success prediction.
            </p>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {features.map((feature, index) => (
              <motion.div
                key={feature.name}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: index * 0.2 }}
                viewport={{ once: true }}
                className="relative p-8 rounded-xl glass border border-primary-500/10 hover:border-primary-500/20 transition-all duration-300 group"
              >
                <div className="absolute -top-4 left-6">
                  <span className="inline-flex items-center justify-center p-3 bg-gradient-to-br from-primary-500 to-primary-600 rounded-lg shadow-lg group-hover:shadow-primary-500/20 transition-all duration-300">
                    <feature.icon className="h-6 w-6 text-white" aria-hidden="true" />
                  </span>
                </div>
                <h3 className="mt-8 text-xl font-bold text-white mb-3 group-hover:text-primary-400 transition-colors duration-300">
                  {feature.name}
                </h3>
                <p className="text-gray-300 leading-relaxed">
                  {feature.description}
                </p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-t from-dark-900 via-primary-900/10 to-dark-900" />
        <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
            className="max-w-3xl mx-auto"
          >
            <h2 className="text-3xl sm:text-5xl font-bold text-white mb-6">
              Ready to Test Your Movie Prediction Skills?
            </h2>
            <p className="text-gray-300 text-lg mb-10">
              Join thousands of movie enthusiasts who are already challenging our AI
              and improving their understanding of the film industry.
            </p>
            <Link
              to="/quiz"
              className="inline-flex items-center px-8 py-4 border-2 border-primary-500 text-lg font-medium rounded-lg text-white bg-primary-500 hover:bg-primary-400 hover:border-primary-400 transition-all duration-300 shadow-lg hover:shadow-primary-500/20 transform hover:-translate-y-1"
            >
              Start the Quiz
            </Link>
          </motion.div>
        </div>
      </section>
    </div>
  );
};

export default LandingPage; 