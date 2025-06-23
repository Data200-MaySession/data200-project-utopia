import { motion } from 'framer-motion';
import { ChartBarIcon, LightBulbIcon, BeakerIcon, BoltIcon, RocketLaunchIcon } from '@heroicons/react/24/outline';

const AboutPage = () => {
  return (
    <div className="min-h-screen bg-dark-900">
      {/* Hero Section */}
      <section className="relative pt-32 pb-20 overflow-hidden">
        <div className="absolute inset-0">
          <div className="absolute inset-0 bg-gradient-to-b from-primary-900/20 via-dark-900 to-dark-900" />
          <div className="absolute top-0 left-1/4 w-96 h-96 bg-primary-500/10 rounded-full filter blur-3xl animate-pulse" />
          <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-primary-500/10 rounded-full filter blur-3xl animate-pulse delay-1000" />
        </div>

        <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="text-center"
          >
            <h1 className="text-4xl sm:text-6xl font-bold text-white mb-6">
              About <span className="text-transparent bg-clip-text bg-gradient-to-r from-primary-400 to-primary-600">MoviePredictor</span>
            </h1>
            <p className="text-xl text-gray-300 max-w-3xl mx-auto leading-relaxed">
              Discover how we're revolutionizing movie success prediction through the power of machine learning and data science.
            </p>
          </motion.div>
        </div>
      </section>

      {/* Mission Section */}
      <section className="py-20 relative">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-3xl sm:text-4xl font-bold text-white mb-6">
              Our Mission
            </h2>
            <p className="text-lg text-gray-300 max-w-3xl mx-auto">
              To bridge the gap between intuition and data-driven decision making in the film industry,
              making movie success prediction accessible and engaging for everyone.
            </p>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {[
              {
                icon: LightBulbIcon,
                title: "Innovation",
                description: "Leveraging cutting-edge machine learning algorithms to analyze complex patterns in movie data."
              },
              {
                icon: ChartBarIcon,
                title: "Accuracy",
                description: "Achieving 70.47% prediction accuracy through rigorous model training and validation."
              },
              {
                icon: RocketLaunchIcon,
                title: "Accessibility",
                description: "Making sophisticated prediction models accessible through an intuitive, engaging interface."
              }
            ].map((item, index) => (
              <motion.div
                key={item.title}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: index * 0.2 }}
                viewport={{ once: true }}
                className="relative p-8 rounded-xl bg-dark-800/50 border border-primary-500/10 hover:border-primary-500/20 transition-all duration-300 group"
              >
                <div className="absolute -top-4 left-6">
                  <span className="inline-flex items-center justify-center p-3 bg-gradient-to-br from-primary-500 to-primary-600 rounded-lg shadow-lg group-hover:shadow-primary-500/20 transition-all duration-300">
                    <item.icon className="h-6 w-6 text-white" aria-hidden="true" />
                  </span>
                </div>
                <h3 className="mt-8 text-xl font-bold text-white mb-3 group-hover:text-primary-400 transition-colors duration-300">
                  {item.title}
                </h3>
                <p className="text-gray-300 leading-relaxed">
                  {item.description}
                </p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Technology Stack */}
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
            <h2 className="text-3xl sm:text-4xl font-bold text-white mb-6">
              Our Technology Stack
            </h2>
            <p className="text-lg text-gray-300 max-w-3xl mx-auto">
              Built with modern technologies to ensure reliability, scalability, and an exceptional user experience.
            </p>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-8 max-w-4xl mx-auto">
            <div className="bg-dark-800/50 p-8 rounded-xl border border-primary-500/10 hover:border-primary-500/20 transition-all duration-300">
              <h3 className="text-xl font-bold text-white mb-4">Frontend</h3>
              <ul className="space-y-3 text-gray-300">
                <li className="flex items-center">
                  <BoltIcon className="h-5 w-5 mr-2 text-primary-400" />
                  React with TypeScript for robust development
                </li>
                <li className="flex items-center">
                  <BoltIcon className="h-5 w-5 mr-2 text-primary-400" />
                  Tailwind CSS for modern, responsive design
                </li>
                <li className="flex items-center">
                  <BoltIcon className="h-5 w-5 mr-2 text-primary-400" />
                  Framer Motion for smooth animations
                </li>
              </ul>
            </div>
            <div className="bg-dark-800/50 p-8 rounded-xl border border-primary-500/10 hover:border-primary-500/20 transition-all duration-300">
              <h3 className="text-xl font-bold text-white mb-4">Backend</h3>
              <ul className="space-y-3 text-gray-300">
                <li className="flex items-center">
                  <BoltIcon className="h-5 w-5 mr-2 text-primary-400" />
                  Python with scikit-learn for ML models
                </li>
                <li className="flex items-center">
                  <BoltIcon className="h-5 w-5 mr-2 text-primary-400" />
                  FastAPI for high-performance API
                </li>
                <li className="flex items-center">
                  <BoltIcon className="h-5 w-5 mr-2 text-primary-400" />
                  TMDB API integration for movie data
                </li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      {/* Research & Development */}
      <section className="py-20 relative">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-3xl sm:text-4xl font-bold text-white mb-6">
              Research & Development
            </h2>
            <p className="text-lg text-gray-300 max-w-3xl mx-auto mb-12">
              Our journey of developing an accurate movie success prediction model involved extensive research and experimentation.
            </p>
          </motion.div>

          <div className="space-y-8 max-w-4xl mx-auto">
            {[
              {
                title: "Data Collection",
                description: "Gathered data from 375,377 movies dating from 1884 to 2018 using TMDB API, ensuring a comprehensive dataset for training."
              },
              {
                title: "Model Selection",
                description: "Tested multiple algorithms including Logistic Regression, KNN, Decision Trees, and Random Forest to find the optimal solution."
              },
              {
                title: "Feature Engineering",
                description: "Analyzed and processed key features including budget, runtime, genre, ratings, and production countries for optimal prediction."
              },
              {
                title: "Validation & Testing",
                description: "Implemented rigorous cross-validation techniques to ensure model reliability and prevent overfitting."
              }
            ].map((item, index) => (
              <motion.div
                key={item.title}
                initial={{ opacity: 0, x: index % 2 === 0 ? -20 : 20 }}
                whileInView={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.5 }}
                viewport={{ once: true }}
                className="bg-dark-800/50 p-8 rounded-xl border border-primary-500/10 hover:border-primary-500/20 transition-all duration-300"
              >
                <h3 className="text-xl font-bold text-white mb-3">{item.title}</h3>
                <p className="text-gray-300 leading-relaxed">{item.description}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>
    </div>
  );
};

export default AboutPage; 