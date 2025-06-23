import { motion } from 'framer-motion';
import {
  ChartBarIcon,
  CpuChipIcon,
  UserGroupIcon,
  PresentationChartLineIcon,
  AcademicCapIcon,
  SparklesIcon,
  ClockIcon,
  GlobeAltIcon,
  ShieldCheckIcon,
  CursorArrowRaysIcon
} from '@heroicons/react/24/outline';

const features = [
  {
    icon: CpuChipIcon,
    title: "Advanced ML Algorithm",
    description: "Our model achieves 70.47% accuracy in predicting movie success using sophisticated machine learning techniques.",
    details: [
      "Logistic Regression optimization",
      "Feature importance analysis",
      "Cross-validation techniques",
      "Continuous model improvement"
    ]
  },
  {
    icon: UserGroupIcon,
    title: "Interactive Experience",
    description: "Engage with our prediction model through an intuitive quiz interface that makes learning fun and interactive.",
    details: [
      "User-friendly interface",
      "Real-time feedback",
      "Score tracking",
      "Performance analytics"
    ]
  },
  {
    icon: PresentationChartLineIcon,
    title: "Data Visualization",
    description: "Understand movie success patterns through comprehensive data visualizations and statistical analysis.",
    details: [
      "Success rate trends",
      "Genre analysis",
      "Budget impact visualization",
      "Rating distribution"
    ]
  },
  {
    icon: GlobeAltIcon,
    title: "Global Coverage",
    description: "Access movie data from around the world, with comprehensive coverage of international film industries.",
    details: [
      "Multiple countries",
      "Various languages",
      "Different markets",
      "Cultural factors"
    ]
  },
  {
    icon: ClockIcon,
    title: "Historical Analysis",
    description: "Analyze movie success patterns across different time periods, from classic films to modern blockbusters.",
    details: [
      "Temporal analysis",
      "Trend identification",
      "Era comparisons",
      "Success evolution"
    ]
  },
  {
    icon: ShieldCheckIcon,
    title: "Reliable Predictions",
    description: "Trust in our thoroughly tested and validated prediction model, backed by extensive research.",
    details: [
      "Rigorous testing",
      "Validation protocols",
      "Error analysis",
      "Continuous updates"
    ]
  }
];

const FeaturesPage = () => {
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
              Powerful <span className="text-transparent bg-clip-text bg-gradient-to-r from-primary-400 to-primary-600">Features</span>
            </h1>
            <p className="text-xl text-gray-300 max-w-3xl mx-auto leading-relaxed">
              Explore the cutting-edge capabilities that make our movie success prediction platform unique and powerful.
            </p>
          </motion.div>
        </div>
      </section>

      {/* Features Grid */}
      <section className="py-20 relative">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {features.map((feature, index) => (
              <motion.div
                key={feature.title}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
                viewport={{ once: true }}
                className="relative p-8 rounded-xl bg-dark-800/50 border border-primary-500/10 hover:border-primary-500/20 transition-all duration-300 group"
              >
                <div className="absolute -top-4 left-6">
                  <span className="inline-flex items-center justify-center p-3 bg-gradient-to-br from-primary-500 to-primary-600 rounded-lg shadow-lg group-hover:shadow-primary-500/20 transition-all duration-300">
                    <feature.icon className="h-6 w-6 text-white" aria-hidden="true" />
                  </span>
                </div>
                <h3 className="mt-8 text-xl font-bold text-white mb-3 group-hover:text-primary-400 transition-colors duration-300">
                  {feature.title}
                </h3>
                <p className="text-gray-300 leading-relaxed mb-4">
                  {feature.description}
                </p>
                <ul className="space-y-2">
                  {feature.details.map((detail, detailIndex) => (
                    <li key={detailIndex} className="flex items-center text-gray-400">
                      <SparklesIcon className="h-4 w-4 mr-2 text-primary-400" />
                      <span>{detail}</span>
                    </li>
                  ))}
                </ul>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Interactive Demo Section */}
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
              Interactive Experience
            </h2>
            <p className="text-lg text-gray-300 max-w-3xl mx-auto">
              Experience our features firsthand through our interactive quiz interface.
            </p>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-8 max-w-4xl mx-auto">
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              whileInView={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.5 }}
              viewport={{ once: true }}
              className="bg-dark-800/50 p-8 rounded-xl border border-primary-500/10 hover:border-primary-500/20 transition-all duration-300"
            >
              <h3 className="text-xl font-bold text-white mb-4">How It Works</h3>
              <ol className="space-y-4">
                <li className="flex items-start">
                  <span className="flex items-center justify-center w-6 h-6 rounded-full bg-primary-500 text-white text-sm font-bold mr-3 mt-1">1</span>
                  <p className="text-gray-300">View movie details including poster, title, and relevant information</p>
                </li>
                <li className="flex items-start">
                  <span className="flex items-center justify-center w-6 h-6 rounded-full bg-primary-500 text-white text-sm font-bold mr-3 mt-1">2</span>
                  <p className="text-gray-300">Make your prediction: Hit or Flop</p>
                </li>
                <li className="flex items-start">
                  <span className="flex items-center justify-center w-6 h-6 rounded-full bg-primary-500 text-white text-sm font-bold mr-3 mt-1">3</span>
                  <p className="text-gray-300">Get instant feedback and compare with our AI's prediction</p>
                </li>
                <li className="flex items-start">
                  <span className="flex items-center justify-center w-6 h-6 rounded-full bg-primary-500 text-white text-sm font-bold mr-3 mt-1">4</span>
                  <p className="text-gray-300">Track your performance and improve your prediction skills</p>
                </li>
              </ol>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, x: 20 }}
              whileInView={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.5 }}
              viewport={{ once: true }}
              className="bg-dark-800/50 p-8 rounded-xl border border-primary-500/10 hover:border-primary-500/20 transition-all duration-300"
            >
              <h3 className="text-xl font-bold text-white mb-4">Key Benefits</h3>
              <ul className="space-y-4">
                <li className="flex items-center">
                  <CursorArrowRaysIcon className="h-5 w-5 mr-3 text-primary-400" />
                  <span className="text-gray-300">Learn from real movie data and market trends</span>
                </li>
                <li className="flex items-center">
                  <AcademicCapIcon className="h-5 w-5 mr-3 text-primary-400" />
                  <span className="text-gray-300">Understand key success factors in the film industry</span>
                </li>
                <li className="flex items-center">
                  <ChartBarIcon className="h-5 w-5 mr-3 text-primary-400" />
                  <span className="text-gray-300">Track your prediction accuracy over time</span>
                </li>
                <li className="flex items-center">
                  <SparklesIcon className="h-5 w-5 mr-3 text-primary-400" />
                  <span className="text-gray-300">Engage with an intuitive, user-friendly interface</span>
                </li>
              </ul>
            </motion.div>
          </div>
        </div>
      </section>
    </div>
  );
};

export default FeaturesPage; 