import { motion } from 'framer-motion';
import { FaGithub, FaLinkedin } from 'react-icons/fa';

const teamMembers = [
  {
    name: "Prashant Koirala",
    role: "Project Leader",
    image: "/team/prashant.jpg",
    bio: "Full-stack developer and machine learning enthusiast with a passion for building innovative solutions. Led the development of the movie success prediction model and the web application.",
    expertise: [
      "Machine Learning",
      "Full Stack Development",
      "Project Management",
      "Data Analysis"
    ],
    socials: {
      github: "https://github.com/prashantkoirala",
      linkedin: "https://linkedin.com/in/prashantkoirala"
    }
  },
  {
    name: "Aaska Koirala",
    role: "Data Scientist",
    image: "/team/aaska.jpg",
    bio: "Data scientist specializing in predictive modeling and statistical analysis. Responsible for feature engineering and model optimization.",
    expertise: [
      "Statistical Analysis",
      "Data Visualization",
      "Model Development",
      "Feature Engineering"
    ],
    socials: {
      github: "https://github.com/aaskakoirala",
      linkedin: "https://linkedin.com/in/aaskakoirala"
    }
  },
  {
    name: "Aishmita Yonzan",
    role: "Full Stack Developer",
    image: "/team/aishmita.jpg",
    bio: "Full-stack developer with expertise in modern web technologies. Led the frontend development and user experience design.",
    expertise: [
      "Frontend Development",
      "UI/UX Design",
      "API Development",
      "Web Performance"
    ],
    socials: {
      github: "https://github.com/aishmitayonzan",
      linkedin: "https://linkedin.com/in/aishmitayonzan"
    }
  }
];

const TeamPage = () => {
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
              Meet Our <span className="text-transparent bg-clip-text bg-gradient-to-r from-primary-400 to-primary-600">Team</span>
            </h1>
            <p className="text-xl text-gray-300 max-w-3xl mx-auto leading-relaxed">
              The talented individuals behind MoviePredictor, bringing together expertise in data science,
              machine learning, and web development.
            </p>
          </motion.div>
        </div>
      </section>

      {/* Team Members Section */}
      <section className="py-20 relative">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {teamMembers.map((member, index) => (
              <motion.div
                key={member.name}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: index * 0.2 }}
                viewport={{ once: true }}
                className="relative group"
              >
                {/* Card */}
                <div className="relative rounded-xl overflow-hidden bg-dark-800/50 border border-primary-500/10 hover:border-primary-500/20 transition-all duration-300">
                  {/* Image Container */}
                  <div className="relative aspect-[4/3] overflow-hidden">
                    <img
                      src={member.image}
                      alt={member.name}
                      className="w-full h-full object-cover transition-transform duration-300 group-hover:scale-105"
                    />
                    <div className="absolute inset-0 bg-gradient-to-t from-dark-900 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
                    
                    {/* Social Links Overlay */}
                    <div className="absolute inset-0 bg-dark-900/80 opacity-0 group-hover:opacity-100 transition-opacity duration-300 flex items-center justify-center gap-4">
                      <a
                        href={member.socials.github}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-white hover:text-primary-400 transform hover:scale-110 transition-all duration-200"
                      >
                        <FaGithub className="w-8 h-8" />
                      </a>
                      <a
                        href={member.socials.linkedin}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-white hover:text-primary-400 transform hover:scale-110 transition-all duration-200"
                      >
                        <FaLinkedin className="w-8 h-8" />
                      </a>
                    </div>
                  </div>

                  {/* Content */}
                  <div className="p-6">
                    <h3 className="text-xl font-bold text-white mb-1">{member.name}</h3>
                    <p className="text-primary-400 font-medium mb-4">{member.role}</p>
                    <p className="text-gray-300 mb-6 leading-relaxed">
                      {member.bio}
                    </p>
                    
                    {/* Expertise */}
                    <div className="space-y-2">
                      <h4 className="text-sm font-semibold text-white mb-2">Expertise</h4>
                      <div className="flex flex-wrap gap-2">
                        {member.expertise.map((skill, skillIndex) => (
                          <span
                            key={skillIndex}
                            className="px-3 py-1 text-sm bg-primary-500/10 text-primary-400 rounded-full border border-primary-500/20"
                          >
                            {skill}
                          </span>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Join Us Section */}
      <section className="py-20 relative">
        <div className="absolute inset-0 bg-gradient-to-b from-dark-900 via-primary-900/5 to-dark-900" />
        <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
            className="max-w-3xl mx-auto"
          >
            <h2 className="text-3xl sm:text-4xl font-bold text-white mb-6">
              Want to Contribute?
            </h2>
            <p className="text-lg text-gray-300 mb-8">
              We're always looking for talented individuals who are passionate about machine learning,
              web development, and the film industry. Join us in revolutionizing movie success prediction!
            </p>
            <a
              href="https://github.com/prashantkoirala/movie-success-predictor"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center px-6 py-3 border-2 border-primary-500 text-lg font-medium rounded-lg text-white bg-primary-500 hover:bg-primary-400 hover:border-primary-400 transition-all duration-300 shadow-lg hover:shadow-primary-500/20 transform hover:-translate-y-1"
            >
              <FaGithub className="w-5 h-5 mr-2" />
              View on GitHub
            </a>
          </motion.div>
        </div>
      </section>
    </div>
  );
};

export default TeamPage; 