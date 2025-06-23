import { Routes, Route } from 'react-router-dom';
import { useEffect } from 'react';
import Navbar from './components/Navbar';
import Footer from './components/Footer';
import LandingPage from './pages/LandingPage';
import QuizPage from './pages/QuizPage';
import Documentation from './pages/Documentation';
import AboutPage from './pages/AboutPage';
import FeaturesPage from './pages/FeaturesPage';
import TeamPage from './pages/TeamPage';

// Import all weights of Bricolage Grotesque
import '@fontsource/bricolage-grotesque/300.css';  // Light
import '@fontsource/bricolage-grotesque/400.css';  // Regular
import '@fontsource/bricolage-grotesque/500.css';  // Medium
import '@fontsource/bricolage-grotesque/600.css';  // Semi-bold
import '@fontsource/bricolage-grotesque/700.css';  // Bold
import '@fontsource/bricolage-grotesque/800.css';  // Extra Bold

const App = () => {
  // Add class to body for global styles
  useEffect(() => {
    document.body.className = 'bg-dark-900 text-white antialiased';
  }, []);

  return (
    <div className="min-h-screen flex flex-col">
      <Navbar />
      <main className="flex-grow">
        <Routes>
          <Route path="/" element={<LandingPage />} />
          <Route path="/quiz" element={<QuizPage />} />
          <Route path="/docs" element={<Documentation />} />
          <Route path="/about" element={<AboutPage />} />
          <Route path="/features" element={<FeaturesPage />} />
          <Route path="/team" element={<TeamPage />} />
        </Routes>
      </main>
      <Footer />
    </div>
  );
};

export default App;
