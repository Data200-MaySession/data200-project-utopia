import axios from 'axios';

// Updated to point to your live Flask backend
const BACKEND_API_URL = 'http://127.0.0.1:5001/api'; 

const backendApi = axios.create({
  baseURL: BACKEND_API_URL,
});

export interface QuizMovieFromBackend {
  id: string; // TMDB ID, e.g., "299534" (backend sends it as string already)
  title: string;
  posterUrl: string | null; // This will be poster_path from backend, e.g., "/or06FN3Dka5tukK1e9sl16pB3iy.jpg"
}

export interface GuessResponse {
  movieId: string;
  userGuess: 'Hit' | 'Flop';
  prediction: 'Hit' | 'Flop';
  actualResult: 'Hit' | 'Flop';
  isCorrect: boolean;
  feedbackMessage: string;
}

export interface FilterOptions {
  genres: string[];
  countries: string[];
  certifications: string[];
}

export interface GetMovieParams {
  genre?: string;
  country?: string;
  certification?: string;
}

/**
 * Fetches available filter options (genres, countries, certifications) from the backend.
 */
export const getFilterOptions = async (): Promise<FilterOptions> => {
  console.log('Fetching filter options from backend...');
  try {
    const response = await backendApi.get<FilterOptions>('/quiz/filter-options');
    return response.data;
  } catch (error) {
    console.error('Error fetching filter options:', error);
    throw error; // Re-throw for QuizPage to handle
  }
};

/**
 * Fetches a movie for the user to guess, optionally applying filters.
 */
export const getMovieToGuess = async (params?: GetMovieParams): Promise<QuizMovieFromBackend> => {
  console.log('Fetching movie to guess from backend with params:', params);
  try {
    const response = await backendApi.get<QuizMovieFromBackend>('/quiz/next-movie', { params });
    return response.data;
  } catch (error) {
    console.error('Error fetching movie to guess from backend:', error);
    if (axios.isAxiosError(error) && !error.response) {
      throw new Error("Cannot connect to the movie quiz server. Please ensure it's running.");
    }
    if (axios.isAxiosError(error) && error.response && error.response.status === 404) {
      throw new Error(error.response.data.error || "No movies found matching your criteria. Try broadening your filters!");
    }
    throw error;
  }
};

/**
 * Submits the user's guess to the live backend and gets the result.
 * @param movieId The ID of the movie being guessed (e.g., "tmdb-299534" or just "299534").
 * @param guess The user's guess ('Hit' or 'Flop').
 */
export const submitGuess = async (
  movieId: string, // Frontend sends the ID as it has it (e.g. "299534")
  guess: 'Hit' | 'Flop'
): Promise<GuessResponse> => {
  console.log(`Submitting guess: ${guess} for movie ID: ${movieId} to live backend...`);
  try {
    const response = await backendApi.post<GuessResponse>('/quiz/submit-guess', { 
      movieId: movieId, // Backend expects this field name
      guess: guess      // Backend expects this field name
    });
    return response.data;
  } catch (error) {
    console.error('Error submitting guess to backend:', error);
    if (axios.isAxiosError(error) && !error.response) {
        throw new Error("Cannot connect to the movie quiz server to submit guess.");
    }
    throw error; // Re-throw for QuizPage to handle
  }
}; 