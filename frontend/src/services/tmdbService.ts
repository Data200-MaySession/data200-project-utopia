import axios from 'axios';

const TMDB_API_KEY = '6ca18b9680392b11b867697feba84f55'; // <-- REPLACE WITH YOUR ACTUAL TMDB API KEY
const TMDB_BASE_URL = 'https://api.themoviedb.org/3';
const TMDB_IMAGE_BASE_URL = 'https://image.tmdb.org/t/p/';

const tmdbApi = axios.create({
  baseURL: TMDB_BASE_URL,
  params: {
    api_key: TMDB_API_KEY,
  },
});

export interface TmdbMovieData {
  id: number;
  title: string;
  original_title: string;
  overview: string;
  poster_path: string | null;
  backdrop_path: string | null;
  release_date: string;
  vote_average: number;
  vote_count: number;
  // Add more fields as needed from the TMDB API response
}

/**
 * Fetches movie details from TMDB by its ID.
 * @param movieId The TMDB ID of the movie.
 * @returns Promise<TmdbMovieData>
 */
export const getMovieDetails = async (movieId: number): Promise<TmdbMovieData> => {
  try {
    const response = await tmdbApi.get<TmdbMovieData>(`/movie/${movieId}`);
    return response.data;
  } catch (error) {
    console.error(`Error fetching movie details for ID ${movieId}:`, error);
    throw error; // Re-throw the error to be handled by the caller
  }
};

/**
 * Constructs the full URL for a TMDB movie poster.
 * @param posterPath The path of the poster (from TMDB API response).
 * @param size The desired size of the poster (e.g., 'w500'). Defaults to 'w500'.
 * @returns The full URL of the poster image, or null if posterPath is null.
 */
export const getPosterUrl = (posterPath: string | null, size: string = 'w500'): string | null => {
  if (!posterPath) {
    return null; // Or a placeholder image URL
  }
  return `${TMDB_IMAGE_BASE_URL}${size}${posterPath}`;
};

// Example function to get a random popular movie (you might want a more sophisticated way to get movies for the quiz)
/**
 * Fetches a list of popular movies from TMDB.
 * @returns Promise<TmdbMovieData[]>
 */
export const getPopularMovies = async (page: number = 1): Promise<TmdbMovieData[]> => {
  try {
    const response = await tmdbApi.get('/movie/popular', { params: { page } });
    return response.data.results as TmdbMovieData[];
  } catch (error) {
    console.error('Error fetching popular movies:', error);
    throw error;
  }
}; 