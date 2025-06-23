import React from 'react';
import type { FilterOptions, GetMovieParams } from '../services/backendService';

interface FilterControlsProps {
  filterOptions: FilterOptions | null;
  selectedFilters: GetMovieParams;
  onFilterChange: (filterType: keyof GetMovieParams, value: string) => void;
  onApplyFilters: () => void;
  isLoading: boolean;
  isSetupMode?: boolean;
}

const FilterControls: React.FC<FilterControlsProps> = ({
  filterOptions,
  selectedFilters,
  onFilterChange,
  onApplyFilters,
  isLoading,
  isSetupMode = false,
}) => {
  if (!filterOptions) {
    return <div className="my-4 text-center text-slate-400">Loading filter options...</div>;
  }

  const handleSelectChange = (filterType: keyof GetMovieParams, e: React.ChangeEvent<HTMLSelectElement>) => {
    onFilterChange(filterType, e.target.value);
  };

  const selectStyle =
    'block w-full px-4 py-3 text-base text-slate-100 placeholder-slate-400 ' +
    'bg-slate-700/50 border border-slate-600 rounded-lg shadow-sm ' +
    'focus:outline-none focus:ring-2 focus:ring-sky-500 focus:border-sky-500 ' +
    'transition-colors duration-200 ease-in-out disabled:opacity-50 disabled:cursor-not-allowed';

  return (
    <div className={`p-6 my-8 bg-dark-800/60 backdrop-blur-md rounded-xl shadow-2xl border border-slate-700/50 w-full ${isSetupMode ? 'max-w-full' : 'max-w-lg mx-auto'}`}>
      {!isSetupMode && (
         <h3 className="text-2xl font-semibold text-sky-300 mb-6 text-center">Filter Movies</h3>
      )}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-x-6 gap-y-6 mb-6">
        {/* Genre Filter */}
        <div>
          <label htmlFor="genre-filter" className="block text-sm font-medium text-slate-300 mb-1.5">
            Genre
          </label>
          <select
            id="genre-filter"
            name="genre"
            value={selectedFilters.genre || ''}
            onChange={(e) => handleSelectChange('genre', e)}
            disabled={isLoading || filterOptions.genres.length === 0}
            className={selectStyle}
          >
            <option value="">All Genres</option>
            {filterOptions.genres.map((genre) => (
              <option key={genre} value={genre}>
                {genre}
              </option>
            ))}
          </select>
        </div>

        {/* Country Filter */}
        <div>
          <label htmlFor="country-filter" className="block text-sm font-medium text-slate-300 mb-1.5">
            Country
          </label>
          <select
            id="country-filter"
            name="country"
            value={selectedFilters.country || ''}
            onChange={(e) => handleSelectChange('country', e)}
            disabled={isLoading || filterOptions.countries.length === 0}
            className={selectStyle}
          >
            <option value="">All Countries</option>
            {filterOptions.countries.map((country) => (
              <option key={country} value={country}>
                {country}
              </option>
            ))}
          </select>
        </div>

        {/* Certification Filter */}
        <div>
          <label htmlFor="certification-filter" className="block text-sm font-medium text-slate-300 mb-1.5">
            Rating (US)
          </label>
          <select
            id="certification-filter"
            name="certification"
            value={selectedFilters.certification || ''}
            onChange={(e) => handleSelectChange('certification', e)}
            disabled={isLoading || filterOptions.certifications.length === 0}
            className={selectStyle}
          >
            <option value="">All Ratings</option>
            {filterOptions.certifications.map((cert) => (
              <option key={cert} value={cert}>
                {cert}
              </option>
            ))}
          </select>
        </div>
      </div>
      {!isSetupMode && (
        <button
          onClick={onApplyFilters}
          disabled={isLoading}
          className="w-full px-6 py-3 text-lg font-semibold text-white 
                     bg-sky-600 hover:bg-sky-500 
                     rounded-lg shadow-md hover:shadow-lg 
                     transition-all duration-300 ease-in-out 
                     transform focus:outline-none focus:ring-4 
                     focus:ring-sky-400 focus:ring-opacity-60 active:scale-95 
                     disabled:opacity-60 disabled:cursor-wait"
        >
          {isLoading ? 'Applying...' : 'Apply Filters & Get Movie'}
        </button>
      )}
    </div>
  );
};

export default FilterControls; 